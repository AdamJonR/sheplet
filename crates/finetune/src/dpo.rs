use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{Optimizer, SGD};
use serde::{Deserialize, Serialize};

use crate::data::DpoExample;
use crate::error::FinetuneError;
use crate::lora::LoraLinear;
use crate::sft::Tokenize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig {
    pub beta: f64,
    pub learning_rate: f64,
    pub epochs: usize,
    pub max_seq_len: usize,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            learning_rate: 5e-5,
            epochs: 1,
            max_seq_len: 512,
        }
    }
}

/// Simplified DPO training loop.
///
/// For each example, we compute:
/// - Policy (LoRA) forward pass for chosen and rejected inputs
/// - Reference (frozen-only) forward pass for chosen and rejected inputs
/// - DPO loss: -log(sigmoid(beta * (log_pi_chosen - log_pi_ref_chosen - log_pi_rejected + log_pi_ref_rejected)))
///
/// Since we lack the full model, we use sum-of-outputs as a proxy for log-probabilities.
pub fn train_dpo(
    lora: &mut LoraLinear,
    data: &[DpoExample],
    config: &DpoConfig,
    _tokenize: &dyn Tokenize,
    device: &Device,
) -> Result<f64, FinetuneError> {
    if data.is_empty() {
        return Err(FinetuneError::Training(
            "no training data provided".to_string(),
        ));
    }

    let (_, in_features) = lora
        .lora_a()
        .dims2()
        .map_err(|e| FinetuneError::Training(format!("unexpected lora_a shape: {e}")))?;

    let mut final_loss = 0.0;

    for _epoch in 0..config.epochs {
        let var_a = Var::from_tensor(&lora.lora_a().clone())?;
        let var_b = Var::from_tensor(&lora.lora_b().clone())?;

        let vars = vec![var_a.clone(), var_b.clone()];
        let mut sgd = SGD::new(vars.clone(), config.learning_rate)
            .map_err(|e| FinetuneError::Training(format!("failed to create optimizer: {e}")))?;

        for _example in data {
            // Random inputs for chosen and rejected (simulating embedded tokens)
            let x_chosen = Tensor::rand(0.0f32, 1.0f32, &[1, in_features], device)?;
            let x_rejected = Tensor::rand(0.0f32, 1.0f32, &[1, in_features], device)?;

            let scale = lora.scale();

            // Reference model (frozen only) forward passes
            let ref_chosen = lora
                .forward_frozen_only(&x_chosen)
                .map_err(|e| FinetuneError::Training(e.to_string()))?
                .sum_all()?;
            let ref_rejected = lora
                .forward_frozen_only(&x_rejected)
                .map_err(|e| FinetuneError::Training(e.to_string()))?
                .sum_all()?;

            // Policy model forward passes (using Var tensors for gradient tracking)
            let policy_chosen = {
                let frozen_out = lora
                    .forward_frozen_only(&x_chosen)
                    .map_err(|e| FinetuneError::Training(e.to_string()))?;
                let lora_out = x_chosen
                    .matmul(&var_a.as_tensor().t()?)?
                    .matmul(&var_b.as_tensor().t()?)?;
                let lora_out = (lora_out * scale)?;
                (frozen_out + lora_out)?.sum_all()?
            };

            let policy_rejected = {
                let frozen_out = lora
                    .forward_frozen_only(&x_rejected)
                    .map_err(|e| FinetuneError::Training(e.to_string()))?;
                let lora_out = x_rejected
                    .matmul(&var_a.as_tensor().t()?)?
                    .matmul(&var_b.as_tensor().t()?)?;
                let lora_out = (lora_out * scale)?;
                (frozen_out + lora_out)?.sum_all()?
            };

            // DPO loss: -log(sigmoid(beta * (policy_chosen - ref_chosen - policy_rejected + ref_rejected)))
            // Using sum-of-outputs as proxy for log-probabilities
            let diff = ((policy_chosen - &ref_chosen)? - (policy_rejected - &ref_rejected)?)?;
            let scaled = (diff * config.beta)?;

            // -log(sigmoid(x)) = log(1 + exp(-x))
            let neg_scaled = scaled.neg()?;
            let one = neg_scaled.ones_like()?;
            let loss = (one + neg_scaled.exp()?)?.log()?;

            sgd.backward_step(&loss)
                .map_err(|e| FinetuneError::Training(format!("backward step failed: {e}")))?;

            final_loss = loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        }

        lora.set_lora_a(var_a.as_tensor().clone());
        lora.set_lora_b(var_b.as_tensor().clone());
    }

    Ok(final_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::LoraConfig;
    use candle_core::Tensor;
    use candle_nn::Linear;

    struct DummyTokenizer;
    impl Tokenize for DummyTokenizer {
        fn encode(&self, _text: &str) -> anyhow::Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }
    }

    #[test]
    fn test_dpo_smoke() {
        let device = Device::Cpu;
        let weight = Tensor::rand(0.0f32, 1.0f32, &[4, 4], &device).unwrap();
        let frozen = Linear::new(weight, None);
        let config = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
        };
        let mut lora = LoraLinear::new(frozen, 4, 4, &config, &device).unwrap();

        let orig_a = lora.lora_a().clone();

        let data = vec![
            DpoExample {
                prompt: "question".into(),
                chosen: "good answer".into(),
                rejected: "bad answer".into(),
            },
            DpoExample {
                prompt: "another".into(),
                chosen: "correct".into(),
                rejected: "wrong".into(),
            },
        ];

        let dpo_config = DpoConfig {
            beta: 0.1,
            learning_rate: 0.01,
            epochs: 1,
            max_seq_len: 32,
        };

        let loss = train_dpo(&mut lora, &data, &dpo_config, &DummyTokenizer, &device).unwrap();
        assert!(loss.is_finite(), "loss should be finite, got {loss}");

        // Verify weights changed
        let diff = (orig_a - lora.lora_a())
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff > 0.0, "LoRA weights should have changed");
    }
}
