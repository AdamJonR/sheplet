use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{Optimizer, SGD};
use serde::{Deserialize, Serialize};

use crate::data::SftExample;
use crate::error::FinetuneError;
use crate::lora::LoraLinear;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftConfig {
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub max_seq_len: usize,
}

impl Default for SftConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            epochs: 3,
            batch_size: 1,
            max_seq_len: 512,
        }
    }
}

pub trait Tokenize: Send + Sync {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;
}

/// Simplified SFT training loop over a single LoRA layer.
///
/// Since we do not have the full model architecture, this creates random
/// input tensors (simulating embedded tokens) and trains the LoRA weights
/// to minimize a sum-of-squares loss.
pub fn train_sft(
    lora: &mut LoraLinear,
    data: &[SftExample],
    config: &SftConfig,
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
        // Create Var wrappers for gradient tracking
        let var_a = Var::from_tensor(&lora.lora_a().clone())?;
        let var_b = Var::from_tensor(&lora.lora_b().clone())?;

        let vars = vec![var_a.clone(), var_b.clone()];
        let mut sgd = SGD::new(vars.clone(), config.learning_rate)
            .map_err(|e| FinetuneError::Training(format!("failed to create optimizer: {e}")))?;

        for _example in data {
            // Create a random input tensor simulating embedded tokens
            let x = Tensor::rand(0.0f32, 1.0f32, &[1, in_features], device)?;

            // Forward: frozen_out + x @ A^T @ B^T * scale
            let frozen_out = lora
                .forward_frozen_only(&x)
                .map_err(|e| FinetuneError::Training(e.to_string()))?;
            let lora_out = x
                .matmul(&var_a.as_tensor().t()?)?
                .matmul(&var_b.as_tensor().t()?)?;
            let scale = lora.scale();
            let lora_out = (lora_out * scale)?;
            let output = (frozen_out + lora_out)?;

            // Sum-of-squares loss
            let loss = output.sqr()?.sum_all()?;

            sgd.backward_step(&loss)
                .map_err(|e| FinetuneError::Training(format!("backward step failed: {e}")))?;

            final_loss = loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        }

        // Update LoRA weights from Vars
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
    fn test_sft_smoke() {
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
            SftExample {
                input: "hello".into(),
                output: "world".into(),
            },
            SftExample {
                input: "foo".into(),
                output: "bar".into(),
            },
        ];

        let sft_config = SftConfig {
            learning_rate: 0.01,
            epochs: 1,
            batch_size: 1,
            max_seq_len: 32,
        };

        let loss = train_sft(&mut lora, &data, &sft_config, &DummyTokenizer, &device).unwrap();
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
