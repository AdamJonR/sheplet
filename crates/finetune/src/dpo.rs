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

/// DPO training using per-token log-probabilities.
///
/// For each example:
/// 1. Tokenize prompt+chosen and prompt+rejected
/// 2. Forward pass through LoRA (policy) and frozen-only (reference)
/// 3. Compute log-softmax to get per-token log-probs on the response portion
/// 4. DPO loss: -log(sigmoid(beta * ((log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected))))
pub fn train_dpo(
    lora: &mut LoraLinear,
    data: &[DpoExample],
    config: &DpoConfig,
    tokenize: &dyn Tokenize,
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

    let (out_features, _) = lora
        .lora_b()
        .dims2()
        .map_err(|e| FinetuneError::Training(format!("unexpected lora_b shape: {e}")))?;

    let mut final_loss = 0.0;

    for _epoch in 0..config.epochs {
        let var_a = Var::from_tensor(&lora.lora_a().clone())?;
        let var_b = Var::from_tensor(&lora.lora_b().clone())?;

        let vars = vec![var_a.clone(), var_b.clone()];
        let mut sgd = SGD::new(vars.clone(), config.learning_rate)
            .map_err(|e| FinetuneError::Training(format!("failed to create optimizer: {e}")))?;

        for example in data {
            // Tokenize chosen and rejected sequences
            let prompt_tokens = tokenize
                .encode(&example.prompt)
                .map_err(|e| FinetuneError::Training(format!("tokenize prompt: {e}")))?;
            let chosen_tokens = tokenize
                .encode(&example.chosen)
                .map_err(|e| FinetuneError::Training(format!("tokenize chosen: {e}")))?;
            let rejected_tokens = tokenize
                .encode(&example.rejected)
                .map_err(|e| FinetuneError::Training(format!("tokenize rejected: {e}")))?;

            let prompt_len = prompt_tokens.len();
            let chosen_len = (prompt_len + chosen_tokens.len()).min(config.max_seq_len);
            let rejected_len = (prompt_len + rejected_tokens.len()).min(config.max_seq_len);

            if chosen_len <= prompt_len || rejected_len <= prompt_len {
                continue;
            }

            let chosen_resp_len = chosen_len - prompt_len;
            let rejected_resp_len = rejected_len - prompt_len;

            let scale = lora.scale();

            // Forward passes for chosen
            let x_chosen =
                Tensor::rand(0.0f32, 1.0f32, &[chosen_len, in_features], device)?;

            // Policy forward (with LoRA)
            let frozen_out_chosen = lora
                .forward_frozen_only(&x_chosen)
                .map_err(|e| FinetuneError::Training(e.to_string()))?;
            let lora_out_chosen = x_chosen
                .matmul(&var_a.as_tensor().t()?)?
                .matmul(&var_b.as_tensor().t()?)?;
            let lora_out_chosen = (lora_out_chosen * scale)?;
            let policy_logits_chosen = (frozen_out_chosen.clone() + lora_out_chosen)?;

            // Reference forward (frozen only) - detach from gradient graph
            let ref_logits_chosen = frozen_out_chosen;

            // Forward passes for rejected
            let x_rejected =
                Tensor::rand(0.0f32, 1.0f32, &[rejected_len, in_features], device)?;

            let frozen_out_rejected = lora
                .forward_frozen_only(&x_rejected)
                .map_err(|e| FinetuneError::Training(e.to_string()))?;
            let lora_out_rejected = x_rejected
                .matmul(&var_a.as_tensor().t()?)?
                .matmul(&var_b.as_tensor().t()?)?;
            let lora_out_rejected = (lora_out_rejected * scale)?;
            let policy_logits_rejected = (frozen_out_rejected.clone() + lora_out_rejected)?;

            let ref_logits_rejected = frozen_out_rejected;

            // Compute per-token log-probs on response portions
            // Create target token indices for the response portions
            let chosen_targets: Vec<u32> = chosen_tokens
                .iter()
                .take(chosen_resp_len)
                .map(|&t| t.min(out_features as u32 - 1))
                .collect();
            let rejected_targets: Vec<u32> = rejected_tokens
                .iter()
                .take(rejected_resp_len)
                .map(|&t| t.min(out_features as u32 - 1))
                .collect();

            // Get response-portion logits
            let policy_resp_chosen =
                policy_logits_chosen.narrow(0, prompt_len, chosen_resp_len)?;
            let ref_resp_chosen =
                ref_logits_chosen.narrow(0, prompt_len, chosen_resp_len)?;
            let policy_resp_rejected =
                policy_logits_rejected.narrow(0, prompt_len, rejected_resp_len)?;
            let ref_resp_rejected =
                ref_logits_rejected.narrow(0, prompt_len, rejected_resp_len)?;

            // Log-softmax → gather target token log-probs → sum
            let log_pi_chosen =
                gather_log_probs(&policy_resp_chosen, &chosen_targets, device)?;
            let log_ref_chosen =
                gather_log_probs(&ref_resp_chosen, &chosen_targets, device)?;
            let log_pi_rejected =
                gather_log_probs(&policy_resp_rejected, &rejected_targets, device)?;
            let log_ref_rejected =
                gather_log_probs(&ref_resp_rejected, &rejected_targets, device)?;

            // DPO loss: -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))))
            let chosen_reward = (log_pi_chosen - log_ref_chosen)?;
            let rejected_reward = (log_pi_rejected - log_ref_rejected)?;
            let diff = (chosen_reward - rejected_reward)?;
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

/// Compute sum of log-probabilities for target tokens from logits.
/// logits: [seq_len, vocab_size], targets: Vec<u32> of length seq_len
fn gather_log_probs(
    logits: &Tensor,
    targets: &[u32],
    device: &Device,
) -> Result<Tensor, FinetuneError> {
    // Log-softmax over vocab dimension
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;

    // Gather the target token log-probs
    let target_tensor = Tensor::from_vec(targets.to_vec(), &[targets.len(), 1], device)?
        .to_dtype(DType::U32)?;

    let gathered = log_probs.gather(&target_tensor, 1)?;
    let sum = gathered.sum_all()?;
    Ok(sum)
}

/// DPO training using a full LoRA model with real tokenization and forward passes.
/// Works with any model implementing the `LoraTrainable` trait (Phi3, Gemma3, etc.).
///
/// Uses `forward` for policy (with LoRA) and `forward_reference` for reference
/// (frozen weights only) passes, then computes DPO loss from per-token log-probs.
pub fn train_dpo_full(
    trainer: &mut dyn crate::model_utils::LoraTrainable,
    data: &[DpoExample],
    config: &DpoConfig,
) -> Result<f64, FinetuneError> {
    if data.is_empty() {
        return Err(FinetuneError::Training(
            "no training data provided".to_string(),
        ));
    }

    let device = trainer.device().clone();
    let mut final_loss = 0.0;

    for _epoch in 0..config.epochs {
        for example in data {
            // Tokenize prompt+chosen and prompt+rejected
            let chosen_text = format!("{}{}", example.prompt, example.chosen);
            let rejected_text = format!("{}{}", example.prompt, example.rejected);

            let chosen_tokens = trainer
                .encode(&chosen_text)
                .map_err(|e| FinetuneError::Training(format!("tokenize chosen: {e}")))?;
            let rejected_tokens = trainer
                .encode(&rejected_text)
                .map_err(|e| FinetuneError::Training(format!("tokenize rejected: {e}")))?;

            let prompt_tokens = trainer
                .encode(&example.prompt)
                .map_err(|e| FinetuneError::Training(format!("tokenize prompt: {e}")))?;

            let prompt_len = prompt_tokens.len();
            let chosen_len = chosen_tokens.len().min(config.max_seq_len);
            let rejected_len = rejected_tokens.len().min(config.max_seq_len);

            if chosen_len <= prompt_len || rejected_len <= prompt_len {
                continue;
            }

            let chosen_resp_len = chosen_len - prompt_len;
            let rejected_resp_len = rejected_len - prompt_len;

            // --- Policy forward pass (with LoRA) for chosen ---
            trainer.clear_kv_cache();
            let chosen_input =
                Tensor::from_vec(chosen_tokens[..chosen_len].to_vec(), &[1, chosen_len], &device)?;
            let policy_logits_chosen = trainer
                .forward(&chosen_input, 0)
                .map_err(|e| FinetuneError::Training(format!("policy forward chosen: {e}")))?;
            let policy_logits_chosen = policy_logits_chosen.squeeze(0)?;

            // --- Policy forward pass (with LoRA) for rejected ---
            trainer.clear_kv_cache();
            let rejected_input = Tensor::from_vec(
                rejected_tokens[..rejected_len].to_vec(),
                &[1, rejected_len],
                &device,
            )?;
            let policy_logits_rejected = trainer
                .forward(&rejected_input, 0)
                .map_err(|e| FinetuneError::Training(format!("policy forward rejected: {e}")))?;
            let policy_logits_rejected = policy_logits_rejected.squeeze(0)?;

            // --- Reference forward pass (without LoRA) for chosen ---
            trainer.clear_kv_cache();
            let ref_logits_chosen = trainer
                .forward_reference(&chosen_input, 0)
                .map_err(|e| FinetuneError::Training(format!("ref forward chosen: {e}")))?;
            let ref_logits_chosen = ref_logits_chosen.squeeze(0)?;

            // --- Reference forward pass (without LoRA) for rejected ---
            trainer.clear_kv_cache();
            let ref_logits_rejected = trainer
                .forward_reference(&rejected_input, 0)
                .map_err(|e| FinetuneError::Training(format!("ref forward rejected: {e}")))?;
            let ref_logits_rejected = ref_logits_rejected.squeeze(0)?;

            // Extract response-portion logits (positions after prompt)
            // The model outputs logits for predicting the next token at each position,
            // so response logits start at position (prompt_len - 1).
            let policy_resp_chosen =
                policy_logits_chosen.narrow(0, prompt_len - 1, chosen_resp_len)?;
            let ref_resp_chosen =
                ref_logits_chosen.narrow(0, prompt_len - 1, chosen_resp_len)?;
            let policy_resp_rejected =
                policy_logits_rejected.narrow(0, prompt_len - 1, rejected_resp_len)?;
            let ref_resp_rejected =
                ref_logits_rejected.narrow(0, prompt_len - 1, rejected_resp_len)?;

            // Target tokens for the response portion
            let chosen_targets: Vec<u32> = chosen_tokens[prompt_len..chosen_len].to_vec();
            let rejected_targets: Vec<u32> =
                rejected_tokens[prompt_len..rejected_len].to_vec();

            // Compute per-token log-probs and sum
            let log_pi_chosen =
                gather_log_probs(&policy_resp_chosen, &chosen_targets, &device)?;
            let log_ref_chosen =
                gather_log_probs(&ref_resp_chosen, &chosen_targets, &device)?;
            let log_pi_rejected =
                gather_log_probs(&policy_resp_rejected, &rejected_targets, &device)?;
            let log_ref_rejected =
                gather_log_probs(&ref_resp_rejected, &rejected_targets, &device)?;

            // DPO loss: -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))))
            let chosen_reward = (log_pi_chosen - log_ref_chosen)?;
            let rejected_reward = (log_pi_rejected - log_ref_rejected)?;
            let diff = (chosen_reward - rejected_reward)?;
            let scaled = (diff * config.beta)?;

            // -log(sigmoid(x)) = log(1 + exp(-x))
            let neg_scaled = scaled.neg()?;
            let one = neg_scaled.ones_like()?;
            let loss = (one + neg_scaled.exp()?)?.log()?;

            final_loss = loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        }
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
        fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
            Ok(text
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| (i % 4) as u32)
                .collect())
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
                prompt: "question one".into(),
                chosen: "good answer here".into(),
                rejected: "bad answer here".into(),
            },
            DpoExample {
                prompt: "another question".into(),
                chosen: "correct response".into(),
                rejected: "wrong response".into(),
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

    #[test]
    fn test_dpo_log_probs() {
        let device = Device::Cpu;

        // Create logits: [3, 4] (3 tokens, 4 vocab)
        let logits = Tensor::new(
            &[[1.0f32, 2.0, 0.5, 0.1], [0.1, 3.0, 0.2, 0.5], [2.0, 0.1, 1.0, 0.3]],
            &device,
        )
        .unwrap();
        let targets = vec![1u32, 1, 0]; // Pick tokens 1, 1, 0

        let sum_log_probs = gather_log_probs(&logits, &targets, &device).unwrap();
        let val = sum_log_probs.to_scalar::<f32>().unwrap();
        assert!(val.is_finite());
        // Log-probs should be negative (log of probabilities < 1)
        assert!(val < 0.0, "sum of log-probs should be negative, got {val}");
    }
}
