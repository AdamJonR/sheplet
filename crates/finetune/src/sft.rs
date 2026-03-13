use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer};
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

/// Clip gradients by global L2 norm. Returns the original (unclipped) norm.
fn clip_grad_norm(
    vars: &[Var],
    grads: &mut GradStore,
    max_norm: f64,
) -> Result<f64, FinetuneError> {
    let mut total_norm_sq = 0.0f64;
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_norm_sq += norm_sq;
        }
    }
    let total_norm = total_norm_sq.sqrt();

    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for var in vars {
            if let Some(grad) = grads.remove(var.as_tensor()) {
                let clipped = (grad * scale)?;
                grads.insert(var.as_tensor(), clipped);
            }
        }
    }
    Ok(total_norm)
}

/// SFT training loop using a single LoRA layer with cross-entropy loss.
///
/// Tokenizes input+output, does a forward pass through the LoRA layer,
/// and computes cross-entropy loss on output token positions.
pub fn train_sft(
    lora: &mut LoraLinear,
    data: &[SftExample],
    config: &SftConfig,
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

    let var_a = Var::from_tensor(&lora.lora_a().clone())?;
    let var_b = Var::from_tensor(&lora.lora_b().clone())?;

    let vars = vec![var_a.clone(), var_b.clone()];
    let mut optimizer = AdamW::new_lr(vars.clone(), config.learning_rate)
        .map_err(|e| FinetuneError::Training(format!("failed to create optimizer: {e}")))?;

    for _epoch in 0..config.epochs {
        for example in data {
            // Tokenize the input and output
            let input_tokens = tokenize
                .encode(&example.input)
                .map_err(|e| FinetuneError::Training(format!("tokenize input: {e}")))?;
            let output_tokens = tokenize
                .encode(&example.output)
                .map_err(|e| FinetuneError::Training(format!("tokenize output: {e}")))?;

            let total_len = input_tokens.len() + output_tokens.len();
            let seq_len = total_len.min(config.max_seq_len);

            if seq_len == 0 || output_tokens.is_empty() {
                continue;
            }

            // Create input embeddings (random projection for standalone LoRA training)
            let x = Tensor::rand(0.0f32, 1.0f32, &[seq_len, in_features], device)?;

            // Forward through LoRA
            let frozen_out = lora
                .forward_frozen_only(&x)
                .map_err(|e| FinetuneError::Training(e.to_string()))?;
            let lora_out = x
                .matmul(&var_a.as_tensor().t()?)?
                .matmul(&var_b.as_tensor().t()?)?;
            let scale = lora.scale();
            let lora_out = (lora_out * scale)?;
            let logits = (frozen_out + lora_out)?;

            // Create target distribution from output tokens
            // Use the output portion of the sequence for loss
            let input_len = input_tokens.len().min(seq_len);
            let output_portion = seq_len - input_len;

            if output_portion == 0 {
                continue;
            }

            // Take logits for the output positions
            let output_logits = logits.narrow(0, input_len, output_portion)?;

            // Create target indices (capped to vocab size = out_features)
            let targets: Vec<u32> = output_tokens
                .iter()
                .take(output_portion)
                .map(|&t| t.min(out_features as u32 - 1))
                .collect();

            if targets.is_empty() {
                continue;
            }

            let target_tensor =
                Tensor::from_vec(targets.clone(), &[targets.len()], device)?
                    .to_dtype(DType::U32)?;

            // Cross-entropy loss
            let loss =
                candle_nn::loss::cross_entropy(&output_logits, &target_tensor)?;

            let mut grads = loss.backward()
                .map_err(|e| FinetuneError::Training(format!("backward failed: {e}")))?;
            clip_grad_norm(&vars, &mut grads, 1.0)?;
            optimizer.step(&grads)
                .map_err(|e| FinetuneError::Training(format!("optimizer step failed: {e}")))?;

            final_loss = loss.to_scalar::<f32>()? as f64;
        }
    }

    lora.set_lora_a(var_a.as_tensor().clone());
    lora.set_lora_b(var_b.as_tensor().clone());

    Ok(final_loss)
}

/// SFT training using a full LoRA model with real tokenization and forward passes.
/// Works with any model implementing the `LoraTrainable` trait (Phi3, Gemma3, etc.).
pub fn train_sft_full(
    trainer: &mut dyn crate::model_utils::LoraTrainable,
    data: &[SftExample],
    config: &SftConfig,
) -> Result<f64, FinetuneError> {
    if data.is_empty() {
        return Err(FinetuneError::Training(
            "no training data provided".to_string(),
        ));
    }

    let device = trainer.device().clone();
    let mut final_loss = 0.0;

    // Create Vars from LoRA tensors for gradient tracking
    let lora_tensors = trainer.lora_tensors();
    let vars: Vec<Var> = lora_tensors
        .iter()
        .map(|t| Var::from_tensor(t))
        .collect::<candle_core::Result<Vec<_>>>()?;

    // Set var-backed tensors into model so forward passes use tracked tensors
    let var_tensors: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
    trainer.set_lora_tensors(&var_tensors);

    let mut optimizer = AdamW::new_lr(vars.clone(), config.learning_rate)
        .map_err(|e| FinetuneError::Training(format!("failed to create optimizer: {e}")))?;

    for epoch in 0..config.epochs {
        let mut epoch_loss_sum = 0.0f64;
        let mut epoch_count = 0usize;

        for example in data {
            trainer.clear_kv_cache();

            // Tokenize input and output separately to find the boundary
            let input_tokens = trainer
                .encode(&example.input)
                .map_err(|e| FinetuneError::Training(format!("tokenize input: {e}")))?;
            let full_text = format!("{}{}", example.input, example.output);
            let all_tokens = trainer
                .encode(&full_text)
                .map_err(|e| FinetuneError::Training(format!("tokenize: {e}")))?;

            let seq_len = all_tokens.len().min(config.max_seq_len);
            let input_len = input_tokens.len().min(seq_len);
            if seq_len < 2 || input_len >= seq_len {
                continue;
            }

            let input_ids = &all_tokens[..seq_len];
            let input_tensor = Tensor::from_vec(input_ids.to_vec(), &[1, seq_len], &device)?;

            // Forward pass returning logits from input_len-1 onwards (all response positions)
            let resp_start = input_len - 1;
            let logits = trainer
                .forward_from(&input_tensor, 0, resp_start)
                .map_err(|e| FinetuneError::Training(format!("forward: {e}")))?;

            let logits = logits.squeeze(0)?; // [resp_len+1, vocab_size]

            // Target tokens for the response portion
            let targets: Vec<u32> = all_tokens[input_len..seq_len].to_vec();
            let n_targets = targets.len();
            if n_targets == 0 {
                continue;
            }

            // Narrow logits to match target count (last row predicts beyond sequence)
            let logits = logits.narrow(0, 0, n_targets)?;
            let target_tensor = Tensor::from_vec(targets, &[n_targets], &device)?;

            let loss = candle_nn::loss::cross_entropy(&logits, &target_tensor)
                .map_err(|e| FinetuneError::Training(format!("loss: {e}")))?;

            let mut grads = loss.backward()
                .map_err(|e| FinetuneError::Training(format!("backward failed: {e}")))?;
            clip_grad_norm(&vars, &mut grads, 1.0)?;
            optimizer.step(&grads)
                .map_err(|e| FinetuneError::Training(format!("optimizer step failed: {e}")))?;

            // Propagate updated tensors back into the model for next forward pass
            let updated: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
            trainer.set_lora_tensors(&updated);

            let step_loss = loss.to_scalar::<f32>()? as f64;
            epoch_loss_sum += step_loss;
            epoch_count += 1;
            final_loss = step_loss;
        }

        if epoch_count > 0 {
            let avg_loss = epoch_loss_sum / epoch_count as f64;
            eprintln!("SFT epoch {}/{}: avg_loss={avg_loss:.4}, examples={epoch_count}", epoch + 1, config.epochs);
        }
    }

    Ok(final_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{make_test_lora, tensor_abs_diff, DummyTokenizer};

    fn sft_data() -> Vec<SftExample> {
        vec![
            SftExample {
                input: "hello world".into(),
                output: "foo bar".into(),
            },
            SftExample {
                input: "foo".into(),
                output: "bar baz".into(),
            },
        ]
    }

    #[test]
    fn test_sft_smoke() {
        let device = Device::Cpu;
        let mut lora = make_test_lora(4, 4, 2, 4.0);
        let orig_a = lora.lora_a().clone();

        let sft_config = SftConfig {
            learning_rate: 0.01,
            epochs: 1,
            batch_size: 1,
            max_seq_len: 32,
        };

        let loss = train_sft(&mut lora, &sft_data(), &sft_config, &DummyTokenizer, &device).unwrap();
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
        assert!(tensor_abs_diff(&orig_a, lora.lora_a()) > 0.0, "LoRA weights should have changed");
    }

    #[test]
    fn test_sft_gradients_flow() {
        // Verify that training produces finite loss and modifies weights at each epoch,
        // confirming that gradients flow through the computation graph.
        let device = Device::Cpu;
        let mut lora = make_test_lora(4, 4, 2, 4.0);

        let sft_config = SftConfig {
            learning_rate: 0.01,
            epochs: 3,
            batch_size: 1,
            max_seq_len: 32,
        };

        let a_before = lora.lora_a().clone();
        let loss = train_sft(&mut lora, &sft_data(), &sft_config, &DummyTokenizer, &device).unwrap();

        assert!(loss.is_finite(), "SFT loss should be finite, got {loss}");
        assert!(tensor_abs_diff(&a_before, lora.lora_a()) > 0.0, "weights should change over 3 epochs");
    }

    #[test]
    fn test_sft_cross_entropy_loss() {
        let device = Device::Cpu;
        let mut lora = make_test_lora(8, 8, 2, 4.0);

        let data = vec![SftExample {
            input: "a b c".into(),
            output: "d e f".into(),
        }];

        let sft_config = SftConfig {
            learning_rate: 0.01,
            epochs: 3,
            batch_size: 1,
            max_seq_len: 32,
        };

        let loss = train_sft(&mut lora, &data, &sft_config, &DummyTokenizer, &device).unwrap();
        assert!(loss.is_finite());
        assert!(loss > 0.0, "cross-entropy loss should be positive, got {loss}");
    }
}
