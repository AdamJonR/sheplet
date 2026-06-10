use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer};
use serde::{Deserialize, Serialize};

use crate::data::DpoExample;
use crate::error::FinetuneError;
use crate::lora::LoraLinear;
use crate::model_utils::check_gradient_flow;
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
            epochs: 3,
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

    let var_a = Var::from_tensor(&lora.lora_a().clone())?;
    let var_b = Var::from_tensor(&lora.lora_b().clone())?;

    let vars = vec![var_a.clone(), var_b.clone()];
    let mut optimizer = AdamW::new_lr(vars.clone(), config.learning_rate)
        .map_err(|e| FinetuneError::Training(format!("failed to create optimizer: {e}")))?;

    for _epoch in 0..config.epochs {
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

            let loss = dpo_loss(
                log_pi_chosen, log_ref_chosen, log_pi_rejected, log_ref_rejected, config.beta,
            )?;

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

/// Compute DPO loss: -log(sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))))
/// Equivalent to softplus(-beta * reward_diff), computed in the numerically
/// stable form max(z, 0) + log(1 + exp(-|z|)) so large reward margins don't
/// overflow exp() to inf.
fn dpo_loss(
    log_pi_chosen: Tensor,
    log_ref_chosen: Tensor,
    log_pi_rejected: Tensor,
    log_ref_rejected: Tensor,
    beta: f64,
) -> Result<Tensor, FinetuneError> {
    let chosen_reward = (log_pi_chosen - log_ref_chosen)?;
    let rejected_reward = (log_pi_rejected - log_ref_rejected)?;
    let diff = (chosen_reward - rejected_reward)?;
    let z = (diff * (-beta))?;
    let softplus = (z.relu()? + ((z.abs()?.neg()?.exp()? + 1.0)?).log()?)?;
    Ok(softplus)
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

/// Compute the sum of log-probabilities for target tokens from logits.
/// Standard DPO uses the summed sequence log-prob; averaging would divide the
/// implicit reward by response length, shrinking the effective beta and
/// confounding preferences with length.
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
/// Works with any model implementing the `LoraTrainable` trait (Phi3, Llama, etc.).
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

    // Create Vars from LoRA tensors for gradient tracking
    let lora_tensors = trainer.lora_tensors();
    let vars: Vec<Var> = lora_tensors
        .iter()
        .map(Var::from_tensor)
        .collect::<candle_core::Result<Vec<_>>>()?;

    // Set var-backed tensors into model so forward passes use tracked tensors
    let var_tensors: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
    trainer.set_lora_tensors(&var_tensors);

    // LoRA convention: no weight decay on adapter params (candle's default is 0.01)
    let mut optimizer = AdamW::new(
        vars.clone(),
        candle_nn::ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
    .map_err(|e| FinetuneError::Training(format!("failed to create optimizer: {e}")))?;

    let mut order: Vec<usize> = (0..data.len()).collect();
    for epoch in 0..config.epochs {
        // Deterministic per-epoch shuffle so example order doesn't bias training
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(epoch as u64);
        order.shuffle(&mut rng);

        let mut epoch_loss_sum = 0.0f64;
        let mut epoch_count = 0usize;

        for &example_idx in &order {
            let example = &data[example_idx];
            // Tokenize prompt and responses separately and concatenate the ids.
            // This matches inference exactly: the prompt is encoded standalone
            // (with the tokenizer's special-token handling, e.g. BOS) and response
            // tokens follow it, with no BPE merges across the boundary.
            let prompt_tokens = trainer
                .encode_prompt(&example.prompt)
                .map_err(|e| FinetuneError::Training(format!("tokenize prompt: {e}")))?;
            let chosen_resp_tokens = trainer
                .encode(&example.chosen)
                .map_err(|e| FinetuneError::Training(format!("tokenize chosen: {e}")))?;
            let rejected_resp_tokens = trainer
                .encode(&example.rejected)
                .map_err(|e| FinetuneError::Training(format!("tokenize rejected: {e}")))?;

            let mut chosen_tokens = prompt_tokens.clone();
            chosen_tokens.extend_from_slice(&chosen_resp_tokens);
            let mut rejected_tokens = prompt_tokens.clone();
            rejected_tokens.extend_from_slice(&rejected_resp_tokens);

            let prompt_len = prompt_tokens.len();
            let chosen_len = chosen_tokens.len().min(config.max_seq_len);
            let rejected_len = rejected_tokens.len().min(config.max_seq_len);

            if chosen_len <= prompt_len || rejected_len <= prompt_len {
                continue;
            }

            // forward_from returns logits only from prompt_len-1 onwards,
            // so the output already contains just the response-portion logits.
            let resp_start = prompt_len - 1;

            // --- Policy forward pass (with LoRA) for chosen ---
            trainer.clear_kv_cache();
            let chosen_input =
                Tensor::from_vec(chosen_tokens[..chosen_len].to_vec(), &[1, chosen_len], &device)?;
            let policy_resp_chosen = trainer
                .forward_from(&chosen_input, 0, resp_start)
                .map_err(|e| FinetuneError::Training(format!("policy forward chosen: {e}")))?
                .squeeze(0)?;

            // --- Policy forward pass (with LoRA) for rejected ---
            trainer.clear_kv_cache();
            let rejected_input = Tensor::from_vec(
                rejected_tokens[..rejected_len].to_vec(),
                &[1, rejected_len],
                &device,
            )?;
            let policy_resp_rejected = trainer
                .forward_from(&rejected_input, 0, resp_start)
                .map_err(|e| FinetuneError::Training(format!("policy forward rejected: {e}")))?
                .squeeze(0)?;

            // --- Reference forward pass (without LoRA) for chosen ---
            trainer.clear_kv_cache();
            let ref_resp_chosen = trainer
                .forward_reference_from(&chosen_input, 0, resp_start)
                .map_err(|e| FinetuneError::Training(format!("ref forward chosen: {e}")))?
                .squeeze(0)?;

            // --- Reference forward pass (without LoRA) for rejected ---
            trainer.clear_kv_cache();
            let ref_resp_rejected = trainer
                .forward_reference_from(&rejected_input, 0, resp_start)
                .map_err(|e| FinetuneError::Training(format!("ref forward rejected: {e}")))?
                .squeeze(0)?;

            // Target tokens for the response portion
            let chosen_targets: Vec<u32> = chosen_tokens[prompt_len..chosen_len].to_vec();
            let rejected_targets: Vec<u32> =
                rejected_tokens[prompt_len..rejected_len].to_vec();

            // Narrow logits to match target count (last logit row predicts
            // beyond the sequence and has no corresponding target token)
            let n_chosen = chosen_targets.len();
            let policy_resp_chosen = policy_resp_chosen.narrow(0, 0, n_chosen)?;
            let ref_resp_chosen = ref_resp_chosen.narrow(0, 0, n_chosen)?;

            let n_rejected = rejected_targets.len();
            let policy_resp_rejected = policy_resp_rejected.narrow(0, 0, n_rejected)?;
            let ref_resp_rejected = ref_resp_rejected.narrow(0, 0, n_rejected)?;

            // Compute per-token log-probs and sum
            let log_pi_chosen =
                gather_log_probs(&policy_resp_chosen, &chosen_targets, &device)?;
            let log_ref_chosen =
                gather_log_probs(&ref_resp_chosen, &chosen_targets, &device)?;
            let log_pi_rejected =
                gather_log_probs(&policy_resp_rejected, &rejected_targets, &device)?;
            let log_ref_rejected =
                gather_log_probs(&ref_resp_rejected, &rejected_targets, &device)?;

            let loss = dpo_loss(
                log_pi_chosen, log_ref_chosen, log_pi_rejected, log_ref_rejected, config.beta,
            )?;

            let mut grads = loss.backward()
                .map_err(|e| FinetuneError::Training(format!("backward failed: {e}")))?;
            if epoch == 0 && epoch_count == 0 {
                check_gradient_flow(&vars, &grads, "DPO")?;
            }
            let grad_norm = clip_grad_norm(&vars, &mut grads, 1.0)?;
            optimizer.step(&grads)
                .map_err(|e| FinetuneError::Training(format!("optimizer step failed: {e}")))?;

            // Propagate updated tensors back into the model for next forward pass
            let updated: Vec<Tensor> = vars.iter().map(|v| v.as_tensor().clone()).collect();
            trainer.set_lora_tensors(&updated);

            let step_loss = loss.to_scalar::<f32>()? as f64;
            epoch_loss_sum += step_loss;
            epoch_count += 1;
            final_loss = step_loss;
            let _ = grad_norm; // used below in diagnostics
        }

        if epoch_count > 0 {
            let avg_loss = epoch_loss_sum / epoch_count as f64;
            eprintln!("DPO epoch {}/{}: avg_loss={avg_loss:.4}, examples={epoch_count}", epoch + 1, config.epochs);
        }
    }

    Ok(final_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_utils::LoraTrainable;
    use crate::test_fixtures::{make_test_lora, tensor_abs_diff, DummyTokenizer};
    /// Mock model implementing `LoraTrainable` for testing `train_dpo_full`
    /// without requiring real model weights.
    struct MockLoraTrainable {
        device: Device,
        vocab_size: usize,
        weight: Tensor,
        lora_a: Tensor,
        lora_b: Tensor,
    }

    impl MockLoraTrainable {
        fn new(vocab_size: usize) -> Self {
            let device = Device::Cpu;
            let weight =
                Tensor::rand(0.0f32, 1.0f32, &[vocab_size, vocab_size], &device).unwrap();
            let rank = 4;
            let lora_a =
                Tensor::randn(0.0f32, 0.01f32, &[rank, vocab_size], &device).unwrap();
            let lora_b =
                Tensor::randn(0.0f32, 0.01f32, &[vocab_size, rank], &device).unwrap();
            Self {
                device,
                vocab_size,
                weight,
                lora_a,
                lora_b,
            }
        }

        /// Produce logits [1, seq_len, vocab_size] from input IDs.
        /// Uses random activations (constant w.r.t. Vars) multiplied through
        /// LoRA weights to ensure gradient flow.
        fn run_forward(&self, input_ids: &Tensor, with_lora: bool) -> candle_core::Result<Tensor> {
            let (_batch, seq_len) = input_ids.dims2()?;
            // Use random activations as input features (constant w.r.t. LoRA vars)
            let activations = Tensor::rand(
                0.0f32,
                1.0f32,
                &[seq_len, self.vocab_size],
                &self.device,
            )?;
            let logits = activations.matmul(&self.weight.t()?)?; // [seq_len, vocab_size]
            let logits = if with_lora {
                let lora_out = activations
                    .matmul(&self.lora_a.t()?)?
                    .matmul(&self.lora_b.t()?)?;
                (logits + lora_out)?
            } else {
                logits
            };
            logits.unsqueeze(0) // [1, seq_len, vocab_size]
        }

        /// Forward returning logits from `start_pos` onwards (mimics real model behavior).
        fn run_forward_from(
            &self,
            input_ids: &Tensor,
            with_lora: bool,
            start_pos: usize,
        ) -> candle_core::Result<Tensor> {
            let full = self.run_forward(input_ids, with_lora)?;
            let (_b, seq_len, _v) = full.dims3()?;
            // Real models return seq_len - start_pos rows from start_pos
            full.narrow(1, start_pos, seq_len - start_pos)
        }
    }

    impl LoraTrainable for MockLoraTrainable {
        fn device(&self) -> &Device {
            &self.device
        }

        fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
            Ok(text
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| (i % self.vocab_size) as u32)
                .collect())
        }

        fn encode_prompt(&self, text: &str) -> anyhow::Result<Vec<u32>> {
            self.encode(text)
        }

        fn clear_kv_cache(&mut self) {}

        fn forward(
            &mut self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
        ) -> candle_core::Result<Tensor> {
            self.run_forward(input_ids, true)
        }

        fn forward_reference(
            &mut self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
        ) -> candle_core::Result<Tensor> {
            self.run_forward(input_ids, false)
        }

        fn forward_from(
            &mut self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            start_pos: usize,
        ) -> candle_core::Result<Tensor> {
            self.run_forward_from(input_ids, true, start_pos)
        }

        fn forward_reference_from(
            &mut self,
            input_ids: &Tensor,
            _seqlen_offset: usize,
            start_pos: usize,
        ) -> candle_core::Result<Tensor> {
            self.run_forward_from(input_ids, false, start_pos)
        }

        fn save_adapter(&self, _path: &std::path::Path) -> anyhow::Result<()> {
            Ok(())
        }

        fn lora_tensors(&self) -> Vec<Tensor> {
            vec![self.lora_a.clone(), self.lora_b.clone()]
        }

        fn set_lora_tensors(&mut self, tensors: &[Tensor]) {
            self.lora_a = tensors[0].clone();
            self.lora_b = tensors[1].clone();
        }
    }

    fn dpo_data() -> Vec<DpoExample> {
        vec![
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
        ]
    }

    #[test]
    fn test_dpo_smoke() {
        let device = Device::Cpu;
        let mut lora = make_test_lora(4, 4, 2, 4.0);
        let orig_a = lora.lora_a().clone();

        let dpo_config = DpoConfig {
            beta: 0.1,
            learning_rate: 0.01,
            epochs: 1,
            max_seq_len: 32,
        };

        let loss = train_dpo(&mut lora, &dpo_data(), &dpo_config, &DummyTokenizer, &device).unwrap();
        assert!(loss.is_finite(), "loss should be finite, got {loss}");
        assert!(tensor_abs_diff(&orig_a, lora.lora_a()) > 0.0, "LoRA weights should have changed");
    }

    #[test]
    fn test_dpo_gradients_flow() {
        // Verify that training produces finite loss and modifies weights at each epoch,
        // confirming that gradients flow through the computation graph.
        let device = Device::Cpu;
        let mut lora = make_test_lora(4, 4, 2, 4.0);

        let dpo_config = DpoConfig {
            beta: 0.1,
            learning_rate: 0.01,
            epochs: 3,
            max_seq_len: 32,
        };

        let a_before = lora.lora_a().clone();
        let loss = train_dpo(&mut lora, &dpo_data(), &dpo_config, &DummyTokenizer, &device).unwrap();

        assert!(loss.is_finite(), "DPO loss should be finite, got {loss}");
        assert!(tensor_abs_diff(&a_before, lora.lora_a()) > 0.0, "weights should change over 3 epochs");
    }

    #[test]
    fn test_dpo_both_lora_weights_change() {
        let device = Device::Cpu;
        let mut lora = make_test_lora(4, 4, 2, 4.0);
        let orig_a = lora.lora_a().clone();
        let orig_b = lora.lora_b().clone();

        let dpo_config = DpoConfig {
            beta: 0.1,
            learning_rate: 0.01,
            epochs: 3,
            max_seq_len: 32,
        };

        train_dpo(&mut lora, &dpo_data()[..1].to_vec(), &dpo_config, &DummyTokenizer, &device).unwrap();

        assert!(tensor_abs_diff(&orig_a, lora.lora_a()) > 0.0, "lora_a should change after training");
        assert!(tensor_abs_diff(&orig_b, lora.lora_b()) > 0.0, "lora_b should change after training");
    }

    #[test]
    fn test_dpo_loss_dtype_is_f32() {
        let device = Device::Cpu;

        let logits = Tensor::new(
            &[[1.0f32, 2.0, 0.5, 0.1], [0.1, 3.0, 0.2, 0.5]],
            &device,
        )
        .unwrap();
        let targets = vec![1u32, 0];

        let log_probs = gather_log_probs(&logits, &targets, &device).unwrap();
        assert_eq!(
            log_probs.dtype(),
            DType::F32,
            "log-prob result should be F32"
        );
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

    /// Regression test for the scatter-add shape mismatch fix in `train_dpo_full`.
    ///
    /// Without the narrowing fix (lines that narrow logits to match target count),
    /// `gather_log_probs` would fail because `forward_from(start_pos = prompt_len - 1)`
    /// returns one more logit row than there are target tokens.
    #[test]
    fn test_dpo_full_shape_alignment() {
        let vocab_size = 16;
        let mut mock = MockLoraTrainable::new(vocab_size);

        let data = vec![
            DpoExample {
                prompt: "question one two".into(),
                chosen: "good answer here today".into(),
                rejected: "bad answer here now".into(),
            },
            DpoExample {
                prompt: "another prompt words".into(),
                chosen: "correct response text".into(),
                rejected: "wrong response text".into(),
            },
        ];

        let config = DpoConfig {
            beta: 0.1,
            learning_rate: 0.01,
            epochs: 2,
            max_seq_len: 32,
        };

        let lora_before = mock.lora_tensors();

        let loss = train_dpo_full(&mut mock, &data, &config)
            .expect("train_dpo_full should not fail with shape mismatch");

        assert!(loss.is_finite(), "loss should be finite, got {loss}");

        // LoRA weights should have changed after training
        let lora_after = mock.lora_tensors();
        let diff_a = tensor_abs_diff(&lora_before[0], &lora_after[0]);
        let diff_b = tensor_abs_diff(&lora_before[1], &lora_after[1]);
        assert!(
            diff_a > 0.0 && diff_b > 0.0,
            "both LoRA weights should change after training (diff_a={diff_a}, diff_b={diff_b})"
        );
    }
}
