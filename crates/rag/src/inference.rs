use std::path::Path;
use std::sync::mpsc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::error::{RagError, Result};
use crate::quantized_phi3;
use crate::quantized_llama;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Phi3,
    Gemma3,
    Llama,
}

/// Parse `eos_token_id` from a model's config.json string.
/// Handles both single integer and array-of-integers formats.
fn parse_eos_token_ids(config_str: &str) -> Vec<u32> {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(config_str) {
        match &json["eos_token_id"] {
            serde_json::Value::Number(n) => {
                n.as_u64().map(|v| vec![v as u32]).unwrap_or_default()
            }
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as u32))
                .collect(),
            _ => vec![],
        }
    } else {
        vec![]
    }
}

pub fn detect_model_arch(model_dir: impl AsRef<Path>) -> Result<ModelArch> {
    let (arch, _) = detect_model_arch_with_config(model_dir)?;
    Ok(arch)
}

/// Detect model architecture and return the raw config string to avoid re-reading.
fn detect_model_arch_with_config(model_dir: impl AsRef<Path>) -> Result<(ModelArch, String)> {
    let config_path = model_dir.as_ref().join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;
    let arch = match config.get("model_type").and_then(|v| v.as_str()) {
        Some("gemma3_text" | "gemma2") => ModelArch::Gemma3,
        Some("llama") => ModelArch::Llama,
        _ => ModelArch::Phi3,
    };
    Ok((arch, config_str))
}

pub trait TextGenerator: Send {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;
    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String>>>;
    fn generate_to_sender(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        tx: mpsc::Sender<Result<String>>,
    ) -> Result<()>;
    fn clear_cache(&mut self);
}

enum InferenceModel {
    PhiFull(crate::phi3::Model),
    PhiQuantized(quantized_phi3::ModelWeights),
    Gemma3(crate::gemma3::Gemma3Model),
    LlamaFull(crate::llama::LlamaModel),
    LlamaQuantized(quantized_llama::ModelWeights),
}

impl InferenceModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            InferenceModel::PhiFull(m) => m.forward(input, index_pos),
            InferenceModel::PhiQuantized(m) => m.forward(input, index_pos),
            InferenceModel::Gemma3(m) => m.forward(input, index_pos),
            InferenceModel::LlamaFull(m) => m.forward(input, index_pos),
            InferenceModel::LlamaQuantized(m) => m.forward(input, index_pos),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            InferenceModel::PhiFull(m) => m.clear_kv_cache(),
            InferenceModel::PhiQuantized(m) => m.clear_kv_cache(),
            InferenceModel::Gemma3(m) => m.clear_kv_cache(),
            InferenceModel::LlamaFull(m) => m.clear_kv_cache(),
            InferenceModel::LlamaQuantized(m) => m.clear_kv_cache(),
        }
    }
}

pub struct PhiGenerator {
    model: InferenceModel,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_ids: Vec<u32>,
}

impl PhiGenerator {
    pub fn load(
        model_dir: impl AsRef<Path>,
        adapter_path: Option<&Path>,
        device: &Device,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            ?;

        // Check for GGUF file first (quantized model)
        let gguf_path = model_dir.join("model.gguf");
        let model = if gguf_path.exists() {
            let mut file = std::fs::File::open(&gguf_path)?;
            let ct = candle_core::quantized::gguf_file::Content::read(&mut file)
                .map_err(|e| RagError::Other(format!("failed to read GGUF: {e}")))?;
            // Dispatch based on GGUF architecture metadata
            let gguf_arch = ct.metadata.get("general.architecture")
                .and_then(|v| v.to_string().ok())
                .map(|s| s.to_string())
                .unwrap_or_default();
            match gguf_arch.as_str() {
                "llama" => {
                    let weights = quantized_llama::ModelWeights::from_gguf(ct, &mut file, device)
                        .map_err(|e| RagError::Other(format!("failed to load quantized Llama model: {e}")))?;
                    InferenceModel::LlamaQuantized(weights)
                }
                _ => {
                    let weights = quantized_phi3::ModelWeights::from_gguf(ct, &mut file, device)
                        .map_err(|e| RagError::Other(format!("failed to load quantized model: {e}")))?;
                    InferenceModel::PhiQuantized(weights)
                }
            }
        } else {
            // Fall back to SafeTensors path
            let mut safetensors_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
                .collect();
            safetensors_files.sort();

            if safetensors_files.is_empty() {
                return Err(RagError::Other(format!(
                    "No safetensors or GGUF files found in {}",
                    model_dir.display()
                )));
            }

            let (arch, config_str) = detect_model_arch_with_config(model_dir)?;
            let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
            match arch {
                ModelArch::Gemma3 => {
                    let config: crate::gemma3::Gemma3Config =
                        serde_json::from_str(&config_str)?;

                    let sliding_count = config.layer_types.iter().filter(|t| t.as_str() == "sliding_attention").count();
                    let global_count = config.num_hidden_layers - sliding_count;
                    eprintln!(
                        "Gemma3 config: hidden_size={}, head_dim={}, layers={} ({} sliding + {} global), \
                         rope_theta={}, rope_local_base_freq={:?}, sliding_window={:?}",
                        config.hidden_size, config.head_dim, config.num_hidden_layers,
                        sliding_count, global_count,
                        config.rope_theta, config.rope_local_base_freq, config.sliding_window,
                    );

                    let vb = if adapter_path.is_some_and(|p| p.exists()) {
                        merge_lora_adapter(
                            &safetensors_files,
                            adapter_path.unwrap(),
                            device,
                            dtype,
                            false,
                        )?
                    } else {
                        unsafe {
                            VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, device)?
                        }
                    };

                    let model = crate::gemma3::Gemma3Model::new(&config, vb)?;
                    InferenceModel::Gemma3(model)
                }
                ModelArch::Llama => {
                    let config: crate::llama::LlamaConfig =
                        serde_json::from_str(&config_str)?;

                    eprintln!(
                        "Llama config: hidden_size={}, head_dim={}, layers={}, \
                         rope_theta={}, tie_word_embeddings={}",
                        config.hidden_size, config.head_dim(), config.num_hidden_layers,
                        config.rope_theta, config.tie_word_embeddings,
                    );

                    let vb = if adapter_path.is_some_and(|p| p.exists()) {
                        merge_lora_adapter(
                            &safetensors_files,
                            adapter_path.unwrap(),
                            device,
                            dtype,
                            false,
                        )?
                    } else {
                        unsafe {
                            VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, device)?
                        }
                    };

                    let model = crate::llama::LlamaModel::new(&config, vb)?;
                    InferenceModel::LlamaFull(model)
                }
                ModelArch::Phi3 => {
                    let config: crate::phi3::Config =
                        serde_json::from_str(&config_str)?;

                    eprintln!(
                        "Phi3 config: hidden_size={}, head_dim={}, rope_dim={}, layers={}, \
                         partial_rotary_factor={}, rope_scaling={}, tie_word_embeddings={}",
                        config.hidden_size, config.head_dim(), config.rope_dim(),
                        config.num_hidden_layers, config.partial_rotary_factor,
                        if config.rope_scaling.is_some() { "longrope" } else { "none" },
                        config.tie_word_embeddings,
                    );

                    let vb = if adapter_path.is_some_and(|p| p.exists()) {
                        merge_lora_adapter(
                            &safetensors_files,
                            adapter_path.unwrap(),
                            device,
                            dtype,
                            false,
                        )?
                    } else {
                        unsafe {
                            VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, device)?
                        }
                    };

                    let model = crate::phi3::Model::new(&config, vb)?;
                    InferenceModel::PhiFull(model)
                }
            }
        };

        // Determine EOS token IDs: prefer config.json, fall back to tokenizer heuristic
        let eos_token_ids = if gguf_path.exists() {
            // GGUF models don't have config.json — use tokenizer heuristic
            let id = tokenizer
                .token_to_id("<end_of_turn>")
                .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
                .or_else(|| tokenizer.token_to_id("<eos>"))
                .or_else(|| tokenizer.token_to_id("<|end|>"))
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .unwrap_or(1);
            vec![id]
        } else {
            let config_path = model_dir.join("config.json");
            let from_config = if let Ok(config_str) = std::fs::read_to_string(&config_path) {
                parse_eos_token_ids(&config_str)
            } else {
                vec![]
            };
            if from_config.is_empty() {
                // Fall back to tokenizer heuristic
                let id = tokenizer
                    .token_to_id("<end_of_turn>")
                    .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
                    .or_else(|| tokenizer.token_to_id("<eos>"))
                    .or_else(|| tokenizer.token_to_id("<|end|>"))
                    .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                    .unwrap_or(1);
                vec![id]
            } else {
                from_config
            }
        };
        eprintln!("EOS token IDs: {:?}", eos_token_ids);

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            eos_token_ids,
        })
    }

    fn generate_tokens(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
        self.model.clear_kv_cache();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            ?;
        let input_ids = encoding.get_ids();
        let mut tokens: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::new();

        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = last_token_logits(&logits)?;
        let logits_vec = logits.to_vec1::<f32>()?;
        let next_token = sample_token(&logits, 0.7, 0.9, 1.15, &[])?;

        if self.eos_token_ids.contains(&next_token) {
            // Log top-5 for diagnostics
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("WARNING: model produced stop token as very first token (id={next_token})");
            eprintln!("  Top-5 logits:");
            for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
                let token_str = self.tokenizer.decode(&[*idx as u32], false).unwrap_or_default();
                eprintln!("    {}: token {} ({:?}) = {:.4}", i + 1, idx, token_str, val);
            }
            return Ok(generated);
        }
        generated.push(next_token);
        tokens.push(next_token);

        for _ in 1..max_tokens {
            let last_token = *tokens.last().unwrap();
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() - 1)?;
            let logits = last_token_logits(&logits)?;
            let next_token = sample_token(&logits, 0.7, 0.9, 1.15, &generated)?;

            if self.eos_token_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);
            tokens.push(next_token);
        }

        Ok(generated)
    }
}

impl TextGenerator for PhiGenerator {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokens = self.generate_tokens(prompt, max_tokens)?;
        let text = self
            .tokenizer
            .decode(&tokens, true)
            ?;
        Ok(text)
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String>>> {
        let (tx, rx) = mpsc::channel();
        self.generate_to_sender(prompt, max_tokens, tx)?;
        Ok(rx)
    }

    fn generate_to_sender(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        tx: mpsc::Sender<Result<String>>,
    ) -> Result<()> {
        self.model.clear_kv_cache();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            ?;
        let input_ids = encoding.get_ids();
        let mut tokens: Vec<u32> = input_ids.to_vec();

        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = last_token_logits(&logits)?;
        let first_token = sample_token(&logits, 0.7, 0.9, 1.15, &[])?;

        if self.eos_token_ids.contains(&first_token) {
            eprintln!("WARNING: model produced stop token as very first token (id={first_token}) — returning empty response");
            return Ok(());
        }

        tokens.push(first_token);

        let tokenizer = self.tokenizer.clone();
        let mut prev_text = match tokenizer.decode(&[first_token], true) {
            Ok(text) => {
                let _ = tx.send(Ok(text.clone()));
                text
            }
            Err(e) => {
                let _ = tx.send(Err(RagError::Tokenizer(e)));
                return Ok(());
            }
        };

        let mut all_generated = vec![first_token];
        for _ in 1..max_tokens {
            let last_token = *tokens.last().unwrap();
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() - 1)?;
            let logits = last_token_logits(&logits)?;
            let next_token = sample_token(&logits, 0.7, 0.9, 1.15, &all_generated)?;

            if self.eos_token_ids.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            all_generated.push(next_token);

            match tokenizer.decode(&all_generated, true) {
                Ok(full_text) => {
                    if full_text.len() > prev_text.len() {
                        let new_part = full_text[prev_text.len()..].to_string();
                        if tx.send(Ok(new_part)).is_err() {
                            break;
                        }
                        prev_text = full_text;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(RagError::Tokenizer(e)));
                    break;
                }
            }
        }

        Ok(())
    }

    fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

/// Extract the last token's logits as a 1D tensor from model output.
/// Handles varying output shapes: [vocab], [seq, vocab], or [batch, seq, vocab].
fn last_token_logits(logits: &Tensor) -> Result<Tensor> {
    let logits = match logits.dims() {
        [_vocab] => logits.clone(),
        [_seq, _vocab] => {
            let seq = logits.dim(0)?;
            logits.get(seq - 1)?
        }
        [_batch, _seq, _vocab] => {
            let batch_last = logits.get(logits.dim(0)? - 1)?;
            let seq = batch_last.dim(0)?;
            batch_last.get(seq - 1)?
        }
        dims => {
            return Err(RagError::Other(format!(
                "unexpected logits shape: {dims:?}"
            )));
        }
    };
    Ok(logits.to_dtype(DType::F32)?)
}

fn sample_token(
    logits: &Tensor,
    temperature: f64,
    top_p: f64,
    repetition_penalty: f64,
    generated_tokens: &[u32],
) -> Result<u32> {
    let mut logits = logits.to_vec1::<f32>()?;

    // Apply repetition penalty to previously generated tokens
    for &token_id in generated_tokens {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            if *logit > 0.0 {
                *logit /= repetition_penalty as f32;
            } else {
                *logit *= repetition_penalty as f32;
            }
        }
    }

    let scaled: Vec<f64> = logits.iter().map(|&l| l as f64 / temperature).collect();

    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

    let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut candidates = Vec::new();
    for (idx, prob) in &indexed {
        cumulative += prob;
        candidates.push((*idx, *prob));
        if cumulative >= top_p {
            break;
        }
    }

    let total: f64 = candidates.iter().map(|(_, p)| p).sum();
    let threshold = rand::random::<f64>() * total;
    let mut acc = 0.0;
    for (idx, prob) in &candidates {
        acc += prob;
        if acc >= threshold {
            return Ok(*idx as u32);
        }
    }

    Ok(candidates.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_model_arch_phi3() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "phi3"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Phi3);
    }

    #[test]
    fn test_detect_model_arch_gemma3_text() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "gemma3_text"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Gemma3);
    }

    #[test]
    fn test_detect_model_arch_gemma2() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "gemma2"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Gemma3);
    }

    #[test]
    fn test_detect_model_arch_llama() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "llama"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Llama);
    }

    #[test]
    fn test_detect_model_arch_unknown_falls_back_to_phi3() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "mistral"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Phi3, "unknown model_type should fall back to Phi3");
    }

    #[test]
    fn test_detect_model_arch_missing_config() {
        let dir = tempfile::tempdir().unwrap();
        // No config.json
        let result = detect_model_arch(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_layer_to_weight_path_phi3() {
        assert_eq!(
            lora_layer_to_weight_path("layers.0.qkv_proj"),
            "model.layers.0.self_attn.qkv_proj.weight"
        );
    }

    #[test]
    fn test_lora_layer_to_weight_path_gemma3() {
        assert_eq!(
            lora_layer_to_weight_path("layers.0.q_proj"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            lora_layer_to_weight_path("layers.5.o_proj"),
            "model.layers.5.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_merge_lora_adapter_modifies_weights() {
        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a base "model" with a single weight tensor (all ones)
        let base_weight = Tensor::ones(&[4, 8], dtype, &device).unwrap();
        let mut base_tensors = std::collections::HashMap::new();
        base_tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            base_weight.clone(),
        );

        let base_path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&base_tensors, &base_path).unwrap();

        // Create adapter: lora_a=ones(2,8), lora_b=ones(4,2), scale=1.0
        // delta = B @ A * scale = ones(4,2) @ ones(2,8) * 1.0 = 2*ones(4,8)
        let lora_a = Tensor::ones(&[2, 8], DType::F32, &device).unwrap();
        let lora_b = Tensor::ones(&[4, 2], DType::F32, &device).unwrap();
        let scale = Tensor::from_vec(vec![1.0f32], &[1], &device).unwrap();

        let mut adapter_tensors = std::collections::HashMap::new();
        adapter_tensors.insert("layers.0.q_proj.lora_a".to_string(), lora_a);
        adapter_tensors.insert("layers.0.q_proj.lora_b".to_string(), lora_b);
        adapter_tensors.insert("lora_scale".to_string(), scale);

        let adapter_path = dir.path().join("adapter.safetensors");
        candle_core::safetensors::save(&adapter_tensors, &adapter_path).unwrap();

        // Merge
        let vb = merge_lora_adapter(&[base_path], &adapter_path, &device, dtype, false).unwrap();

        // Get merged weight
        let merged_weight = vb
            .pp("model.layers.0.self_attn.q_proj")
            .get((4, 8), "weight")
            .unwrap();

        // Verify weight was modified: base=1.0, delta=2.0, merged=3.0
        let diff_from_base = (&merged_weight - &base_weight)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff_from_base > 0.0, "merged weights should differ from base");

        // Each element should be 3.0 (1.0 + 2.0)
        let expected = Tensor::from_vec(vec![3.0f32; 32], &[4, 8], &device).unwrap();
        let diff = (&merged_weight - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "merged weight values should be 3.0, diff={diff}");
    }

    #[test]
    fn test_merge_lora_adapter_no_lora_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create base model
        let base_weight = Tensor::ones(&[4, 8], dtype, &device).unwrap();
        let mut base_tensors = std::collections::HashMap::new();
        base_tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            base_weight.clone(),
        );
        let base_path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&base_tensors, &base_path).unwrap();

        // Create adapter with no LoRA pairs
        let mut adapter_tensors = std::collections::HashMap::new();
        adapter_tensors.insert(
            "some_other_tensor".to_string(),
            Tensor::ones(&[2], DType::F32, &device).unwrap(),
        );
        let adapter_path = dir.path().join("adapter.safetensors");
        candle_core::safetensors::save(&adapter_tensors, &adapter_path).unwrap();

        // Merge
        let vb = merge_lora_adapter(&[base_path], &adapter_path, &device, dtype, false).unwrap();

        // Weight should be unchanged
        let weight = vb
            .pp("model.layers.0.self_attn.q_proj")
            .get((4, 8), "weight")
            .unwrap();

        let diff = (&weight - &base_weight)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "weights should be unchanged when no LoRA pairs");
    }

    #[test]
    fn test_merge_lora_adapter_mismatched_keys_error() {
        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create base model with one key
        let base_weight = Tensor::ones(&[4, 8], dtype, &device).unwrap();
        let mut base_tensors = std::collections::HashMap::new();
        base_tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            base_weight,
        );
        let base_path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&base_tensors, &base_path).unwrap();

        // Create adapter with LoRA pairs that DON'T match any base key
        let lora_a = Tensor::ones(&[2, 8], DType::F32, &device).unwrap();
        let lora_b = Tensor::ones(&[4, 2], DType::F32, &device).unwrap();
        let scale = Tensor::from_vec(vec![1.0f32], &[1], &device).unwrap();

        let mut adapter_tensors = std::collections::HashMap::new();
        adapter_tensors.insert("layers.99.q_proj.lora_a".to_string(), lora_a);
        adapter_tensors.insert("layers.99.q_proj.lora_b".to_string(), lora_b);
        adapter_tensors.insert("lora_scale".to_string(), scale);

        let adapter_path = dir.path().join("adapter.safetensors");
        candle_core::safetensors::save(&adapter_tensors, &adapter_path).unwrap();

        // Merge should fail because no pairs matched
        let result = merge_lora_adapter(&[base_path], &adapter_path, &device, dtype, false);
        assert!(result.is_err(), "should error when no LoRA pairs match base weights");
        let err_msg = result.err().map(|e| e.to_string()).unwrap();
        assert!(
            err_msg.contains("none of the"),
            "error should mention no pairs matched: {err_msg}"
        );
    }

    #[test]
    fn test_merge_lora_adapter_nan_detection() {
        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create base model with a weight containing Inf (which can produce NaN when added)
        let inf_data = vec![f32::INFINITY; 32];
        let base_weight = Tensor::from_vec(inf_data, &[4, 8], &device).unwrap();
        let mut base_tensors = std::collections::HashMap::new();
        base_tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            base_weight,
        );
        let base_path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&base_tensors, &base_path).unwrap();

        // Create adapter that produces -Inf delta to create NaN (Inf + (-Inf) = NaN)
        let neg_inf_data = vec![f32::NEG_INFINITY; 16];
        let lora_a = Tensor::from_vec(neg_inf_data.clone(), &[2, 8], &device).unwrap();
        let lora_b = Tensor::ones(&[4, 2], DType::F32, &device).unwrap();
        let scale = Tensor::from_vec(vec![1.0f32], &[1], &device).unwrap();

        let mut adapter_tensors = std::collections::HashMap::new();
        adapter_tensors.insert("layers.0.q_proj.lora_a".to_string(), lora_a);
        adapter_tensors.insert("layers.0.q_proj.lora_b".to_string(), lora_b);
        adapter_tensors.insert("lora_scale".to_string(), scale);

        let adapter_path = dir.path().join("adapter.safetensors");
        candle_core::safetensors::save(&adapter_tensors, &adapter_path).unwrap();

        // Merge should fail due to NaN/Inf in merged tensors
        let result = merge_lora_adapter(&[base_path], &adapter_path, &device, dtype, false);
        assert!(result.is_err(), "should error when merged weights contain NaN/Inf");
        let err_msg = result.err().map(|e| e.to_string()).unwrap();
        assert!(
            err_msg.contains("invalid values"),
            "error should mention invalid values: {err_msg}"
        );
    }

    #[test]
    fn test_lora_layer_to_weight_path_all_projections() {
        // q_proj
        assert_eq!(
            lora_layer_to_weight_path("layers.0.q_proj"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        // k_proj
        assert_eq!(
            lora_layer_to_weight_path("layers.3.k_proj"),
            "model.layers.3.self_attn.k_proj.weight"
        );
        // v_proj
        assert_eq!(
            lora_layer_to_weight_path("layers.17.v_proj"),
            "model.layers.17.self_attn.v_proj.weight"
        );
        // o_proj
        assert_eq!(
            lora_layer_to_weight_path("layers.25.o_proj"),
            "model.layers.25.self_attn.o_proj.weight"
        );
        // qkv_proj (Phi3)
        assert_eq!(
            lora_layer_to_weight_path("layers.1.qkv_proj"),
            "model.layers.1.self_attn.qkv_proj.weight"
        );
        // Fallback for unrecognized format
        assert_eq!(
            lora_layer_to_weight_path("something_else"),
            "something_else.weight"
        );
    }

    #[test]
    fn test_parse_eos_token_ids_single() {
        let config = r#"{"model_type": "gemma3_text", "eos_token_id": 1}"#;
        assert_eq!(parse_eos_token_ids(config), vec![1]);
    }

    #[test]
    fn test_parse_eos_token_ids_array() {
        let config = r#"{"model_type": "gemma3_text", "eos_token_id": [1, 106]}"#;
        assert_eq!(parse_eos_token_ids(config), vec![1, 106]);
    }

    #[test]
    fn test_parse_eos_token_ids_missing() {
        let config = r#"{"model_type": "phi3"}"#;
        assert!(parse_eos_token_ids(config).is_empty());
    }

    #[test]
    fn test_parse_eos_token_ids_invalid_json() {
        assert!(parse_eos_token_ids("not json").is_empty());
    }

    #[test]
    fn test_sample_token_greedy() {
        // With very low temperature, should pick highest logit
        let device = Device::Cpu;
        let logits = Tensor::new(&[0.1f32, 0.2, 10.0, 0.3], &device).unwrap();
        // Run multiple times to verify determinism at near-zero temperature
        // top_p=1.0 to not filter, repetition_penalty=1.0 to not modify
        let token = sample_token(&logits, 0.01, 1.0, 1.0, &[]).unwrap();
        assert_eq!(token, 2, "greedy should pick index 2 (highest logit)");
    }

    #[test]
    fn test_phi4_mini_config_deserializes() {
        let config_json = r#"{
            "vocab_size": 200064,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "bos_token_id": 199999,
            "eos_token_id": 199999,
            "rope_scaling": {
                "long_factor": [1.0, 1.1, 1.2],
                "short_factor": [1.0, 1.1, 1.2],
                "type": "longrope"
            },
            "max_position_embeddings": 131072,
            "partial_rotary_factor": 0.75
        }"#;
        let config: crate::phi3::Config = serde_json::from_str(config_json)
            .expect("should deserialize phi-4-mini config");
        assert_eq!(config.vocab_size, 200064);
        assert_eq!(config.num_hidden_layers, 32);
        assert!(config.rope_scaling.is_some());
        assert_eq!(config.rope_scaling.as_ref().unwrap().long_factor.len(), 3);
        assert_eq!(config.rope_dim(), 96); // 128 * 0.75
    }

    #[test]
    fn test_phi3_null_rope_scaling() {
        let config_json = r#"{
            "vocab_size": 32064,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096
        }"#;
        let config: crate::phi3::Config = serde_json::from_str(config_json)
            .expect("should deserialize config without rope_scaling");
        assert!(config.rope_scaling.is_none());
        // Default partial_rotary_factor is 1.0, so rope_dim == head_dim
        assert_eq!(config.rope_dim(), config.head_dim());
    }

    #[test]
    fn test_phi4_mini_tie_word_embeddings_deserialized() {
        let config_json = r#"{
            "vocab_size": 200064,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "max_position_embeddings": 131072
        }"#;
        let config: crate::phi3::Config = serde_json::from_str(config_json).unwrap();
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_phi3_tie_word_embeddings_default_false() {
        let config_json = r#"{
            "vocab_size": 32064,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096
        }"#;
        let config: crate::phi3::Config = serde_json::from_str(config_json).unwrap();
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_sample_token_repetition_penalty() {
        let device = Device::Cpu;
        // Token 2 has highest logit, but we penalize it
        let logits = Tensor::new(&[0.1f32, 0.2, 5.0, 4.9], &device).unwrap();
        // With a high enough penalty, token 2 should be penalized below token 3
        let mut count_2 = 0;
        let mut count_3 = 0;
        for _ in 0..50 {
            let token = sample_token(&logits, 0.01, 1.0, 100.0, &[2]).unwrap();
            if token == 2 {
                count_2 += 1;
            }
            if token == 3 {
                count_3 += 1;
            }
        }
        assert!(
            count_3 > count_2,
            "with high repetition penalty on token 2, token 3 should be picked more often"
        );
    }
}

/// Load safetensors into a HashMap, injecting `lm_head.weight` as an alias for
/// `model.embed_tokens.weight` when `tie_word_embeddings` is true.
/// Merge LoRA adapter weights into base model weights.
///
/// Loads all base safetensors into memory, applies LoRA deltas
/// (W_merged = W_base + B @ A * scale), and returns a VarBuilder
/// from the merged tensors.
fn merge_lora_adapter(
    safetensors_files: &[std::path::PathBuf],
    adapter_path: &Path,
    device: &Device,
    dtype: DType,
    _tie_word_embeddings: bool,
) -> Result<VarBuilder<'static>> {
    // Load all base tensors into memory
    let mut base_tensors: std::collections::HashMap<String, Tensor> =
        std::collections::HashMap::new();
    for file in safetensors_files {
        let tensors = candle_core::safetensors::load(file, device)?;
        base_tensors.extend(tensors);
    }

    eprintln!(
        "LoRA merge: loaded {} base tensors from {} file(s)",
        base_tensors.len(),
        safetensors_files.len()
    );

    // Load adapter tensors
    let adapter_data = candle_core::safetensors::load(adapter_path, device)?;

    // Group LoRA tensors by layer name
    // Keys are like "layers.0.q_proj.lora_a", "layers.0.q_proj.lora_b"
    let mut lora_pairs: std::collections::HashMap<String, (Option<Tensor>, Option<Tensor>)> =
        std::collections::HashMap::new();

    for (name, tensor) in &adapter_data {
        if let Some(base) = name.strip_suffix(".lora_a") {
            lora_pairs
                .entry(base.to_string())
                .or_insert((None, None))
                .0 = Some(tensor.clone());
        } else if let Some(base) = name.strip_suffix(".lora_b") {
            lora_pairs
                .entry(base.to_string())
                .or_insert((None, None))
                .1 = Some(tensor.clone());
        }
    }

    if lora_pairs.is_empty() {
        eprintln!("LoRA merge: WARNING: adapter contains no LoRA pairs — returning unmodified base weights");
        return Ok(VarBuilder::from_tensors(base_tensors, dtype, device));
    }

    // Extract scale from adapter config if present, default to alpha/rank = 16/8 = 2.0
    // Handle both scalar (rank 0) and [1]-shaped tensors
    let scale = adapter_data
        .get("lora_scale")
        .and_then(|t| t.flatten_all().ok())
        .and_then(|t| t.to_vec1::<f32>().ok())
        .and_then(|v| v.into_iter().next())
        .map(|s| s as f64)
        .unwrap_or(2.0);

    eprintln!(
        "LoRA merge: found {} LoRA pair(s), scale = {:.4}",
        lora_pairs.len(),
        scale
    );

    // Apply LoRA deltas to base weights
    let mut matched_count = 0u32;
    let mut unmatched: Vec<String> = Vec::new();
    for (layer_name, (lora_a, lora_b)) in &lora_pairs {
        if let (Some(a), Some(b)) = (lora_a, lora_b) {
            // W_delta = B @ A * scale
            let delta = b.matmul(a)?.to_dtype(dtype)?;
            let delta = (delta * scale)?;

            // Map LoRA layer name to the model weight path
            let weight_path = lora_layer_to_weight_path(layer_name);
            if let Some(base_weight) = base_tensors.get(&weight_path) {
                let merged = (base_weight + &delta)?;

                // Compute delta magnitude relative to base weight
                let base_norm = base_weight
                    .sqr()?
                    .sum_all()?
                    .to_dtype(DType::F64)?
                    .to_scalar::<f64>()?
                    .sqrt();
                let delta_norm = delta
                    .sqr()?
                    .sum_all()?
                    .to_dtype(DType::F64)?
                    .to_scalar::<f64>()?
                    .sqrt();
                let ratio = if base_norm > 0.0 {
                    delta_norm / base_norm
                } else {
                    f64::INFINITY
                };

                if ratio > 1.0 {
                    eprintln!(
                        "LoRA merge: WARNING: delta/base L2 ratio = {:.4} for {} — delta is LARGER than base weight",
                        ratio, weight_path
                    );
                } else {
                    eprintln!(
                        "LoRA merge: merged {} (delta/base L2 ratio = {:.4})",
                        weight_path, ratio
                    );
                }

                base_tensors.insert(weight_path, merged);
                matched_count += 1;
            } else {
                eprintln!(
                    "LoRA merge: WARNING: no matching base weight for adapter layer '{}' (mapped to '{}')",
                    layer_name, weight_path
                );
                unmatched.push(layer_name.clone());
            }
        }
    }

    if matched_count == 0 {
        return Err(RagError::Other(format!(
            "LoRA merge failed: none of the {} adapter pair(s) matched any base weight. \
             Unmatched layers: {:?}",
            lora_pairs.len(),
            unmatched
        )));
    }

    eprintln!(
        "LoRA merge: {matched_count}/{} pair(s) matched and merged",
        lora_pairs.len()
    );

    // Validate merged tensors for NaN/Inf
    for (name, tensor) in &base_tensors {
        let flat = tensor.flatten_all()?.to_dtype(DType::F32)?;
        let values = flat.to_vec1::<f32>()?;
        let has_nan = values.iter().any(|v| v.is_nan());
        let has_inf = values.iter().any(|v| v.is_infinite());
        if has_nan || has_inf {
            return Err(RagError::Other(format!(
                "LoRA merge produced invalid values in tensor '{}': NaN={}, Inf={}",
                name, has_nan, has_inf
            )));
        }
    }

    Ok(VarBuilder::from_tensors(base_tensors, dtype, device))
}

/// Map a LoRA layer name to the corresponding model weight tensor path.
fn lora_layer_to_weight_path(lora_name: &str) -> String {
    // "layers.{i}.qkv_proj" -> "model.layers.{i}.self_attn.qkv_proj.weight"
    // "layers.{i}.o_proj" -> "model.layers.{i}.self_attn.o_proj.weight"
    if let Some(rest) = lora_name.strip_prefix("layers.")
        && let Some(dot_pos) = rest.find('.')
    {
        let idx = &rest[..dot_pos];
        let proj = &rest[dot_pos + 1..];
        return format!("model.layers.{idx}.self_attn.{proj}.weight");
    }
    format!("{lora_name}.weight")
}
