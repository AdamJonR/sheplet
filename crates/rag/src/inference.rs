use std::path::Path;
use std::sync::mpsc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma as ct_gemma;
use candle_transformers::models::gemma2 as ct_gemma2;
use candle_transformers::models::llama as ct_llama;
use candle_transformers::models::mistral as ct_mistral;
use candle_transformers::models::phi3 as ct_phi3;
use candle_transformers::models::qwen2 as ct_qwen2;
use tokenizers::Tokenizer;

use crate::error::{RagError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Phi3,
    Llama,
    Qwen2,
    Gemma,
    Gemma2,
    Mistral,
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
        Some("llama") => ModelArch::Llama,
        Some("qwen2") => ModelArch::Qwen2,
        Some("gemma") => ModelArch::Gemma,
        Some("gemma2") => ModelArch::Gemma2,
        Some("mistral") => ModelArch::Mistral,
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

// One instance exists per loaded course and lives for the whole session, so
// the per-variant size spread doesn't matter; boxing would only add noise.
#[allow(clippy::large_enum_variant)]
enum InferenceModel {
    Phi(ct_phi3::Model),
    Llama {
        model: ct_llama::Llama,
        cache: ct_llama::Cache,
        config: ct_llama::Config,
        dtype: DType,
        device: Device,
    },
    Qwen2(ct_qwen2::ModelForCausalLM),
    Gemma(ct_gemma::Model),
    Gemma2(ct_gemma2::Model),
    Mistral(ct_mistral::Model),
}

impl InferenceModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            InferenceModel::Phi(m) => m.forward(input, index_pos),
            InferenceModel::Llama { model, cache, .. } => {
                model.forward(input, index_pos, cache)
            }
            InferenceModel::Qwen2(m) => m.forward(input, index_pos),
            InferenceModel::Gemma(m) => m.forward(input, index_pos),
            InferenceModel::Gemma2(m) => m.forward(input, index_pos),
            InferenceModel::Mistral(m) => m.forward(input, index_pos),
        }
    }

    fn clear_kv_cache(&mut self) -> Result<()> {
        match self {
            InferenceModel::Phi(m) => m.clear_kv_cache(),
            InferenceModel::Llama { config, dtype, device, cache, .. } => {
                // Recreate cache to clear KV state (kvs field is private).
                // Propagate failure: silently keeping the previous request's
                // cache would contaminate the next generation.
                *cache = ct_llama::Cache::new(true, *dtype, config, device)?;
            }
            InferenceModel::Qwen2(m) => m.clear_kv_cache(),
            InferenceModel::Gemma(m) => m.clear_kv_cache(),
            InferenceModel::Gemma2(m) => m.clear_kv_cache(),
            InferenceModel::Mistral(m) => m.clear_kv_cache(),
        }
        Ok(())
    }
}

/// Per-architecture sampling configuration.
struct SamplingConfig {
    temperature: f64,
    top_p: f64,
    repetition_penalty: f64,
}

impl SamplingConfig {
    fn for_arch(arch: ModelArch) -> Self {
        match arch {
            ModelArch::Qwen2 => Self {
                temperature: 0.5,
                top_p: 0.85,
                repetition_penalty: 1.05,
            },
            ModelArch::Gemma | ModelArch::Gemma2 => Self {
                temperature: 0.5,
                top_p: 0.9,
                repetition_penalty: 1.05,
            },
            _ => Self {
                temperature: 0.6,
                top_p: 0.9,
                repetition_penalty: 1.1,
            },
        }
    }
}

pub struct PhiGenerator {
    model: InferenceModel,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_ids: Vec<u32>,
    special_token_ids: Vec<u32>,
    non_eos_special_ids: Vec<u32>,
    stop_sequences: Vec<String>,
    sampling: SamplingConfig,
    max_context_tokens: usize,
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

        // Load SafeTensors model
        let mut safetensors_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        safetensors_files.sort();

        if safetensors_files.is_empty() {
            return Err(RagError::Other(format!(
                "No safetensors files found in {}",
                model_dir.display()
            )));
        }

        let (arch, config_str) = detect_model_arch_with_config(model_dir)?;
        // Gemma2 requires F32 on GPU: candle's Metal/CUDA softmax runs in native
        // dtype, but Gemma2's attention logit softcapping compresses dynamic range,
        // making BF16 softmax precision insufficient. HuggingFace explicitly uses
        // F32 softmax for Gemma2. Without F32, precision errors compound through
        // 26 layers, producing corrupted logit distributions (e.g. a garbled
        // number like "7000" instead of "seven").
        let dtype = match (device.is_cpu(), &arch) {
            (true, _) => DType::F32,
            (_, ModelArch::Gemma2) => DType::F32,
            (false, _) => DType::BF16,
        };
        let model = match arch {
            ModelArch::Llama => {
                let llama_config: ct_llama::LlamaConfig =
                    serde_json::from_str(&config_str)?;

                eprintln!(
                    "Llama config: hidden_size={}, layers={}, heads={}, kv_heads={}",
                    llama_config.hidden_size, llama_config.num_hidden_layers,
                    llama_config.num_attention_heads,
                    llama_config.num_key_value_heads.unwrap_or(llama_config.num_attention_heads),
                );

                let config = llama_config.into_config(false);

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

                let cache = ct_llama::Cache::new(true, dtype, &config, device)?;
                let model = ct_llama::Llama::load(vb, &config)?;
                InferenceModel::Llama {
                    model,
                    cache,
                    config,
                    dtype,
                    device: device.clone(),
                }
            }
            ModelArch::Phi3 => {
                let config: ct_phi3::Config =
                    serde_json::from_str(&config_str)?;

                eprintln!(
                    "Phi3 config: hidden_size={}, head_dim={}, layers={}, \
                     rope_scaling={}",
                    config.hidden_size, config.head_dim(),
                    config.num_hidden_layers,
                    if config.rope_scaling.is_some() { "yes" } else { "none" },
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

                let model = ct_phi3::Model::new(&config, vb)?;
                InferenceModel::Phi(model)
            }
            ModelArch::Qwen2 => {
                let config: ct_qwen2::Config =
                    serde_json::from_str(&config_str)?;

                eprintln!(
                    "Qwen2 config: hidden_size={}, layers={}, heads={}, kv_heads={}",
                    config.hidden_size, config.num_hidden_layers,
                    config.num_attention_heads, config.num_key_value_heads,
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

                let model = ct_qwen2::ModelForCausalLM::new(&config, vb)?;
                InferenceModel::Qwen2(model)
            }
            ModelArch::Gemma => {
                // Gemma2 configs from HuggingFace contain both hidden_act and
                // hidden_activation; candle-transformers rejects this. Keep only one.
                let config_str_gemma = sanitize_gemma_config(&config_str);
                let config: ct_gemma::Config =
                    serde_json::from_str(&config_str_gemma)?;

                eprintln!(
                    "Gemma config: hidden_size={}, head_dim={}, layers={}, heads={}, kv_heads={}",
                    config.hidden_size, config.head_dim,
                    config.num_hidden_layers, config.num_attention_heads,
                    config.num_key_value_heads,
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

                let model = ct_gemma::Model::new(false, &config, vb)?;
                InferenceModel::Gemma(model)
            }
            ModelArch::Gemma2 => {
                let config: ct_gemma2::Config =
                    serde_json::from_str(&config_str)?;

                eprintln!(
                    "Gemma2 config: hidden_size={}, head_dim={}, layers={}, heads={}, kv_heads={}",
                    config.hidden_size, config.head_dim,
                    config.num_hidden_layers, config.num_attention_heads,
                    config.num_key_value_heads,
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

                let model = ct_gemma2::Model::new(false, &config, vb)?;
                InferenceModel::Gemma2(model)
            }
            ModelArch::Mistral => {
                let config: ct_mistral::Config =
                    serde_json::from_str(&config_str)?;

                eprintln!(
                    "Mistral config: hidden_size={}, layers={}, heads={}, kv_heads={}",
                    config.hidden_size, config.num_hidden_layers,
                    config.num_attention_heads, config.num_key_value_heads,
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

                let model = ct_mistral::Model::new(&config, vb)?;
                InferenceModel::Mistral(model)
            }
        };

        // Determine EOS token IDs from config.json, fall back to tokenizer heuristic
        let config_path = model_dir.join("config.json");
        let from_config = if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            parse_eos_token_ids(&config_str)
        } else {
            vec![]
        };
        let mut eos_token_ids = if from_config.is_empty() {
            let id = tokenizer
                .token_to_id("<|eot_id|>")
                .or_else(|| tokenizer.token_to_id("<|end|>"))
                .or_else(|| tokenizer.token_to_id("<|im_end|>"))
                .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
                .or_else(|| tokenizer.token_to_id("</s>"))
                .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                .unwrap_or(1);
            vec![id]
        } else {
            from_config
        };
        // Ensure architecture-specific end-of-turn tokens are in the EOS list
        let turn_end_tokens: &[&str] = match arch {
            ModelArch::Phi3 => &["<|end|>"],
            ModelArch::Llama => &["<|eot_id|>"],
            ModelArch::Qwen2 => &["<|im_end|>"],
            ModelArch::Gemma | ModelArch::Gemma2 => &["<end_of_turn>"],
            ModelArch::Mistral => &["</s>"],
        };
        for tok_str in turn_end_tokens {
            if let Some(id) = tokenizer.token_to_id(tok_str)
                && !eos_token_ids.contains(&id) {
                    eprintln!("Adding turn-end token '{}' (ID {}) to EOS list", tok_str, id);
                    eos_token_ids.push(id);
                }
        }

        eprintln!("EOS token IDs: {:?}", eos_token_ids);

        let mut special_token_ids: Vec<u32> = tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, token)| token.special)
            .map(|(id, _)| *id)
            .collect();

        // Also mask tool/FIM tokens that may not be marked "special" in tokenizer
        let extra_mask_tokens = [
            "<tool_call>", "</tool_call>", "<|tool_sep|>",
            "<|tool_start|>", "<|tool_end|>",
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
            "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
        ];
        for tok_str in &extra_mask_tokens {
            if let Some(id) = tokenizer.token_to_id(tok_str)
                && !special_token_ids.contains(&id) {
                    special_token_ids.push(id);
                }
        }
        eprintln!("Special token IDs: {} total", special_token_ids.len());

        // Build non-EOS special IDs for masking on all generated tokens
        let non_eos_special_ids: Vec<u32> = special_token_ids
            .iter()
            .copied()
            .filter(|id| !eos_token_ids.contains(id))
            .collect();

        // Per-architecture stop sequences (text-level detection)
        let stop_sequences = match arch {
            ModelArch::Phi3 => vec![
                "<|user|>".to_string(), "<|system|>".to_string(),
                "### Question".to_string(), "###".to_string(),
            ],
            ModelArch::Llama => vec!["<|start_header_id|>".to_string()],
            ModelArch::Qwen2 => vec![
                "<|im_start|>".to_string(), "<tool_call>".to_string(),
            ],
            ModelArch::Gemma | ModelArch::Gemma2 => vec!["<start_of_turn>".to_string()],
            ModelArch::Mistral => vec!["[INST]".to_string()],
        };

        // Read max_position_embeddings from config.json for context window guard
        let max_context_tokens = if let Ok(json) = serde_json::from_str::<serde_json::Value>(&config_str) {
            json.get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(4096)
        } else {
            4096
        };

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            eos_token_ids,
            special_token_ids,
            non_eos_special_ids,
            stop_sequences,
            sampling: SamplingConfig::for_arch(arch),
            max_context_tokens,
        })
    }

    /// Context window safety guard: if the prompt plus generation budget
    /// exceeds the model's context window, truncate from the middle,
    /// preserving the start (system prompt) and end (question).
    fn truncate_to_context(&self, tokens: Vec<u32>, max_tokens: usize) -> Vec<u32> {
        let headroom = 64;
        if tokens.len() + max_tokens + headroom <= self.max_context_tokens {
            return tokens;
        }
        let available = self.max_context_tokens.saturating_sub(max_tokens + headroom);
        if tokens.len() <= available || available == 0 {
            return tokens;
        }
        eprintln!(
            "WARNING: prompt ({} tokens) exceeds safe limit ({} tokens for context window {}). Truncating.",
            tokens.len(), available, self.max_context_tokens
        );
        let keep_start = available / 2;
        let keep_end = available - keep_start;
        let end_start = tokens.len() - keep_end;
        let mut truncated = tokens[..keep_start].to_vec();
        truncated.extend_from_slice(&tokens[end_start..]);
        truncated
    }

    fn generate_tokens(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
        self.model.clear_kv_cache()?;
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            ?;
        let input_ids = encoding.get_ids();
        let mut tokens: Vec<u32> = self.truncate_to_context(input_ids.to_vec(), max_tokens);
        let mut generated = Vec::new();

        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = last_token_logits(&logits)?;

        // Log top-5 raw logits for diagnostics
        {
            let logits_vec = logits.to_vec1::<f32>()?;
            let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("First-token top-5 logits:");
            for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
                let token_str = self.tokenizer.decode(&[*idx as u32], false).unwrap_or_default();
                eprintln!("    {}: token {} ({:?}) = {:.4}", i + 1, idx, token_str, val);
            }
        }

        let SamplingConfig { temperature, top_p, repetition_penalty } = self.sampling;

        // Prevent ALL special tokens (including EOS) as very first generated token
        let logits = mask_token_ids(&logits, &self.special_token_ids)?;
        let next_token = sample_token(&logits, temperature, top_p, repetition_penalty, &[])?;
        generated.push(next_token);
        tokens.push(next_token);

        for _ in 1..max_tokens {
            let last_token = *tokens.last().unwrap();
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() - 1)?;
            let logits = last_token_logits(&logits)?;
            // Mask non-EOS special tokens; EOS is allowed so the model can stop naturally
            let logits = mask_token_ids(&logits, &self.non_eos_special_ids)?;
            let next_token = sample_token(&logits, temperature, top_p, repetition_penalty, &generated)?;

            if self.eos_token_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);
            tokens.push(next_token);

            // Check text-level stop sequences
            if !self.stop_sequences.is_empty()
                && let Ok(text) = self.tokenizer.decode(&generated, true)
                    && let Some(pos) = check_stop_sequences(&text, &self.stop_sequences) {
                        // Re-encode the truncated text to get the right token count
                        let truncated_text = &text[..pos];
                        if let Ok(enc) = self.tokenizer.encode(truncated_text, false) {
                            generated = enc.get_ids().to_vec();
                        }
                        break;
                    }
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
        Ok(clean_output(&text))
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
        self.model.clear_kv_cache()?;
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            ?;
        let input_ids = encoding.get_ids();
        let mut tokens: Vec<u32> = self.truncate_to_context(input_ids.to_vec(), max_tokens);

        let SamplingConfig { temperature, top_p, repetition_penalty } = self.sampling;

        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = last_token_logits(&logits)?;
        // Prevent ALL special tokens (including EOS) as very first generated token
        let logits = mask_token_ids(&logits, &self.special_token_ids)?;
        let first_token = sample_token(&logits, temperature, top_p, repetition_penalty, &[])?;

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
            // Mask non-EOS special tokens; EOS is allowed so the model can stop naturally
            let logits = mask_token_ids(&logits, &self.non_eos_special_ids)?;
            let next_token = sample_token(&logits, temperature, top_p, repetition_penalty, &all_generated)?;

            if self.eos_token_ids.contains(&next_token) {
                break;
            }

            tokens.push(next_token);
            all_generated.push(next_token);

            match tokenizer.decode(&all_generated, true) {
                Ok(full_text) => {
                    // Check text-level stop sequences
                    if let Some(pos) = check_stop_sequences(&full_text, &self.stop_sequences) {
                        // Send any remaining text up to the stop point
                        if pos > prev_text.len() {
                            let _ = tx.send(Ok(full_text[prev_text.len()..pos].to_string()));
                        }
                        break;
                    }
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
        if let Err(e) = self.model.clear_kv_cache() {
            eprintln!("WARNING: failed to clear KV cache: {e}");
        }
    }
}

/// Check if the generated text contains any stop sequence.
/// Returns the byte offset of the first match, or None.
fn check_stop_sequences(text: &str, stop_seqs: &[String]) -> Option<usize> {
    stop_seqs
        .iter()
        .filter_map(|seq| text.find(seq.as_str()))
        .min()
}

/// Clean up model output: trim and remove leaked special tokens.
fn clean_output(text: &str) -> String {
    let mut result = text.trim().to_string();

    // Remove residual special token text that leaked through decoding
    let leaked_tokens = [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<start_of_turn>", "<end_of_turn>",
        "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
        "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
        "<tool_call>", "</tool_call>",
    ];
    for tok in &leaked_tokens {
        result = result.replace(tok, "");
    }

    result.trim().to_string()
}

/// Gemma/Gemma2 HuggingFace configs may set both `hidden_act` and
/// `hidden_activation`. candle-transformers errors when both are present.
/// Remove `hidden_activation` when `hidden_act` exists (they're always identical).
fn sanitize_gemma_config(config_str: &str) -> String {
    if let Ok(mut json) = serde_json::from_str::<serde_json::Value>(config_str) {
        if let Some(obj) = json.as_object_mut()
            && obj.contains_key("hidden_act") && obj.contains_key("hidden_activation") {
                obj.remove("hidden_activation");
            }
        serde_json::to_string(&json).unwrap_or_else(|_| config_str.to_string())
    } else {
        config_str.to_string()
    }
}

/// Set logits for the given token IDs to -inf so they cannot be sampled.
fn mask_token_ids(logits: &Tensor, token_ids: &[u32]) -> candle_core::Result<Tensor> {
    if token_ids.is_empty() {
        return Ok(logits.clone());
    }
    let mut logits_vec = logits.to_vec1::<f32>()?;
    for &id in token_ids {
        if (id as usize) < logits_vec.len() {
            logits_vec[id as usize] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(logits_vec, logits.shape(), logits.device())
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

    // Apply repetition penalty once per unique token (not per occurrence)
    let unique_tokens: std::collections::HashSet<u32> = generated_tokens.iter().copied().collect();
    for token_id in &unique_tokens {
        if let Some(logit) = logits.get_mut(*token_id as usize) {
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
            r#"{"model_type": "unknown_model"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Phi3, "unknown model_type should fall back to Phi3");
    }

    #[test]
    fn test_detect_model_arch_qwen2() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "qwen2"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Qwen2);
    }

    #[test]
    fn test_detect_model_arch_gemma() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "gemma"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Gemma);
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
        assert_eq!(arch, ModelArch::Gemma2, "gemma2 should map to Gemma2 arch");
    }

    #[test]
    fn test_detect_model_arch_mistral() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "mistral"}"#,
        )
        .unwrap();
        let arch = detect_model_arch(dir.path()).unwrap();
        assert_eq!(arch, ModelArch::Mistral);
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
    fn test_lora_layer_to_weight_path_llama() {
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
    fn test_merge_lora_adapter_bf16_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let dtype = DType::BF16;

        // Create base weights in BF16
        let base_f32 = Tensor::ones(&[4, 8], DType::F32, &device).unwrap();
        let base_weight = base_f32.to_dtype(DType::BF16).unwrap();
        let mut base_tensors = std::collections::HashMap::new();
        base_tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            base_weight,
        );
        let base_path = dir.path().join("model.safetensors");
        candle_core::safetensors::save(&base_tensors, &base_path).unwrap();

        // Create adapter tensors (F32 — adapter files are typically F32)
        let lora_a = Tensor::ones(&[2, 8], DType::F32, &device).unwrap();
        let lora_b = Tensor::ones(&[4, 2], DType::F32, &device).unwrap();
        let scale = Tensor::from_vec(vec![1.0f32], &[1], &device).unwrap();

        let mut adapter_tensors = std::collections::HashMap::new();
        adapter_tensors.insert("layers.0.q_proj.lora_a".to_string(), lora_a);
        adapter_tensors.insert("layers.0.q_proj.lora_b".to_string(), lora_b);
        adapter_tensors.insert("lora_scale".to_string(), scale);

        let adapter_path = dir.path().join("adapter.safetensors");
        candle_core::safetensors::save(&adapter_tensors, &adapter_path).unwrap();

        // Merge with BF16 dtype — should not crash (previously failed with BF16→F64 conversion)
        let vb = merge_lora_adapter(&[base_path], &adapter_path, &device, dtype, false).unwrap();

        // Verify merge produced correct values (1.0 + 2.0 = 3.0)
        let merged_weight = vb
            .pp("model.layers.0.self_attn.q_proj")
            .get((4, 8), "weight")
            .unwrap();

        let merged_f32 = merged_weight.to_dtype(DType::F32).unwrap();
        let expected = Tensor::from_vec(vec![3.0f32; 32], &[4, 8], &device).unwrap();
        let diff = (&merged_f32 - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 0.1, "merged BF16 weight values should be ~3.0, diff={diff}");
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
        let config = r#"{"model_type": "phi3", "eos_token_id": 1}"#;
        assert_eq!(parse_eos_token_ids(config), vec![1]);
    }

    #[test]
    fn test_parse_eos_token_ids_array() {
        let config = r#"{"model_type": "llama", "eos_token_id": [1, 106]}"#;
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
    fn test_sanitize_gemma_config_removes_hidden_activation() {
        let input = r#"{"hidden_act":"gelu","hidden_activation":"gelu","hidden_size":2048}"#;
        let result = sanitize_gemma_config(input);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed.get("hidden_act").is_some());
        assert!(parsed.get("hidden_activation").is_none());
        assert!(parsed.get("hidden_size").is_some());
    }

    #[test]
    fn test_sanitize_gemma_config_keeps_single_field() {
        let input = r#"{"hidden_act":"gelu","hidden_size":2048}"#;
        let result = sanitize_gemma_config(input);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed.get("hidden_act").is_some());
        assert!(parsed.get("hidden_activation").is_none());
    }

    #[test]
    fn test_sanitize_gemma_config_passthrough_invalid_json() {
        let input = "not json at all";
        assert_eq!(sanitize_gemma_config(input), input);
    }

    #[test]
    fn test_mask_token_ids_masks_correctly() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
        let masked = mask_token_ids(&logits, &[1, 3]).unwrap();
        let vals = masked.to_vec1::<f32>().unwrap();
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], f32::NEG_INFINITY);
        assert_eq!(vals[2], 3.0);
        assert_eq!(vals[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_mask_token_ids_empty_ids() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let masked = mask_token_ids(&logits, &[]).unwrap();
        let vals = masked.to_vec1::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mask_token_ids_out_of_range_id() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[1.0f32, 2.0], &device).unwrap();
        // ID 999 is out of range — should not panic
        let masked = mask_token_ids(&logits, &[999]).unwrap();
        let vals = masked.to_vec1::<f32>().unwrap();
        assert_eq!(vals, vec![1.0, 2.0]);
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

    // Extract scale from adapter if present; default to 1.0 (safe: applies delta at face value)
    // Handle both scalar (rank 0) and [1]-shaped tensors
    let scale = adapter_data
        .get("lora_scale")
        .and_then(|t| t.flatten_all().ok())
        .and_then(|t| t.to_vec1::<f32>().ok())
        .and_then(|v| v.into_iter().next())
        .map(|s| s as f64)
        .unwrap_or_else(|| {
            eprintln!("LoRA merge: WARNING: adapter missing lora_scale tensor, defaulting to 1.0");
            1.0
        });

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
                let base_weight = base_weight.to_dtype(dtype)?;
                let merged = (&base_weight + &delta)?;

                // Compute delta magnitude relative to base weight
                let base_norm = base_weight
                    .sqr()?
                    .sum_all()?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?
                    .sqrt();
                let delta_norm = delta
                    .sqr()?
                    .sum_all()?
                    .to_dtype(DType::F32)?
                    .to_scalar::<f32>()?
                    .sqrt();
                let ratio = if base_norm > 0.0 {
                    delta_norm / base_norm
                } else {
                    f32::INFINITY
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

    // Ensure all tensors match the target dtype (e.g. base weights may be BF16
    // but Gemma2 needs F32 for softmax precision)
    let base_tensors: std::collections::HashMap<String, Tensor> = base_tensors
        .into_iter()
        .map(|(name, tensor)| {
            let tensor = if tensor.dtype() != dtype {
                tensor.to_dtype(dtype).unwrap_or(tensor)
            } else {
                tensor
            };
            (name, tensor)
        })
        .collect();

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
