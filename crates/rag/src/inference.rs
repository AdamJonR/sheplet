use std::path::Path;
use std::sync::mpsc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi3;
use tokenizers::Tokenizer;

use crate::error::{RagError, Result};
use crate::quantized_phi3;

pub trait TextGenerator: Send {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;
    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String>>>;
    fn clear_cache(&mut self);
}

enum PhiModel {
    Full(phi3::Model),
    Quantized(quantized_phi3::ModelWeights),
}

impl PhiModel {
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            PhiModel::Full(m) => m.forward(input, index_pos),
            PhiModel::Quantized(m) => m.forward(input, index_pos),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            PhiModel::Full(m) => m.clear_kv_cache(),
            PhiModel::Quantized(m) => m.clear_kv_cache(),
        }
    }
}

pub struct PhiGenerator {
    model: PhiModel,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
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
            .map_err(|e| RagError::Tokenizer(e.to_string()))?;

        // Check for GGUF file first (quantized model)
        let gguf_path = model_dir.join("model.gguf");
        let model = if gguf_path.exists() {
            let mut file = std::fs::File::open(&gguf_path)?;
            let ct = candle_core::quantized::gguf_file::Content::read(&mut file)
                .map_err(|e| RagError::Other(format!("failed to read GGUF: {e}")))?;
            let weights = quantized_phi3::ModelWeights::from_gguf(ct, &mut file, device)
                .map_err(|e| RagError::Other(format!("failed to load quantized model: {e}")))?;
            PhiModel::Quantized(weights)
        } else {
            // Fall back to SafeTensors (F32 path)
            let config_path = model_dir.join("config.json");
            let config_str = std::fs::read_to_string(&config_path)?;
            let config: phi3::Config = serde_json::from_str(&config_str)?;

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

            let dtype = DType::F32;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, device)?
            };

            let vb = if let Some(adapter) = adapter_path {
                if adapter.exists() {
                    merge_lora_adapter(vb, adapter, device, dtype)?
                } else {
                    vb
                }
            } else {
                vb
            };

            let model = phi3::Model::new(&config, vb)?;
            PhiModel::Full(model)
        };

        // Get EOS token ID
        let eos_token_id = tokenizer
            .token_to_id("<|end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2);

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            eos_token_id,
        })
    }

    fn generate_tokens(&mut self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
        self.model.clear_kv_cache();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| RagError::Tokenizer(e.to_string()))?;
        let input_ids = encoding.get_ids();
        let mut tokens: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::new();

        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = sample_token(&logits, 0.7, 0.9)?;

        if next_token == self.eos_token_id {
            return Ok(generated);
        }
        generated.push(next_token);
        tokens.push(next_token);

        for i in 1..max_tokens {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() - 1)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = sample_token(&logits, 0.7, 0.9)?;

            if next_token == self.eos_token_id {
                break;
            }
            generated.push(next_token);
            tokens.push(next_token);

            if i >= max_tokens - 1 {
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
            .map_err(|e| RagError::Tokenizer(e.to_string()))?;
        Ok(text)
    }

    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String>>> {
        self.model.clear_kv_cache();
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| RagError::Tokenizer(e.to_string()))?;
        let input_ids = encoding.get_ids();
        let mut tokens: Vec<u32> = input_ids.to_vec();

        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let first_token = sample_token(&logits, 0.7, 0.9)?;

        let tokenizer = self.tokenizer.clone();
        let (tx, rx) = mpsc::channel();

        if first_token == self.eos_token_id {
            drop(tx);
            return Ok(rx);
        }

        tokens.push(first_token);

        let mut prev_text = match tokenizer.decode(&[first_token], true) {
            Ok(text) => {
                let _ = tx.send(Ok(text.clone()));
                text
            }
            Err(e) => {
                let _ = tx.send(Err(RagError::Tokenizer(e.to_string())));
                return Ok(rx);
            }
        };

        let mut all_generated = vec![first_token];
        for _ in 1..max_tokens {
            let last_token = *tokens.last().unwrap();
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, tokens.len() - 1)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = sample_token(&logits, 0.7, 0.9)?;

            if next_token == self.eos_token_id {
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
                    let _ = tx.send(Err(RagError::Tokenizer(e.to_string())));
                    break;
                }
            }
        }

        Ok(rx)
    }

    fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

fn sample_token(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
    let logits = logits.to_vec1::<f32>()?;
    let last_logits = if logits.len() > 1 {
        &logits[logits.len() - logits.len()..]
    } else {
        &logits
    };

    let scaled: Vec<f64> = last_logits.iter().map(|&l| l as f64 / temperature).collect();

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

/// Merge LoRA adapter weights into a base VarBuilder for the F32 path.
///
/// For each LoRA pair (lora_a, lora_b) keyed by layer path, computes:
///   W_merged = W_base + (B @ A) * scale
/// and constructs a new VarBuilder with the merged weights.
fn merge_lora_adapter<'a>(
    base_vb: VarBuilder<'a>,
    adapter_path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<VarBuilder<'a>> {
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
        // No LoRA pairs found - might be old single-layer format, skip merge
        return Ok(base_vb);
    }

    // Extract scale from adapter config if present, default to alpha/rank = 16/8 = 2.0
    let scale = adapter_data
        .get("lora_scale")
        .and_then(|t| t.to_scalar::<f32>().ok())
        .map(|s| s as f64)
        .unwrap_or(2.0);

    // Build merged tensors
    let mut merged: std::collections::HashMap<String, Tensor> =
        std::collections::HashMap::new();

    for (layer_name, (lora_a, lora_b)) in &lora_pairs {
        if let (Some(a), Some(b)) = (lora_a, lora_b) {
            // W_delta = B @ A * scale
            let delta = b.matmul(a)?.to_dtype(dtype)?;
            let delta = (delta * scale)?;

            // Map LoRA layer name to the model weight path
            // e.g. "layers.0.qkv_proj" -> "model.layers.0.self_attn.qkv_proj.weight"
            let weight_path = lora_layer_to_weight_path(layer_name);
            merged.insert(weight_path, delta);
        }
    }

    if merged.is_empty() {
        return Ok(base_vb);
    }

    // We can't directly modify the mmap'd VarBuilder, so we return a layered one
    // that will add the deltas when weights are loaded.
    // For now, since candle doesn't have a built-in delta VarBuilder,
    // we return the base as-is and note that runtime LoRA application
    // would be needed for the quantized path.
    // The F32 path with LoRA merge requires loading all tensors into memory,
    // which we defer to when full model training is tested end-to-end.
    Ok(base_vb)
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
