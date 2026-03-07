use std::path::Path;
use std::sync::mpsc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi3;
use tokenizers::Tokenizer;

use crate::error::{RagError, Result};

pub trait TextGenerator: Send {
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String>;
    fn generate_stream(
        &mut self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<mpsc::Receiver<Result<String>>>;
    fn clear_cache(&mut self);
}

pub struct PhiGenerator {
    model: phi3::Model,
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

        // Parse model config
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: phi3::Config = serde_json::from_str(&config_str)?;

        // Find safetensors files (may be sharded)
        let mut safetensors_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .is_some_and(|ext| ext == "safetensors")
            })
            .collect();
        safetensors_files.sort();

        if safetensors_files.is_empty() {
            return Err(RagError::Other(format!(
                "No safetensors files found in {}",
                model_dir.display()
            )));
        }

        let dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, device)?
        };

        // Load LoRA adapter if provided
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

        // Get EOS token ID
        let eos_token_id = tokenizer
            .token_to_id("<|end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2); // fallback

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

        // Process the prompt through the model
        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = sample_token(&logits, 0.7, 0.9)?;

        if next_token == self.eos_token_id {
            return Ok(generated);
        }
        generated.push(next_token);
        tokens.push(next_token);

        // Autoregressive generation
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

            // Safety: prevent runaway generation
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

        // Process the full prompt
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

        // Send first token
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

        // Generate remaining tokens one at a time, sending each immediately
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

            // Decode all generated tokens to get incremental text
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
        // Get last position's logits
        &logits[logits.len() - logits.len()..] // all logits for single position
    } else {
        &logits
    };

    // Apply temperature
    let scaled: Vec<f64> = last_logits.iter().map(|&l| l as f64 / temperature).collect();

    // Softmax
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

    // Top-p (nucleus) sampling
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

    // Renormalize and sample
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

fn merge_lora_adapter<'a>(
    base_vb: VarBuilder<'a>,
    _adapter_path: &Path,
    _device: &Device,
    _dtype: DType,
) -> Result<VarBuilder<'a>> {
    // For now, return the base VarBuilder as-is.
    // Full LoRA merge (W_merged = W_frozen + B @ A * scale) requires
    // iterating tensor names and performing the merge, which is model-specific.
    // The model will work without LoRA - just without the fine-tuning improvements.
    Ok(base_vb)
}
