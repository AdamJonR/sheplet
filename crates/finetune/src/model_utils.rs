//! Shared utilities for LoRA model implementations.
//!
//! Contains helpers used by both Phi3 and Llama LoRA models.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;

/// Linear layer without bias.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(candle_nn::Linear::new(weight, None))
}

/// Repeat KV heads for grouped-query attention.
pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// Rotary position embedding, parameterized by raw values (not config structs).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Causal decoder attention mask (no sliding window).
pub fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..tgt_len)
        .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
        .collect();
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

/// Causal decoder attention mask with sliding window.
/// Tokens can only attend to positions within the last `window` tokens.
pub fn prepare_sliding_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total_len = tgt_len + seqlen_offset;
    let mask: Vec<_> = (0..tgt_len)
        .flat_map(|i| {
            let abs_i = i + seqlen_offset;
            (0..total_len).map(move |j| {
                let is_future = j > abs_i;
                let is_outside_window = j < abs_i.saturating_sub(window - 1);
                if is_future || is_outside_window {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    let mask = Tensor::from_slice(&mask, (tgt_len, total_len), device)?;
    mask.expand((b_size, 1, tgt_len, total_len))?
        .to_dtype(dtype)
}

/// Load tokenizer and SafeTensors files from a model directory.
/// Returns `(tokenizer, sorted safetensors paths)`.
pub fn load_model_files(
    model_dir: &std::path::Path,
) -> anyhow::Result<(tokenizers::Tokenizer, Vec<std::path::PathBuf>)> {
    let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        anyhow::bail!("No safetensors files found in {}", model_dir.display());
    }

    Ok((tokenizer, st_files))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_kv_no_repeat() {
        let device = Device::Cpu;
        let x = Tensor::rand(0.0f32, 1.0f32, &[1, 4, 8, 16], &device).unwrap();
        let out = repeat_kv(x.clone(), 1).unwrap();
        // Should return unchanged tensor
        assert_eq!(out.dims(), x.dims());
        let diff = (out - x)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_repeat_kv_doubles() {
        let device = Device::Cpu;
        // [batch=1, n_kv_head=2, seq_len=4, head_dim=8]
        let x = Tensor::rand(0.0f32, 1.0f32, &[1, 2, 4, 8], &device).unwrap();
        let out = repeat_kv(x, 2).unwrap();
        // n_kv_head * n_rep = 2 * 2 = 4
        assert_eq!(out.dims(), &[1, 4, 4, 8]);
    }

    #[test]
    fn test_decoder_attention_mask_shape() {
        let device = Device::Cpu;
        let mask = prepare_decoder_attention_mask(2, 5, 3, &device, DType::F32).unwrap();
        // [batch=2, 1, tgt_len=5, tgt_len + offset = 5+3 = 8]
        assert_eq!(mask.dims(), &[2, 1, 5, 8]);
    }

    #[test]
    fn test_decoder_attention_mask_is_causal() {
        let device = Device::Cpu;
        let mask = prepare_decoder_attention_mask(1, 4, 0, &device, DType::F32).unwrap();
        // Shape: [1, 1, 4, 4]
        let mask_2d: Vec<Vec<f32>> = mask.squeeze(0).unwrap().squeeze(0).unwrap().to_vec2().unwrap();
        // Lower triangle (i >= j) should be 0.0, upper triangle (i < j) should be -inf
        for i in 0..4 {
            for j in 0..4 {
                if i < j {
                    assert!(
                        mask_2d[i][j].is_infinite() && mask_2d[i][j] < 0.0,
                        "position [{i},{j}] should be -inf, got {}",
                        mask_2d[i][j]
                    );
                } else {
                    assert_eq!(
                        mask_2d[i][j], 0.0,
                        "position [{i},{j}] should be 0.0, got {}",
                        mask_2d[i][j]
                    );
                }
            }
        }
    }
}

/// Trait for LoRA-trainable models, enabling generic training functions.
pub trait LoraTrainable {
    fn device(&self) -> &Device;
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;
    fn clear_kv_cache(&mut self);
    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;
    fn forward_reference(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor>;
    /// Forward pass returning logits from `start_pos` onwards: [batch, len-start_pos, vocab_size].
    /// Used by DPO training which needs per-token log-probs for the response portion.
    fn forward_from(&mut self, input_ids: &Tensor, seqlen_offset: usize, start_pos: usize) -> Result<Tensor>;
    fn forward_reference_from(&mut self, input_ids: &Tensor, seqlen_offset: usize, start_pos: usize) -> Result<Tensor>;
    fn save_adapter(&self, path: &std::path::Path) -> anyhow::Result<()>;
    /// Get all LoRA tensors (a and b for each projection, each layer) for optimizer.
    fn lora_tensors(&self) -> Vec<Tensor>;
    /// Set LoRA tensors back after optimizer step (same order as lora_tensors).
    fn set_lora_tensors(&mut self, tensors: &[Tensor]);
}
