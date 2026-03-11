//! Forked from candle-transformers phi3.rs to support:
//! - Partial rotary embeddings (partial_rotary_factor < 1.0)
//! - LongRoPE scaling (per-dimension frequency scaling factors)
//! - Tied word embeddings (tie_word_embeddings)
//!
//! These features are required for Phi-4-mini-instruct full-precision inference.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScaling {
    pub long_factor: Vec<f64>,
    pub short_factor: Vec<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub max_position_embeddings: usize,
    #[serde(default = "default_original_max_position_embeddings")]
    pub original_max_position_embeddings: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_original_max_position_embeddings() -> usize {
    4096
}

fn default_partial_rotary_factor() -> f64 {
    1.0
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn rope_dim(&self) -> usize {
        (self.head_dim() as f64 * self.partial_rotary_factor) as usize
    }
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight.to_dtype(x_dtype)?)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rope_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let rope_dim = cfg.rope_dim();
        // For LongRoPE: use short_factor for sequences within original context length.
        // Our inference never exceeds original_max_position_embeddings (4096).
        let max_seq_len = cfg.original_max_position_embeddings;

        // Base inverse frequencies for rope_dim (not full head_dim)
        let mut inv_freq: Vec<f32> = (0..rope_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / rope_dim as f64) as f32)
            .collect();

        // Apply short_factor scaling (all 1.0 for Phi-4-mini, effectively a no-op)
        if let Some(ref scaling) = cfg.rope_scaling {
            let factors = &scaling.short_factor;
            for (i, freq) in inv_freq.iter_mut().enumerate() {
                if i < factors.len() {
                    *freq /= factors[i] as f32;
                }
            }
        }

        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        // Compute LongRoPE attention_factor scaling
        let attention_factor = if cfg.rope_scaling.is_some() {
            let factor = cfg.max_position_embeddings as f64
                / cfg.original_max_position_embeddings as f64;
            if factor <= 1.0 {
                1.0
            } else {
                (1.0 + factor.ln() / (cfg.original_max_position_embeddings as f64).ln()).sqrt()
            }
        } else {
            1.0
        };

        Ok(Self {
            sin: (freqs.sin()? * attention_factor)?,
            cos: (freqs.cos()? * attention_factor)?,
            rope_dim,
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        if self.rope_dim < n_embd {
            // Partial rotary: apply RoPE only to first rope_dim dimensions
            let q_rot = q.narrow(D::Minus1, 0, self.rope_dim)?.contiguous()?;
            let q_pass = q.narrow(D::Minus1, self.rope_dim, n_embd - self.rope_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, self.rope_dim)?.contiguous()?;
            let k_pass = k.narrow(D::Minus1, self.rope_dim, n_embd - self.rope_dim)?;

            let q_rot = candle_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot, &cos, &sin)?;

            let q_embed = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;
            let k_embed = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
            Ok((q_embed, k_embed))
        } else {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        }
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear_no_bias(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            kv_cache: None,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear_no_bias(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(i_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = xs.apply(&self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        up_states.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.post_attention_layernorm)?
            .apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phi4_mini_config() -> Config {
        Config {
            vocab_size: 200064,
            hidden_act: candle_nn::Activation::Silu,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            bos_token_id: Some(199999),
            eos_token_id: Some(199999),
            rope_scaling: Some(RopeScaling {
                long_factor: vec![1.0; 48],  // 48 = rope_dim/2 = 96/2
                short_factor: vec![1.0; 48],
            }),
            max_position_embeddings: 131072,
            original_max_position_embeddings: 4096,
            partial_rotary_factor: 0.75,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn test_rotary_table_uses_original_max_position_embeddings() {
        let cfg = phi4_mini_config();
        let dev = Device::Cpu;
        let rotary = RotaryEmbedding::new(DType::F32, &cfg, &dev).unwrap();
        // Table should be sized to original_max_position_embeddings (4096), not max_position_embeddings (131072)
        let sin_shape = rotary.sin.dims();
        assert_eq!(sin_shape[0], 4096, "sin table should have 4096 rows (original_max_position_embeddings)");
        let cos_shape = rotary.cos.dims();
        assert_eq!(cos_shape[0], 4096, "cos table should have 4096 rows (original_max_position_embeddings)");
    }

    #[test]
    fn test_rotary_uses_short_factor_not_long_factor() {
        // Create config where short_factor=1.0 and long_factor=50.0
        // If long_factor were used, frequencies would be drastically different
        let mut cfg = phi4_mini_config();
        cfg.rope_scaling = Some(RopeScaling {
            long_factor: vec![50.0; 48],
            short_factor: vec![1.0; 48],
        });
        let dev = Device::Cpu;
        let rotary_with_scaling = RotaryEmbedding::new(DType::F32, &cfg, &dev).unwrap();

        // Compare against no rope_scaling (vanilla RoPE)
        let mut cfg_vanilla = cfg.clone();
        cfg_vanilla.rope_scaling = None;
        let rotary_vanilla = RotaryEmbedding::new(DType::F32, &cfg_vanilla, &dev).unwrap();

        // With short_factor=1.0 + attention_factor, the base frequencies should match vanilla
        // (only the attention_factor scaling differs). If long_factor=50.0 were used,
        // the values would be completely different.
        let cos_scaled = rotary_with_scaling.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cos_vanilla = rotary_vanilla.cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Compute expected attention_factor for this config
        let factor = 131072.0_f64 / 4096.0_f64; // 32
        let attention_factor = (1.0 + factor.ln() / 4096.0_f64.ln()).sqrt();

        // cos_scaled should equal cos_vanilla * attention_factor (within float tolerance)
        for (i, (scaled, vanilla)) in cos_scaled.iter().zip(cos_vanilla.iter()).enumerate().take(100) {
            let expected = *vanilla as f64 * attention_factor;
            let diff = (*scaled as f64 - expected).abs();
            assert!(diff < 1e-5, "cos mismatch at index {i}: scaled={scaled}, expected={expected}");
        }
    }

    #[test]
    fn test_rotary_attention_factor_applied() {
        let cfg = phi4_mini_config();
        let dev = Device::Cpu;
        let rotary = RotaryEmbedding::new(DType::F32, &cfg, &dev).unwrap();

        // For position 0, cos should be attention_factor * 1.0 (since cos(0)=1)
        let cos_row0 = rotary.cos.get(0).unwrap().to_vec1::<f32>().unwrap();
        let factor = 131072.0_f64 / 4096.0_f64;
        let attention_factor = (1.0 + factor.ln() / 4096.0_f64.ln()).sqrt();

        // All cos values at position 0 should be attention_factor (cos(0)=1)
        for (i, val) in cos_row0.iter().enumerate() {
            let diff = (*val as f64 - attention_factor).abs();
            assert!(diff < 1e-5, "cos[0][{i}] = {val}, expected {attention_factor}");
        }

        // sin at position 0 should be 0 * attention_factor = 0
        let sin_row0 = rotary.sin.get(0).unwrap().to_vec1::<f32>().unwrap();
        for (i, val) in sin_row0.iter().enumerate() {
            assert!(val.abs() < 1e-6, "sin[0][{i}] = {val}, expected ~0");
        }
    }

    #[test]
    fn test_rotary_no_scaling_when_no_rope_config() {
        let mut cfg = phi4_mini_config();
        cfg.rope_scaling = None;
        cfg.max_position_embeddings = 4096;
        let dev = Device::Cpu;
        let rotary = RotaryEmbedding::new(DType::F32, &cfg, &dev).unwrap();

        // Without rope_scaling, attention_factor=1.0, so cos[0] should be exactly 1.0
        let cos_row0 = rotary.cos.get(0).unwrap().to_vec1::<f32>().unwrap();
        for (i, val) in cos_row0.iter().enumerate() {
            let diff = (*val - 1.0).abs();
            assert!(diff < 1e-6, "cos[0][{i}] = {val}, expected 1.0 (no scaling)");
        }
    }

    #[test]
    fn test_rotary_partial_rotary_dim() {
        let cfg = phi4_mini_config();
        let dev = Device::Cpu;
        let rotary = RotaryEmbedding::new(DType::F32, &cfg, &dev).unwrap();
        // head_dim = 3072/24 = 128, partial_rotary_factor = 0.75, rope_dim = 96
        assert_eq!(rotary.rope_dim, 96);
        // cos/sin should have rope_dim/2 = 48 columns
        assert_eq!(rotary.cos.dims()[1], 48);
        assert_eq!(rotary.sin.dims()[1], 48);
    }
}
