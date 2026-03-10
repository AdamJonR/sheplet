//! Custom Gemma3 inference model with QK-normalization support.
//!
//! Based on candle's gemma2 model but adds q_norm/k_norm per-head RmsNorm
//! layers that are present in Gemma 3 weights. Without these norms,
//! attention scores can blow up or flatten, producing garbage output.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, Linear, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Gemma3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: Activation,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default)]
    pub attn_logit_softcapping: Option<f64>,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_local_base_freq: Option<f64>,
}

fn default_max_position_embeddings() -> usize {
    4096
}

fn default_query_pre_attn_scalar() -> usize {
    256
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// Gemma-style RMS normalization: (1.0 + weight) * x_normed
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
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, head_dim: usize, max_seq_len: usize, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = head_dim;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
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

    fn apply_rotary_emb_qkv(
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

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    sliding_window: Option<usize>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, sliding_window: Option<usize>, cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;

        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
            sliding_window,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-normalization before rotary embedding
        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;

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
        // For sliding window layers, truncate KV cache to the window size.
        // This ensures that during autoregressive generation (seq_len=1, no mask),
        // the layer only attends to recent tokens.
        let (key_states, value_states) = if let Some(w) = self.sliding_window {
            let kv_len = key_states.dim(2)?;
            if kv_len > w {
                let start = kv_len - w;
                (
                    key_states.narrow(2, start, w)?.contiguous()?,
                    value_states.narrow(2, start, w)?.contiguous()?,
                )
            } else {
                (key_states, value_states)
            }
        } else {
            (key_states, value_states)
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match self.attn_logit_softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, is_sliding: bool, sliding_window: Option<usize>, cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let sw = if is_sliding { sliding_window } else { None };
        let self_attn = Attention::new(rotary_emb, sw, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            is_sliding,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        full_mask: Option<&Tensor>,
        sliding_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let attention_mask = if self.is_sliding { sliding_mask } else { full_mask };
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct Gemma3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    final_logit_softcapping: Option<f64>,
    sliding_window: Option<usize>,
    device: Device,
    dtype: DType,
    hidden_size: usize,
}

impl Gemma3Model {
    pub fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // Create global RoPE (for full attention layers) and local RoPE (for sliding layers)
        let rotary_global = Arc::new(RotaryEmbedding::new(
            vb.dtype(), cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, vb_m.device(),
        )?);
        let local_theta = cfg.rope_local_base_freq.unwrap_or(cfg.rope_theta);
        let rotary_local = if (local_theta - cfg.rope_theta).abs() > 1e-6 {
            Arc::new(RotaryEmbedding::new(
                vb.dtype(), cfg.head_dim, cfg.max_position_embeddings, local_theta, vb_m.device(),
            )?)
        } else {
            rotary_global.clone()
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let is_sliding = if cfg.layer_types.is_empty() {
                false // no layer_types specified → treat all as full attention (backward compat)
            } else {
                cfg.layer_types.get(layer_idx).map(|s| s.as_str()) == Some("sliding_attention")
            };
            let rotary = if is_sliding { rotary_local.clone() } else { rotary_global.clone() };
            let layer = DecoderLayer::new(rotary, is_sliding, cfg.sliding_window, cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            final_logit_softcapping: cfg.final_logit_softcapping,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        let total_len = tgt_len + seqlen_offset;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                let abs_i = i + seqlen_offset;
                (0..total_len).map(move |j| {
                    let is_future = j > abs_i;
                    let is_outside_window = match sliding_window {
                        Some(w) => j < abs_i.saturating_sub(w - 1),
                        None => false,
                    };
                    if is_future || is_outside_window {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, total_len), &self.device)?;
        mask.expand((b_size, 1, tgt_len, total_len))?
            .to_dtype(self.dtype)
    }

    /// Check if any layer uses sliding attention.
    fn has_sliding_layers(&self) -> bool {
        self.layers.iter().any(|l| l.is_sliding)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let (full_mask, sliding_mask) = if seq_len <= 1 {
            (None, None)
        } else {
            let full = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset, None)?;
            let sliding = if self.has_sliding_layers() {
                Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset, self.sliding_window)?)
            } else {
                None
            };
            (Some(full), sliding)
        };
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, full_mask.as_ref(), sliding_mask.as_ref(), seqlen_offset)?;
        }
        // Tied lm_head: reuse embed_tokens weight
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .broadcast_matmul(&self.embed_tokens.embeddings().t()?)?;
        let logits = match self.final_logit_softcapping {
            None => logits,
            Some(sc) => ((logits / sc)?.tanh()? * sc)?,
        };
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma3_model_config_parse() {
        let json = r#"{
            "vocab_size": 262144,
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "attention_bias": false,
            "query_pre_attn_scalar": 256,
            "max_position_embeddings": 32768
        }"#;
        let config: Gemma3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.num_hidden_layers, 26);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.hidden_size, 1152);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 1);
        assert!(config.attn_logit_softcapping.is_none());
        assert!(config.final_logit_softcapping.is_none());
        // New fields should default when absent
        assert!(config.sliding_window.is_none());
        assert!(config.layer_types.is_empty());
        assert!(config.rope_local_base_freq.is_none());
    }

    #[test]
    fn test_gemma3_config_with_sliding_window() {
        let json = r#"{
            "vocab_size": 262144,
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 18,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "sliding_window": 512,
            "rope_local_base_freq": 10000.0,
            "layer_types": [
                "sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","global_attention",
                "sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","global_attention",
                "sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","sliding_attention","global_attention"
            ]
        }"#;
        let config: Gemma3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.sliding_window, Some(512));
        assert_eq!(config.rope_local_base_freq, Some(10000.0));
        assert_eq!(config.layer_types.len(), 18);
        assert_eq!(config.layer_types[0], "sliding_attention");
        assert_eq!(config.layer_types[5], "global_attention");
    }
}
