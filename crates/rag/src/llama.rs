//! Llama 3.2 full-precision inference model.
//!
//! Simpler than Gemma3: no QK-norm, no sliding window, no embedding scaling,
//! 2 norms per layer (not 4), standard RmsNorm (not Gemma's 1+weight variant).
//! Supports Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
}

fn default_rope_theta() -> f64 {
    500000.0
}

fn default_max_position_embeddings() -> usize {
    131072
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

impl LlamaConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, head_dim: usize, max_seq_len: usize, rope_theta: f64, dev: &Device) -> Result<Self> {
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
    fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
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
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
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

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

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
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
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
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = xs.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct LlamaModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: candle_nn::RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl LlamaModel {
    pub fn new(cfg: &LlamaConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let head_dim = cfg.head_dim();
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(), head_dim, cfg.max_position_embeddings, cfg.rope_theta, vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        // Llama 3.2 does NOT tie embeddings — lm_head has its own weight
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
        let total_len = tgt_len + seqlen_offset;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                let abs_i = i + seqlen_offset;
                (0..total_len).map(move |j| {
                    if j > abs_i { f32::NEG_INFINITY } else { 0. }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, total_len), &self.device)?;
        mask.expand((b_size, 1, tgt_len, total_len))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        let logits = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)?;
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
    fn test_llama_config_parse_1b() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": false,
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128008, 128009],
            "hidden_act": "silu",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "tie_word_embeddings": true,
            "vocab_size": 128256
        }"#;
        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.vocab_size, 128256);
        assert_eq!(config.head_dim(), 64); // 2048/32
        assert!(config.tie_word_embeddings);
        assert_eq!(config.rope_theta, 500000.0);
    }

    #[test]
    fn test_llama_config_parse_3b() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "model_type": "llama",
            "num_attention_heads": 24,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "tie_word_embeddings": false,
            "vocab_size": 128256
        }"#;
        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 24);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim(), 128); // 3072/24
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_llama_head_dim() {
        // 1B: 2048/32 = 64
        assert_eq!(2048 / 32, 64);
        // 3B: 3072/24 = 128
        assert_eq!(3072 / 24, 128);
    }

    #[test]
    fn test_rotary_embedding_shape() {
        let dev = Device::Cpu;
        let head_dim = 64;
        let max_seq_len = 128;
        let rope = RotaryEmbedding::new(DType::F32, head_dim, max_seq_len, 500000.0, &dev).unwrap();
        // sin/cos: [max_seq_len, head_dim/2]
        assert_eq!(rope.sin.dims(), &[max_seq_len, head_dim / 2]);
        assert_eq!(rope.cos.dims(), &[max_seq_len, head_dim / 2]);
    }

    #[test]
    fn test_rotary_embedding_values() {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(DType::F32, 64, 128, 500000.0, &dev).unwrap();
        // At position 0, cos should be 1.0 and sin should be 0.0
        let cos_0: Vec<f32> = rope.cos.get(0).unwrap().to_vec1().unwrap();
        let sin_0: Vec<f32> = rope.sin.get(0).unwrap().to_vec1().unwrap();
        for &c in &cos_0 {
            assert!((c - 1.0).abs() < 1e-5, "cos[0] should be 1.0, got {c}");
        }
        for &s in &sin_0 {
            assert!(s.abs() < 1e-5, "sin[0] should be 0.0, got {s}");
        }
    }
}
