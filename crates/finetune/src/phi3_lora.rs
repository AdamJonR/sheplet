//! Phi-3 model with LoRA injection for fine-tuning.
//!
//! This is a fork of candle-transformers phi3.rs with the attention projections
//! wrapped in LoraLinear for gradient-tracked training.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};

use crate::lora::{LoraConfig, LoraLinear};

/// Phi-3 config (same structure as candle_transformers::models::phi3::Config).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Phi3Config {
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
    pub rope_scaling: Option<String>,
    pub max_position_embeddings: usize,
}

impl Phi3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Phi3Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
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

/// RMS normalization (simplified, no tracing).
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(xs, &self.weight, self.eps as f32)
    }
}

/// Linear layer without bias (simplified, no tracing).
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(candle_nn::Linear::new(weight, None))
}

/// Attention block with LoRA on the qkv and output projections.
struct LoraAttention {
    qkv_proj: LoraLinear,
    o_proj: LoraLinear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LoraAttention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Phi3Config,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;

        let qkv_frozen = linear_no_bias(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let qkv_proj = LoraLinear::new(qkv_frozen, cfg.hidden_size, op_size, lora_cfg, device)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let o_size = num_heads * head_dim;
        let o_frozen = linear_no_bias(o_size, cfg.hidden_size, vb.pp("o_proj"))?;
        let o_proj = LoraLinear::new(o_frozen, o_size, cfg.hidden_size, lora_cfg, device)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

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
        use_lora: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let qkv = if use_lora {
            self.qkv_proj
                .forward(xs)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        } else {
            self.qkv_proj
                .forward_frozen_only(xs)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        };

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

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?;

        if use_lora {
            self.o_proj
                .forward(&attn_output)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        } else {
            self.o_proj
                .forward_frozen_only(&attn_output)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        }
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
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

struct Mlp {
    gate_up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Phi3Config, vb: VarBuilder) -> Result<Self> {
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

struct DecoderLayer {
    self_attn: LoraAttention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Phi3Config,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let self_attn =
            LoraAttention::new(rotary_emb, cfg, lora_cfg, vb.pp("self_attn"), device)?;
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
        use_lora: bool,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offset, use_lora)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.post_attention_layernorm)?
            .apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Phi-3 model with LoRA layers for fine-tuning.
pub struct Phi3LoraModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    device: Device,
    dtype: DType,
}

impl Phi3LoraModel {
    pub fn new(
        cfg: &Phi3Config,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, lora_cfg, vb_l.pp(layer_idx), device)?;
            layers.push(layer);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
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

    /// Forward pass with LoRA enabled (policy model).
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, true)
    }

    /// Forward pass with LoRA disabled (reference model for DPO).
    pub fn forward_reference(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, false)
    }

    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        use_lora: bool,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset, use_lora)?;
        }
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    /// Count the number of trainable LoRA parameters.
    pub fn lora_param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                let qkv = layer.self_attn.qkv_proj.trainable_tensors();
                let o = layer.self_attn.o_proj.trainable_tensors();
                qkv.iter().chain(o.iter()).map(|t| t.elem_count()).sum::<usize>()
            })
            .sum()
    }

    /// Save all LoRA adapter weights to a safetensors file.
    pub fn save_adapter(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let mut tensors = HashMap::new();
        let scale_val = if let Some(layer) = self.layers.first() {
            layer.self_attn.qkv_proj.scale()
        } else {
            2.0
        };

        for (i, layer) in self.layers.iter().enumerate() {
            tensors.insert(
                format!("layers.{i}.qkv_proj.lora_a"),
                layer.self_attn.qkv_proj.lora_a().clone(),
            );
            tensors.insert(
                format!("layers.{i}.qkv_proj.lora_b"),
                layer.self_attn.qkv_proj.lora_b().clone(),
            );
            tensors.insert(
                format!("layers.{i}.o_proj.lora_a"),
                layer.self_attn.o_proj.lora_a().clone(),
            );
            tensors.insert(
                format!("layers.{i}.o_proj.lora_b"),
                layer.self_attn.o_proj.lora_b().clone(),
            );
        }

        // Store the scale as a scalar tensor
        let scale_tensor =
            Tensor::from_vec(vec![scale_val as f32], &[1], &Device::Cpu)?;
        tensors.insert("lora_scale".to_string(), scale_tensor);

        candle_core::safetensors::save(&tensors, path)
            .map_err(|e| anyhow::anyhow!("failed to save adapter: {e}"))?;
        Ok(())
    }
}

/// High-level trainer wrapping a Phi3LoraModel with tokenizer.
pub struct Phi3LoraTrainer {
    pub model: Phi3LoraModel,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
}

impl Phi3LoraTrainer {
    /// Load a pre-trained Phi-3 model with LoRA layers initialized.
    pub fn new(
        model_dir: &std::path::Path,
        lora_cfg: &LoraConfig,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Phi3Config = serde_json::from_str(&config_str)?;

        // Find safetensors files
        let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        st_files.sort();

        if st_files.is_empty() {
            anyhow::bail!("No safetensors files found in {}", model_dir.display());
        }

        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&st_files, dtype, device)? };

        let model = Phi3LoraModel::new(&config, lora_cfg, vb, device)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi3_config_parse() {
        let json = r#"{
            "vocab_size": 32064,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096
        }"#;
        let config: Phi3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_dim(), 96);
        assert_eq!(config.num_hidden_layers, 32);
    }
}
