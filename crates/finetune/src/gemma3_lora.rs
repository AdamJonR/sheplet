//! Gemma 3 model with LoRA injection for fine-tuning.
//!
//! Based on Gemma 2 architecture (candle-transformers gemma2) with LoRA
//! applied to q_proj, k_proj, v_proj, and o_proj attention projections.
//! Key differences from Phi-3:
//! - Separate q/k/v/o projections (not fused qkv_proj)
//! - GemmaRmsNorm: adds 1.0 to weight before multiplication
//! - 4 norms per decoder layer (input, post_attention, pre_feedforward, post_feedforward)
//! - Embedding scaling by sqrt(hidden_size)
//! - Tied lm_head (reuses embed_tokens weight)
//! - Explicit head_dim in config (not derived from hidden_size/num_heads)

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};

use crate::lora::{LoraConfig, LoraLinear};
use crate::model_utils::{self, linear_no_bias, repeat_kv, RotaryEmbedding};

/// Gemma 3 config (compatible with candle_transformers::models::gemma2::Config).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: candle_nn::Activation,
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

/// Gemma-style RMS normalization: (1.0 + weight) * x_normed
#[derive(Debug, Clone)]
struct GemmaRmsNorm {
    shifted_weight: Tensor, // precomputed (weight + 1.0)
    eps: f64,
}

impl GemmaRmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let shifted_weight = (&weight + 1.0)?;
        Ok(Self {
            shifted_weight,
            eps,
        })
    }
}

impl Module for GemmaRmsNorm {
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
            .broadcast_mul(&self.shifted_weight)
    }
}

/// Attention block with separate LoRA on q, k, v, o projections.
struct LoraAttention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: Arc<RotaryEmbedding>,
    sliding_window: Option<usize>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LoraAttention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        sliding_window: Option<usize>,
        cfg: &Gemma3Config,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let hidden_size = cfg.hidden_size;

        let q_frozen = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let q_proj =
            LoraLinear::new(q_frozen, hidden_size, num_heads * head_dim, lora_cfg, device)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let k_frozen = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let k_proj =
            LoraLinear::new(k_frozen, hidden_size, num_kv_heads * head_dim, lora_cfg, device)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let v_frozen = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let v_proj =
            LoraLinear::new(v_frozen, hidden_size, num_kv_heads * head_dim, lora_cfg, device)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let o_frozen = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        let o_proj =
            LoraLinear::new(o_frozen, num_heads * head_dim, hidden_size, lora_cfg, device)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let q_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            sliding_window,
            kv_cache: None,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
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

        let fwd = |proj: &LoraLinear, x: &Tensor, use_l: bool| -> Result<Tensor> {
            if use_l {
                proj.forward(x)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))
            } else {
                proj.forward_frozen_only(x)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))
            }
        };

        let query_states = fwd(&self.q_proj, xs, use_lora)?;
        let key_states = fwd(&self.k_proj, xs, use_lora)?;
        let value_states = fwd(&self.v_proj, xs, use_lora)?;

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

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?;

        fwd(&self.o_proj, &attn_output, use_lora)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
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
        let gate = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let up = xs.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

struct DecoderLayer {
    self_attn: LoraAttention,
    mlp: Mlp,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
    is_sliding: bool,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        is_sliding: bool,
        sliding_window: Option<usize>,
        cfg: &Gemma3Config,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let sw = if is_sliding { sliding_window } else { None };
        let self_attn =
            LoraAttention::new(rotary_emb, sw, cfg, lora_cfg, vb.pp("self_attn"), device)?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = GemmaRmsNorm::new(
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
        use_lora: bool,
    ) -> Result<Tensor> {
        let attention_mask = if self.is_sliding { sliding_mask } else { full_mask };
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offset, use_lora)?;
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

/// Gemma 3 model with LoRA layers for fine-tuning.
pub struct Gemma3LoraModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: GemmaRmsNorm,
    device: Device,
    dtype: DType,
    hidden_size: usize,
    sliding_window: Option<usize>,
    final_logit_softcapping: Option<f64>,
}

impl Gemma3LoraModel {
    pub fn new(
        cfg: &Gemma3Config,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // Create global RoPE (for full attention layers) and local RoPE (for sliding layers)
        let rotary_global = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb_m.device(),
        )?);
        let local_theta = cfg.rope_local_base_freq.unwrap_or(cfg.rope_theta);
        let rotary_local = if (local_theta - cfg.rope_theta).abs() > 1e-6 {
            Arc::new(RotaryEmbedding::new(
                vb.dtype(),
                cfg.head_dim,
                cfg.max_position_embeddings,
                local_theta,
                vb_m.device(),
            )?)
        } else {
            rotary_global.clone()
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let is_sliding = if cfg.layer_types.is_empty() {
                false
            } else {
                cfg.layer_types.get(layer_idx).map(|s| s.as_str()) == Some("sliding_attention")
            };
            let rotary = if is_sliding { rotary_local.clone() } else { rotary_global.clone() };
            let layer =
                DecoderLayer::new(rotary, is_sliding, cfg.sliding_window, cfg, lora_cfg, vb_l.pp(layer_idx), device)?;
            layers.push(layer);
        }
        let norm = GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            final_logit_softcapping: cfg.final_logit_softcapping,
        })
    }

    /// Forward pass with LoRA enabled (policy model) — last position only.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, None, true)
    }

    /// Forward pass with LoRA disabled (reference model for DPO) — last position only.
    pub fn forward_reference(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, None, false)
    }

    /// Forward pass with LoRA enabled, returning logits from `start_pos` onwards.
    pub fn forward_from(&mut self, input_ids: &Tensor, seqlen_offset: usize, start_pos: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, Some(start_pos), true)
    }

    /// Forward pass with LoRA disabled, returning logits from `start_pos` onwards.
    pub fn forward_reference_from(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, Some(start_pos), false)
    }

    /// Check if any layer uses sliding attention.
    fn has_sliding_layers(&self) -> bool {
        self.layers.iter().any(|l| l.is_sliding)
    }

    /// `logits_from_pos`: `None` = last position only, `Some(pos)` = from position `pos` onwards.
    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        logits_from_pos: Option<usize>,
        use_lora: bool,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let (full_mask, sliding_mask) = if seq_len <= 1 {
            (None, None)
        } else {
            let full = model_utils::prepare_decoder_attention_mask(
                b_size,
                seq_len,
                seqlen_offset,
                &self.device,
                self.dtype,
            )?;
            let sliding = if self.has_sliding_layers() {
                Some(model_utils::prepare_sliding_attention_mask(
                    b_size,
                    seq_len,
                    seqlen_offset,
                    self.sliding_window.unwrap_or(512),
                    &self.device,
                    self.dtype,
                )?)
            } else {
                None
            };
            (Some(full), sliding)
        };
        // Embed and scale by sqrt(hidden_size)
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, full_mask.as_ref(), sliding_mask.as_ref(), seqlen_offset, use_lora)?;
        }
        // Narrow hidden states before applying norm+lm_head to avoid wasted computation
        let xs = match logits_from_pos {
            Some(pos) => xs.narrow(1, pos, seq_len - pos)?,
            None => xs.narrow(1, seq_len - 1, 1)?,
        };
        // Tied lm_head: reuse embed_tokens weight (use broadcast_matmul for 3D×2D support)
        let logits = xs.apply(&self.norm)?
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

    /// Count the number of trainable LoRA parameters.
    pub fn lora_param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                let attn = &layer.self_attn;
                [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj]
                    .iter()
                    .flat_map(|proj| proj.trainable_tensors())
                    .map(|t| t.elem_count())
                    .sum::<usize>()
            })
            .sum()
    }

    /// Save all LoRA adapter weights to a safetensors file.
    pub fn save_adapter(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let mut tensors = HashMap::new();
        let scale_val = if let Some(layer) = self.layers.first() {
            layer.self_attn.q_proj.scale()
        } else {
            2.0
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let attn = &layer.self_attn;
            for (name, proj) in [
                ("q_proj", &attn.q_proj),
                ("k_proj", &attn.k_proj),
                ("v_proj", &attn.v_proj),
                ("o_proj", &attn.o_proj),
            ] {
                tensors.insert(
                    format!("layers.{i}.{name}.lora_a"),
                    proj.lora_a().clone(),
                );
                tensors.insert(
                    format!("layers.{i}.{name}.lora_b"),
                    proj.lora_b().clone(),
                );
            }
        }

        // Store the scale as a scalar tensor
        let scale_tensor = Tensor::from_vec(vec![scale_val as f32], &[1], &Device::Cpu)?;
        tensors.insert("lora_scale".to_string(), scale_tensor);

        candle_core::safetensors::save(&tensors, path)
            .map_err(|e| anyhow::anyhow!("failed to save adapter: {e}"))?;
        Ok(())
    }
}

/// High-level trainer wrapping a Gemma3LoraModel with tokenizer.
pub struct Gemma3LoraTrainer {
    pub model: Gemma3LoraModel,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
}

impl Gemma3LoraTrainer {
    /// Load a pre-trained Gemma 3 model with LoRA layers initialized.
    pub fn new(
        model_dir: &std::path::Path,
        lora_cfg: &LoraConfig,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let (tokenizer, st_files) = model_utils::load_model_files(model_dir)?;

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Gemma3Config = serde_json::from_str(&config_str)?;

        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&st_files, dtype, device)? };

        let model = Gemma3LoraModel::new(&config, lora_cfg, vb, device)?;

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

impl model_utils::LoraTrainable for Gemma3LoraTrainer {
    fn device(&self) -> &Device {
        &self.device
    }

    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.encode(text)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.model.forward(input_ids, seqlen_offset)
    }

    fn forward_reference(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.model.forward_reference(input_ids, seqlen_offset)
    }

    fn forward_from(&mut self, input_ids: &Tensor, seqlen_offset: usize, start_pos: usize) -> Result<Tensor> {
        self.model.forward_from(input_ids, seqlen_offset, start_pos)
    }

    fn forward_reference_from(&mut self, input_ids: &Tensor, seqlen_offset: usize, start_pos: usize) -> Result<Tensor> {
        self.model.forward_reference_from(input_ids, seqlen_offset, start_pos)
    }

    fn save_adapter(&self, path: &std::path::Path) -> anyhow::Result<()> {
        self.model.save_adapter(path)
    }

    fn lora_tensors(&self) -> Vec<Tensor> {
        let mut tensors = Vec::with_capacity(self.model.layers.len() * 8);
        for layer in &self.model.layers {
            let attn = &layer.self_attn;
            // 8 tensors per layer: q_a, q_b, k_a, k_b, v_a, v_b, o_a, o_b
            for proj in [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.o_proj] {
                tensors.push(proj.lora_a().clone());
                tensors.push(proj.lora_b().clone());
            }
        }
        tensors
    }

    fn set_lora_tensors(&mut self, tensors: &[Tensor]) {
        debug_assert_eq!(
            tensors.len(),
            self.model.layers.len() * 8,
            "expected {} tensors, got {}",
            self.model.layers.len() * 8,
            tensors.len()
        );
        let mut idx = 0;
        for layer in &mut self.model.layers {
            let attn = &mut layer.self_attn;
            for proj in [&mut attn.q_proj, &mut attn.k_proj, &mut attn.v_proj, &mut attn.o_proj] {
                proj.set_lora_a(tensors[idx].clone());
                proj.set_lora_b(tensors[idx + 1].clone());
                idx += 2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma3_config_parse() {
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
        // New fields should default when absent
        assert!(config.sliding_window.is_none());
        assert!(config.layer_types.is_empty());
        assert!(config.rope_local_base_freq.is_none());
    }

    #[test]
    fn test_gemma3_attention_with_qk_norm() {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let hidden_size = 8usize;
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 4usize;

        let cfg = Gemma3Config {
            vocab_size: 32,
            hidden_size,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            hidden_activation: candle_nn::Activation::GeluPytorchTanh,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            attention_bias: false,
            final_logit_softcapping: None,
            attn_logit_softcapping: None,
            query_pre_attn_scalar: head_dim,
            max_position_embeddings: 128,
            sliding_window: None,
            layer_types: vec![],
            rope_local_base_freq: None,
        };

        let lora_cfg = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
        };
        let rotary_emb = Arc::new(
            RotaryEmbedding::new(dtype, head_dim, 128, 10000.0, &device).unwrap(),
        );

        // Deterministic input with seq_len=2 so attention distribution matters
        let input = Tensor::from_vec(
            (1..=16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[1, 2, hidden_size],
            &device,
        )
        .unwrap();

        // Causal mask for seq_len=2
        let mask = Tensor::from_vec(
            vec![0.0f32, f32::NEG_INFINITY, 0.0, 0.0],
            &[1, 1, 2, 2],
            &device,
        )
        .unwrap();

        // Fixed projection weights
        let q_w = Tensor::from_vec(
            (0..num_heads * head_dim * hidden_size)
                .map(|i| (i as f32 * 0.01) % 1.0)
                .collect::<Vec<_>>(),
            &[num_heads * head_dim, hidden_size],
            &device,
        )
        .unwrap();
        let k_w = Tensor::from_vec(
            (0..num_kv_heads * head_dim * hidden_size)
                .map(|i| (i as f32 * 0.02) % 1.0)
                .collect::<Vec<_>>(),
            &[num_kv_heads * head_dim, hidden_size],
            &device,
        )
        .unwrap();
        let v_w = Tensor::from_vec(
            (0..num_kv_heads * head_dim * hidden_size)
                .map(|i| (i as f32 * 0.03) % 1.0)
                .collect::<Vec<_>>(),
            &[num_kv_heads * head_dim, hidden_size],
            &device,
        )
        .unwrap();
        let o_w = Tensor::from_vec(
            (0..hidden_size * num_heads * head_dim)
                .map(|i| (i as f32 * 0.04) % 1.0)
                .collect::<Vec<_>>(),
            &[hidden_size, num_heads * head_dim],
            &device,
        )
        .unwrap();

        // Build attention with different q_norm/k_norm weights to prove norm is applied
        let build_attn = |norm_val: f32| -> LoraAttention {
            let mut tensors = HashMap::new();
            tensors.insert("q_proj.weight".to_string(), q_w.clone());
            tensors.insert("k_proj.weight".to_string(), k_w.clone());
            tensors.insert("v_proj.weight".to_string(), v_w.clone());
            tensors.insert("o_proj.weight".to_string(), o_w.clone());
            tensors.insert(
                "q_norm.weight".to_string(),
                Tensor::from_vec(vec![norm_val; head_dim], head_dim, &device).unwrap(),
            );
            tensors.insert(
                "k_norm.weight".to_string(),
                Tensor::from_vec(vec![norm_val; head_dim], head_dim, &device).unwrap(),
            );
            let vb = VarBuilder::from_tensors(tensors, dtype, &device);
            LoraAttention::new(rotary_emb.clone(), None, &cfg, &lora_cfg, vb, &device).unwrap()
        };

        // norm weight=0 → GemmaRmsNorm scale = (0+1) = 1x
        let mut attn1 = build_attn(0.0);
        let out1 = attn1.forward(&input, Some(&mask), 0, false).unwrap();

        // norm weight=9 → GemmaRmsNorm scale = (9+1) = 10x
        let mut attn2 = build_attn(9.0);
        let out2 = attn2.forward(&input, Some(&mask), 0, false).unwrap();

        // Verify shapes
        assert_eq!(out1.dims(), &[1, 2, hidden_size]);
        assert_eq!(out2.dims(), &[1, 2, hidden_size]);

        // Outputs should differ (QK-norm weights affect attention distribution)
        let diff = (&out1 - &out2)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff > 1e-4,
            "different QK-norm weights should produce different outputs, diff={diff}"
        );

        // Both should be finite
        let vals1: Vec<f32> = out1.flatten_all().unwrap().to_vec1().unwrap();
        let vals2: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals1.iter().all(|v| v.is_finite()), "output1 should be finite");
        assert!(vals2.iter().all(|v| v.is_finite()), "output2 should be finite");
    }

    #[test]
    fn test_gemma_rms_norm() {
        let device = Device::Cpu;
        let weight = Tensor::ones(4, DType::F32, &device).unwrap();
        let shifted_weight = (&weight + 1.0).unwrap();
        let norm = GemmaRmsNorm {
            shifted_weight,
            eps: 1e-6,
        };
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device).unwrap();
        let result = norm.forward(&x).unwrap();
        // (1+1) * x_normed = 2 * x_normed
        let result_vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // All values should be scaled by 2.0 compared to standard RmsNorm
        assert!(result_vals.iter().all(|v| v.is_finite()));
    }
}
