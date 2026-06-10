//! Mistral model with LoRA injection for fine-tuning.
//!
//! Nearly identical to Llama: separate q/k/v/o projections, no bias,
//! standard RmsNorm, SiLU MLP. Sliding window attention is handled
//! internally during inference (not in LoRA trainer).

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};

use crate::lora::{LoraConfig, LoraLinear};
use crate::model_utils::{self, linear_no_bias, repeat_kv, RotaryEmbedding};

/// Mistral config for LoRA training.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MistralLoraConfig {
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
    pub hidden_act: candle_nn::Activation,
    pub head_dim: Option<usize>,
}

fn default_rope_theta() -> f64 {
    1000000.0
}

fn default_max_position_embeddings() -> usize {
    32768
}

fn default_hidden_act() -> candle_nn::Activation {
    candle_nn::Activation::Silu
}

impl MistralLoraConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// Attention block with separate LoRA on q, k, v, o projections.
struct LoraAttention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
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
        cfg: &MistralLoraConfig,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
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

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
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
    fn new(cfg: &MistralLoraConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;
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
        let gate = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let up = xs.apply(&self.up_proj)?;
        (gate * up)?.apply(&self.down_proj)
    }
}

struct DecoderLayer {
    self_attn: LoraAttention,
    mlp: Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &MistralLoraConfig,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let self_attn =
            LoraAttention::new(rotary_emb, cfg, lora_cfg, vb.pp("self_attn"), device)?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
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
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = xs.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Mistral model with LoRA layers for fine-tuning.
pub struct MistralLoraModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: candle_nn::RmsNorm,
    lm_head: candle_nn::Linear,
    device: Device,
    dtype: DType,
}

impl MistralLoraModel {
    pub fn new(
        cfg: &MistralLoraConfig,
        lora_cfg: &LoraConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let head_dim = cfg.head_dim();
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer =
                DecoderLayer::new(rotary_emb.clone(), cfg, lora_cfg, vb_l.pp(layer_idx), device)?;
            layers.push(layer);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
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

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, None, true)
    }

    pub fn forward_reference(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, None, false)
    }

    pub fn forward_from(&mut self, input_ids: &Tensor, seqlen_offset: usize, start_pos: usize) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, Some(start_pos), true)
    }

    pub fn forward_reference_from(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        self.forward_inner(input_ids, seqlen_offset, Some(start_pos), false)
    }

    fn forward_inner(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        logits_from_pos: Option<usize>,
        use_lora: bool,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(model_utils::prepare_decoder_attention_mask(
                b_size,
                seq_len,
                seqlen_offset,
                &self.device,
                self.dtype,
            )?)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset, use_lora)?;
        }
        let xs = match logits_from_pos {
            Some(pos) => xs.narrow(1, pos, seq_len - pos)?,
            None => xs.narrow(1, seq_len - 1, 1)?,
        };
        let logits = xs.apply(&self.norm)?.apply(&self.lm_head)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

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

        let scale_tensor = Tensor::from_vec(vec![scale_val as f32], &[1], &Device::Cpu)?;
        tensors.insert("lora_scale".to_string(), scale_tensor);

        candle_core::safetensors::save(&tensors, path)
            .map_err(|e| anyhow::anyhow!("failed to save adapter: {e}"))?;
        Ok(())
    }
}

/// High-level trainer wrapping a MistralLoraModel with tokenizer.
pub struct MistralLoraTrainer {
    pub model: MistralLoraModel,
    pub tokenizer: tokenizers::Tokenizer,
    pub device: Device,
}

impl MistralLoraTrainer {
    pub fn new(
        model_dir: &std::path::Path,
        lora_cfg: &LoraConfig,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let (tokenizer, st_files) = model_utils::load_model_files(model_dir)?;

        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: MistralLoraConfig = serde_json::from_str(&config_str)?;

        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&st_files, dtype, device)? };

        let model = MistralLoraModel::new(&config, lora_cfg, vb, device)?;

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

    pub fn encode_prompt(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }
}

impl model_utils::LoraTrainable for MistralLoraTrainer {
    fn device(&self) -> &Device {
        &self.device
    }

    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.encode(text)
    }

    fn encode_prompt(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.encode_prompt(text)
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
    fn test_mistral_lora_config_parse() {
        let json = r#"{
            "vocab_size": 32768,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 32768,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "sliding_window": 4096
        }"#;
        let config: MistralLoraConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.head_dim(), 128);
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_mistral_lora_param_count() {
        // Mistral: nearly identical to Llama — separate q/k/v/o, no bias
        let rank = 8usize;
        let hidden_size = 4096usize;
        let num_heads = 32usize;
        let num_kv_heads = 8usize;
        let head_dim = hidden_size / num_heads; // 128
        let layers = 2usize;

        // q_proj: in=hidden_size, out=num_heads*head_dim → R*4096 + 4096*R
        // k_proj: in=hidden_size, out=num_kv_heads*head_dim → R*4096 + 1024*R
        // v_proj: same as k_proj
        // o_proj: in=num_heads*head_dim, out=hidden_size → R*4096 + 4096*R
        let q_params = rank * hidden_size + (num_heads * head_dim) * rank;
        let k_params = rank * hidden_size + (num_kv_heads * head_dim) * rank;
        let v_params = k_params;
        let o_params = rank * (num_heads * head_dim) + hidden_size * rank;
        let per_layer = q_params + k_params + v_params + o_params;
        let total = per_layer * layers;
        assert!(total > 0, "should have nonzero LoRA params: {total}");
        // q: 32768+32768=65536, k: 32768+8192=40960, v: 40960, o: 32768+32768=65536
        assert_eq!(per_layer, 65536 + 40960 + 40960 + 65536);
        assert_eq!(total, per_layer * 2);
    }
}
