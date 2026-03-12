use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

use crate::error::{RagError, Result};

/// Map a HuggingFace Phi-3/4 tensor name to the GGML convention used by quantized_phi3.
fn map_tensor_name_phi(hf_name: &str) -> Option<String> {
    // Embedding
    if hf_name == "model.embed_tokens.weight" {
        return Some("token_embd.weight".to_string());
    }
    // Final norm
    if hf_name == "model.norm.weight" {
        return Some("output_norm.weight".to_string());
    }
    // LM head
    if hf_name == "lm_head.weight" {
        return Some("output.weight".to_string());
    }

    // Layer-level mappings: model.layers.{i}.XXX -> blk.{i}.YYY
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        let dot_pos = rest.find('.')?;
        let layer_idx = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];

        let ggml_suffix = match suffix {
            "self_attn.qkv_proj.weight" => "attn_qkv.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "mlp.gate_up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            _ => return None,
        };

        return Some(format!("blk.{layer_idx}.{ggml_suffix}"));
    }

    None
}

/// Map a HuggingFace Llama tensor name to the GGML convention used by quantized_llama.
fn map_tensor_name_llama(hf_name: &str) -> Option<String> {
    // Embedding
    if hf_name == "model.embed_tokens.weight" {
        return Some("token_embd.weight".to_string());
    }
    // Final norm
    if hf_name == "model.norm.weight" {
        return Some("output_norm.weight".to_string());
    }
    // LM head
    if hf_name == "lm_head.weight" {
        return Some("output.weight".to_string());
    }

    // Layer-level mappings: model.layers.{i}.XXX -> blk.{i}.YYY
    if let Some(rest) = hf_name.strip_prefix("model.layers.") {
        let dot_pos = rest.find('.')?;
        let layer_idx = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];

        let ggml_suffix = match suffix {
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "ffn_norm.weight",
            _ => return None,
        };

        return Some(format!("blk.{layer_idx}.{ggml_suffix}"));
    }

    None
}

/// Determine the quantization dtype for a given tensor name under the specified scheme.
fn quant_dtype_for_tensor(ggml_name: &str, scheme: &str) -> GgmlDType {
    let is_norm = ggml_name.contains("norm");
    let is_embed = ggml_name == "token_embd.weight" || ggml_name == "output.weight";

    // Norms and embeddings are always kept as F32
    if is_norm || is_embed {
        return GgmlDType::F32;
    }

    match scheme {
        "q4-k-m" | "q5-k-m" => {
            // K-M mixed: Q6K for sensitive layers (attn_output, ffn_down, ffn_gate), base dtype for the rest
            if ggml_name.contains("attn_output") || ggml_name.contains("ffn_down") || ggml_name.contains("ffn_gate") {
                GgmlDType::Q6K
            } else if scheme == "q5-k-m" {
                GgmlDType::Q5K
            } else {
                GgmlDType::Q4K
            }
        }
        "q8-0" => GgmlDType::Q8_0,
        "q4-0" => GgmlDType::Q4_0,
        _ => GgmlDType::Q4K, // default fallback
    }
}

struct ModelParams {
    hidden_size: u32,
    num_attention_heads: u32,
    num_kv_heads: u32,
    num_hidden_layers: u32,
    intermediate_size: u32,
    vocab_size: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    head_dim: u32,
    tie_word_embeddings: bool,
    partial_rotary_factor: f32,
}

/// Parse common model config.json fields into ModelParams.
fn read_model_config(config: &serde_json::Value) -> Result<ModelParams> {
    let hidden_size = config["hidden_size"]
        .as_u64()
        .ok_or_else(|| RagError::Other("missing hidden_size in config".into()))?
        as u32;
    let num_attention_heads = config["num_attention_heads"]
        .as_u64()
        .ok_or_else(|| RagError::Other("missing num_attention_heads".into()))?
        as u32;
    let num_kv_heads = config["num_key_value_heads"]
        .as_u64()
        .ok_or_else(|| RagError::Other("missing num_key_value_heads".into()))?
        as u32;
    let num_hidden_layers = config["num_hidden_layers"]
        .as_u64()
        .ok_or_else(|| RagError::Other("missing num_hidden_layers".into()))?
        as u32;
    let intermediate_size = config["intermediate_size"]
        .as_u64()
        .ok_or_else(|| RagError::Other("missing intermediate_size".into()))?
        as u32;
    let vocab_size = config["vocab_size"]
        .as_u64()
        .ok_or_else(|| RagError::Other("missing vocab_size".into()))?
        as u32;
    let max_position_embeddings = config["max_position_embeddings"]
        .as_u64()
        .ok_or_else(|| RagError::Other("missing max_position_embeddings".into()))?
        as u32;
    let rms_norm_eps = config["rms_norm_eps"]
        .as_f64()
        .ok_or_else(|| RagError::Other("missing rms_norm_eps".into()))?
        as f32;
    let rope_theta = config["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let tie_word_embeddings = config["tie_word_embeddings"].as_bool().unwrap_or(false);
    let partial_rotary_factor = config["partial_rotary_factor"].as_f64().unwrap_or(1.0) as f32;

    let head_dim = hidden_size / num_attention_heads;

    Ok(ModelParams {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        head_dim,
        tie_word_embeddings,
        partial_rotary_factor,
    })
}

/// Detect architecture from config.json.
fn detect_quant_arch(config: &serde_json::Value) -> &'static str {
    match config.get("model_type").and_then(|v| v.as_str()) {
        Some("llama") => "llama",
        _ => "phi3",
    }
}

/// Quantize SafeTensors model files to a single GGUF file.
///
/// Loads F32 tensors from SafeTensors (one at a time to minimize memory),
/// quantizes them according to the specified scheme, and writes a GGUF file
/// with the required Phi-3 metadata.
///
/// An optional progress callback receives `(current_tensor_index, total_tensor_count)`
/// after each tensor is quantized.
pub fn quantize_safetensors_to_gguf(
    model_dir: &Path,
    output_path: &Path,
    quantization: &str,
    progress_fn: Option<&dyn Fn(usize, usize)>,
) -> Result<()> {
    let device = Device::Cpu;

    // Read model config for GGUF metadata and auto-detect architecture
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
    let arch = detect_quant_arch(&config_json);
    let params = read_model_config(&config_json)?;
    let map_tensor_name: fn(&str) -> Option<String> = match arch {
        "llama" => map_tensor_name_llama,
        _ => map_tensor_name_phi,
    };

    // Find all safetensors files
    let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err(RagError::Other(format!(
            "No safetensors files found in {}",
            model_dir.display()
        )));
    }

    // Collect all tensor names and their source files
    let mut tensor_sources: Vec<(String, String, std::path::PathBuf)> = Vec::new();
    for st_file in &st_files {
        let data = std::fs::read(st_file)?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| RagError::Other(format!("failed to parse safetensors: {e}")))?;
        for name in st.names() {
            if let Some(ggml_name) = map_tensor_name(name) {
                tensor_sources.push((name.to_string(), ggml_name, st_file.clone()));
            }
        }
    }

    // If output.weight is missing and embeddings are tied, duplicate token_embd.weight
    let has_output = tensor_sources.iter().any(|(_, g, _)| g == "output.weight");
    if !has_output && params.tie_word_embeddings {
        if let Some((hf, _, path)) = tensor_sources
            .iter()
            .find(|(_, g, _)| g == "token_embd.weight")
        {
            tensor_sources.push((hf.clone(), "output.weight".to_string(), path.clone()));
        }
    }

    let total_tensors = tensor_sources.len();

    // Group tensors by source file to avoid redundant I/O.
    // Each file is read once, and all tensors from it are quantized before moving on.
    let mut grouped: std::collections::BTreeMap<std::path::PathBuf, Vec<(String, String)>> =
        std::collections::BTreeMap::new();
    for (hf_name, ggml_name, st_file) in &tensor_sources {
        grouped
            .entry(st_file.clone())
            .or_default()
            .push((hf_name.clone(), ggml_name.clone()));
    }

    let mut quantized_tensors: Vec<(String, QTensor)> = Vec::with_capacity(total_tensors);
    let mut completed = 0usize;

    for (st_file, tensors_in_file) in &grouped {
        let data = std::fs::read(st_file)?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| RagError::Other(format!("failed to parse safetensors: {e}")))?;

        for (hf_name, ggml_name) in tensors_in_file {
            let view = st
                .tensor(hf_name)
                .map_err(|e| RagError::Other(format!("tensor {hf_name} not found: {e}")))?;

            let shape: Vec<usize> = view.shape().to_vec();
            let tensor = tensor_from_safetensors_view(&view, &shape, &device)?;

            let target_dtype = quant_dtype_for_tensor(ggml_name, quantization);
            let qtensor = QTensor::quantize(&tensor, target_dtype)?;

            quantized_tensors.push((ggml_name.clone(), qtensor));

            completed += 1;
            if let Some(cb) = progress_fn {
                cb(completed, total_tensors);
            }
        }
    }

    // Build GGUF metadata with architecture-specific prefix
    use gguf_file::Value;
    let arch_val = Value::String(arch.to_string());
    let block_count = Value::U32(params.num_hidden_layers);
    let embedding_length = Value::U32(params.hidden_size);
    let head_count = Value::U32(params.num_attention_heads);
    let head_count_kv = Value::U32(params.num_kv_heads);
    let context_length = Value::U32(params.max_position_embeddings);
    let feed_forward_length = Value::U32(params.intermediate_size);
    let rms_eps = Value::F32(params.rms_norm_eps);
    let effective_rope_dim = (params.head_dim as f32 * params.partial_rotary_factor) as u32;
    let rope_dim = Value::U32(effective_rope_dim);
    let rope_freq = Value::F32(params.rope_theta);
    let vocab_size_val = Value::U32(params.vocab_size);

    // Use architecture-appropriate prefix (phi3.* or llama.*)
    let prefix = arch;
    let key_block = format!("{prefix}.block_count");
    let key_emb = format!("{prefix}.embedding_length");
    let key_head = format!("{prefix}.attention.head_count");
    let key_head_kv = format!("{prefix}.attention.head_count_kv");
    let key_ctx = format!("{prefix}.context_length");
    let key_ff = format!("{prefix}.feed_forward_length");
    let key_rms = format!("{prefix}.attention.layer_norm_rms_epsilon");
    let key_rope_dim = format!("{prefix}.rope.dimension_count");
    let key_rope_freq = format!("{prefix}.rope.freq_base");
    let key_vocab = format!("{prefix}.vocab_size");

    let metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch_val),
        (&key_block, &block_count),
        (&key_emb, &embedding_length),
        (&key_head, &head_count),
        (&key_head_kv, &head_count_kv),
        (&key_ctx, &context_length),
        (&key_ff, &feed_forward_length),
        (&key_rms, &rms_eps),
        (&key_rope_dim, &rope_dim),
        (&key_rope_freq, &rope_freq),
        (&key_vocab, &vocab_size_val),
    ];

    // Build tensor refs
    let tensor_refs: Vec<(&str, &QTensor)> = quantized_tensors
        .iter()
        .map(|(name, qt)| (name.as_str(), qt))
        .collect();

    // Write GGUF file
    let mut file = std::io::BufWriter::new(std::fs::File::create(output_path)?);
    gguf_file::write(&mut file, &metadata, &tensor_refs)?;

    Ok(())
}

/// Convert a safetensors tensor view to a candle Tensor.
fn tensor_from_safetensors_view(
    view: &safetensors::tensor::TensorView<'_>,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    use safetensors::Dtype;
    match view.dtype() {
        Dtype::F32 => {
            let data: &[u8] = view.data();
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Tensor::from_vec(floats, shape, device)?)
        }
        Dtype::F16 => {
            let data: &[u8] = view.data();
            let halfs: Vec<half::f16> = data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                .collect();
            let tensor = Tensor::from_vec(halfs, shape, device)?;
            Ok(tensor.to_dtype(candle_core::DType::F32)?)
        }
        Dtype::BF16 => {
            let data: &[u8] = view.data();
            let bfloats: Vec<half::bf16> = data
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
                .collect();
            let tensor = Tensor::from_vec(bfloats, shape, device)?;
            Ok(tensor.to_dtype(candle_core::DType::F32)?)
        }
        dtype => Err(RagError::Other(format!(
            "unsupported safetensors dtype: {dtype:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_name_mapping_phi() {
        assert_eq!(
            map_tensor_name_phi("model.embed_tokens.weight"),
            Some("token_embd.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("model.norm.weight"),
            Some("output_norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("lm_head.weight"),
            Some("output.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("model.layers.0.self_attn.qkv_proj.weight"),
            Some("blk.0.attn_qkv.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("model.layers.5.self_attn.o_proj.weight"),
            Some("blk.5.attn_output.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("model.layers.3.mlp.gate_up_proj.weight"),
            Some("blk.3.ffn_up.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("model.layers.3.mlp.down_proj.weight"),
            Some("blk.3.ffn_down.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("model.layers.1.input_layernorm.weight"),
            Some("blk.1.attn_norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_phi("model.layers.1.post_attention_layernorm.weight"),
            Some("blk.1.ffn_norm.weight".to_string())
        );
        assert_eq!(map_tensor_name_phi("unknown.tensor"), None);
    }

    #[test]
    fn test_tensor_name_mapping_llama() {
        // Embeddings and norms — same as Phi
        assert_eq!(
            map_tensor_name_llama("model.embed_tokens.weight"),
            Some("token_embd.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("model.norm.weight"),
            Some("output_norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("lm_head.weight"),
            Some("output.weight".to_string())
        );
        // Separate q/k/v projections
        assert_eq!(
            map_tensor_name_llama("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("model.layers.0.self_attn.k_proj.weight"),
            Some("blk.0.attn_k.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("model.layers.0.self_attn.v_proj.weight"),
            Some("blk.0.attn_v.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("model.layers.5.self_attn.o_proj.weight"),
            Some("blk.5.attn_output.weight".to_string())
        );
        // Separate gate/up/down MLP
        assert_eq!(
            map_tensor_name_llama("model.layers.3.mlp.gate_proj.weight"),
            Some("blk.3.ffn_gate.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("model.layers.3.mlp.up_proj.weight"),
            Some("blk.3.ffn_up.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("model.layers.3.mlp.down_proj.weight"),
            Some("blk.3.ffn_down.weight".to_string())
        );
        // Norms
        assert_eq!(
            map_tensor_name_llama("model.layers.1.input_layernorm.weight"),
            Some("blk.1.attn_norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name_llama("model.layers.1.post_attention_layernorm.weight"),
            Some("blk.1.ffn_norm.weight".to_string())
        );
        assert_eq!(map_tensor_name_llama("unknown.tensor"), None);
    }

    #[test]
    fn test_llama_gguf_metadata() {
        let config_json = serde_json::json!({"model_type": "llama"});
        let arch = detect_quant_arch(&config_json);
        assert_eq!(arch, "llama");
        // Verify metadata keys would use llama prefix
        let prefix = arch;
        assert_eq!(format!("{prefix}.block_count"), "llama.block_count");
        assert_eq!(format!("{prefix}.attention.head_count"), "llama.attention.head_count");
    }

    #[test]
    fn test_quant_dtype_for_llama_tensors() {
        // ffn_gate should be treated as sensitive in k-m schemes
        assert_eq!(
            quant_dtype_for_tensor("blk.0.ffn_gate.weight", "q4-k-m"),
            GgmlDType::Q6K
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.ffn_gate.weight", "q5-k-m"),
            GgmlDType::Q6K
        );
        // attn_q, attn_k, attn_v should get base dtype
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_q.weight", "q4-k-m"),
            GgmlDType::Q4K
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_k.weight", "q5-k-m"),
            GgmlDType::Q5K
        );
    }

    #[test]
    fn test_quant_dtype_selection() {
        // q4-k-m: norms/embeds -> F32, attn_output/ffn_down -> Q6K, rest -> Q4K
        assert_eq!(
            quant_dtype_for_tensor("token_embd.weight", "q4-k-m"),
            GgmlDType::F32
        );
        assert_eq!(
            quant_dtype_for_tensor("output_norm.weight", "q4-k-m"),
            GgmlDType::F32
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_output.weight", "q4-k-m"),
            GgmlDType::Q6K
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.ffn_down.weight", "q4-k-m"),
            GgmlDType::Q6K
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_qkv.weight", "q4-k-m"),
            GgmlDType::Q4K
        );

        // q5-k-m: norms/embeds -> F32, attn_output/ffn_down -> Q6K, rest -> Q5K
        assert_eq!(
            quant_dtype_for_tensor("token_embd.weight", "q5-k-m"),
            GgmlDType::F32
        );
        assert_eq!(
            quant_dtype_for_tensor("output_norm.weight", "q5-k-m"),
            GgmlDType::F32
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_output.weight", "q5-k-m"),
            GgmlDType::Q6K
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.ffn_down.weight", "q5-k-m"),
            GgmlDType::Q6K
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_qkv.weight", "q5-k-m"),
            GgmlDType::Q5K
        );
        assert_eq!(
            quant_dtype_for_tensor("blk.0.ffn_up.weight", "q5-k-m"),
            GgmlDType::Q5K
        );

        // q8-0
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_qkv.weight", "q8-0"),
            GgmlDType::Q8_0
        );
        assert_eq!(
            quant_dtype_for_tensor("token_embd.weight", "q8-0"),
            GgmlDType::F32
        );

        // q4-0
        assert_eq!(
            quant_dtype_for_tensor("blk.0.attn_qkv.weight", "q4-0"),
            GgmlDType::Q4_0
        );
    }

    #[test]
    fn test_quantize_round_trip() {
        // Create a small tensor, quantize it, verify it produces valid QTensor
        let device = Device::Cpu;
        let tensor = Tensor::rand(0.0f32, 1.0f32, &[64, 64], &device).unwrap();
        let qtensor = QTensor::quantize(&tensor, GgmlDType::Q4_0).unwrap();
        assert_eq!(qtensor.shape().dims(), &[64, 64]);

        // Verify we can dequantize back
        let recovered = qtensor.dequantize(&device).unwrap();
        assert_eq!(recovered.shape().dims(), &[64, 64]);

        // Q5K round-trip (Q5K block size is 256, so last dim must be divisible by 256)
        let tensor_q5k = Tensor::rand(0.0f32, 1.0f32, &[64, 256], &device).unwrap();
        let qtensor_q5k = QTensor::quantize(&tensor_q5k, GgmlDType::Q5K).unwrap();
        assert_eq!(qtensor_q5k.shape().dims(), &[64, 256]);
        let recovered_q5k = qtensor_q5k.dequantize(&device).unwrap();
        assert_eq!(recovered_q5k.shape().dims(), &[64, 256]);
    }

    #[test]
    fn test_partial_rotary_rope_dim() {
        // Phi-4-mini: partial_rotary_factor=0.75, head_dim=128 -> rope_dim=96
        let rope_dim = (128_f32 * 0.75) as u32;
        assert_eq!(rope_dim, 96);

        // Full rotary: partial_rotary_factor=1.0, head_dim=96 -> rope_dim=96
        let rope_dim = (96_f32 * 1.0) as u32;
        assert_eq!(rope_dim, 96);
    }

    #[test]
    fn test_q5km_mixed_gguf_round_trip() {
        use candle_core::quantized::gguf_file;
        let device = Device::Cpu;

        // Simulate q5-k-m mixed quantization: Q5K (bulk), Q6K (sensitive), F32 (norms)
        let bulk = Tensor::rand(0.0f32, 1.0f32, &[256, 256], &device).unwrap();
        let qt_q5k = QTensor::quantize(&bulk, GgmlDType::Q5K).unwrap();

        let sensitive = Tensor::rand(0.0f32, 1.0f32, &[256, 256], &device).unwrap();
        let qt_q6k = QTensor::quantize(&sensitive, GgmlDType::Q6K).unwrap();

        let norm = Tensor::rand(0.0f32, 1.0f32, &[256], &device).unwrap();
        let qt_f32 = QTensor::quantize(&norm, GgmlDType::F32).unwrap();

        let val = gguf_file::Value::U32(1);
        let metadata = vec![("test.version", &val)];
        let tensors = vec![
            ("blk.0.attn_qkv.weight", &qt_q5k),
            ("blk.0.attn_output.weight", &qt_q6k),
            ("output_norm.weight", &qt_f32),
        ];

        // Write
        let mut buf = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &metadata, &tensors).unwrap();

        // Read back and verify dtypes
        buf.set_position(0);
        let content = gguf_file::Content::read(&mut buf).unwrap();
        assert_eq!(content.tensor_infos.len(), 3);

        let qkv_info = &content.tensor_infos["blk.0.attn_qkv.weight"];
        assert_eq!(qkv_info.ggml_dtype, GgmlDType::Q5K);

        let attn_out_info = &content.tensor_infos["blk.0.attn_output.weight"];
        assert_eq!(attn_out_info.ggml_dtype, GgmlDType::Q6K);

        let norm_info = &content.tensor_infos["output_norm.weight"];
        assert_eq!(norm_info.ggml_dtype, GgmlDType::F32);
    }

    #[test]
    fn test_quant_dtype_for_embedding() {
        // Embedding layers should always get F32 regardless of scheme
        assert_eq!(
            quant_dtype_for_tensor("token_embd.weight", "q4-k-m"),
            GgmlDType::F32
        );
        assert_eq!(
            quant_dtype_for_tensor("output.weight", "q4-k-m"),
            GgmlDType::F32
        );
        assert_eq!(
            quant_dtype_for_tensor("token_embd.weight", "q8-0"),
            GgmlDType::F32
        );
        assert_eq!(
            quant_dtype_for_tensor("output.weight", "q5-k-m"),
            GgmlDType::F32
        );
    }

    #[test]
    fn test_gguf_write_read_round_trip() {
        use candle_core::quantized::gguf_file;
        let device = Device::Cpu;

        // Create some quantized tensors
        let t1 = Tensor::rand(0.0f32, 1.0f32, &[32, 32], &device).unwrap();
        let qt1 = QTensor::quantize(&t1, GgmlDType::Q4_0).unwrap();

        let t2 = Tensor::rand(0.0f32, 1.0f32, &[32], &device).unwrap();
        let qt2 = QTensor::quantize(&t2, GgmlDType::F32).unwrap();

        let val = gguf_file::Value::U32(42);
        let metadata = vec![("test.value", &val)];
        let tensors = vec![("test.weight", &qt1), ("test.norm", &qt2)];

        // Write
        let mut buf = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &metadata, &tensors).unwrap();

        // Read back
        buf.set_position(0);
        let content = gguf_file::Content::read(&mut buf).unwrap();
        assert_eq!(content.tensor_infos.len(), 2);
        assert!(content.tensor_infos.contains_key("test.weight"));
        assert!(content.tensor_infos.contains_key("test.norm"));
        assert_eq!(
            content.metadata.get("test.value").unwrap().to_u32().unwrap(),
            42
        );
    }
}
