use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

use crate::error::{RagError, Result};

/// Map a HuggingFace Phi-3/4 tensor name to the GGML convention used by quantized_phi3.
fn map_tensor_name(hf_name: &str) -> Option<String> {
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

/// Determine the quantization dtype for a given tensor name under the specified scheme.
fn quant_dtype_for_tensor(ggml_name: &str, scheme: &str) -> GgmlDType {
    let is_norm = ggml_name.contains("norm");
    let is_embed = ggml_name == "token_embd.weight" || ggml_name == "output.weight";

    // Norms and embeddings are always kept as F32
    if is_norm || is_embed {
        return GgmlDType::F32;
    }

    match scheme {
        "q4-k-m" => {
            // Q6K for attn_output and ffn_down (sensitive layers), Q4K for the rest
            if ggml_name.contains("attn_output") || ggml_name.contains("ffn_down") {
                GgmlDType::Q6K
            } else {
                GgmlDType::Q4K
            }
        }
        "q8-0" => GgmlDType::Q8_0,
        "q4-0" => GgmlDType::Q4_0,
        _ => GgmlDType::Q4K, // default fallback
    }
}

/// Parse the Phi-3/4 config.json to extract model parameters needed for GGUF metadata.
fn read_phi_config(model_dir: &Path) -> Result<PhiParams> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

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

    let head_dim = hidden_size / num_attention_heads;

    Ok(PhiParams {
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
    })
}

struct PhiParams {
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
}

/// Quantize SafeTensors model files to a single GGUF file.
///
/// Loads F32 tensors from SafeTensors (one at a time to minimize memory),
/// quantizes them according to the specified scheme, and writes a GGUF file
/// with the required Phi-3 metadata.
pub fn quantize_safetensors_to_gguf(
    model_dir: &Path,
    output_path: &Path,
    quantization: &str,
) -> Result<()> {
    let device = Device::Cpu;

    // Read model config for GGUF metadata
    let params = read_phi_config(model_dir)?;

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

    // Quantize tensors one at a time to minimize peak memory
    let mut quantized_tensors: Vec<(String, QTensor)> = Vec::new();

    for (hf_name, ggml_name, st_file) in &tensor_sources {
        let data = std::fs::read(st_file)?;
        let st = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| RagError::Other(format!("failed to parse safetensors: {e}")))?;

        let view = st
            .tensor(hf_name)
            .map_err(|e| RagError::Other(format!("tensor {hf_name} not found: {e}")))?;

        let shape: Vec<usize> = view.shape().to_vec();
        let tensor = tensor_from_safetensors_view(&view, &shape, &device)?;

        let target_dtype = quant_dtype_for_tensor(ggml_name, quantization);

        let qtensor = if target_dtype == GgmlDType::F32 {
            QTensor::quantize(&tensor, GgmlDType::F32)?
        } else {
            QTensor::quantize(&tensor, target_dtype)?
        };

        quantized_tensors.push((ggml_name.clone(), qtensor));
    }

    // Build GGUF metadata
    use gguf_file::Value;
    let arch = Value::String("phi3".to_string());
    let block_count = Value::U32(params.num_hidden_layers);
    let embedding_length = Value::U32(params.hidden_size);
    let head_count = Value::U32(params.num_attention_heads);
    let head_count_kv = Value::U32(params.num_kv_heads);
    let context_length = Value::U32(params.max_position_embeddings);
    let feed_forward_length = Value::U32(params.intermediate_size);
    let rms_eps = Value::F32(params.rms_norm_eps);
    let rope_dim = Value::U32(params.head_dim);
    let rope_freq = Value::F32(params.rope_theta);
    let vocab_size = Value::U32(params.vocab_size);

    let metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch),
        ("phi3.block_count", &block_count),
        ("phi3.embedding_length", &embedding_length),
        ("phi3.attention.head_count", &head_count),
        ("phi3.attention.head_count_kv", &head_count_kv),
        ("phi3.context_length", &context_length),
        ("phi3.feed_forward_length", &feed_forward_length),
        ("phi3.attention.layer_norm_rms_epsilon", &rms_eps),
        ("phi3.rope.dimension_count", &rope_dim),
        ("phi3.rope.freq_base", &rope_freq),
        ("phi3.vocab_size", &vocab_size),
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
    fn test_tensor_name_mapping() {
        assert_eq!(
            map_tensor_name("model.embed_tokens.weight"),
            Some("token_embd.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.norm.weight"),
            Some("output_norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("lm_head.weight"),
            Some("output.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.layers.0.self_attn.qkv_proj.weight"),
            Some("blk.0.attn_qkv.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.layers.5.self_attn.o_proj.weight"),
            Some("blk.5.attn_output.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.layers.3.mlp.gate_up_proj.weight"),
            Some("blk.3.ffn_up.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.layers.3.mlp.down_proj.weight"),
            Some("blk.3.ffn_down.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.layers.1.input_layernorm.weight"),
            Some("blk.1.attn_norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("model.layers.1.post_attention_layernorm.weight"),
            Some("blk.1.ffn_norm.weight".to_string())
        );
        assert_eq!(map_tensor_name("unknown.tensor"), None);
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
