use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::FinetuneError;
use crate::lora::{LoraConfig, LoraLinear};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub epoch: usize,
    pub step: usize,
    pub loss: f64,
    pub lora_config: LoraConfig,
}

pub fn save_checkpoint(
    lora: &LoraLinear,
    meta: &CheckpointMeta,
    dir: impl AsRef<Path>,
) -> Result<(), FinetuneError> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)
        .map_err(|e| FinetuneError::Checkpoint(format!("failed to create dir: {e}")))?;

    let tensors_path = dir.join("lora_weights.safetensors");
    lora.save(&tensors_path)
        .map_err(|e| FinetuneError::Checkpoint(format!("failed to save tensors: {e}")))?;

    let meta_path = dir.join("meta.json");
    let meta_json = serde_json::to_string_pretty(meta)
        .map_err(|e| FinetuneError::Checkpoint(format!("failed to serialize meta: {e}")))?;
    std::fs::write(&meta_path, meta_json)
        .map_err(|e| FinetuneError::Checkpoint(format!("failed to write meta: {e}")))?;

    Ok(())
}

pub fn load_checkpoint(
    lora: &mut LoraLinear,
    dir: impl AsRef<Path>,
    device: &Device,
) -> Result<CheckpointMeta, FinetuneError> {
    let dir = dir.as_ref();

    let tensors_path = dir.join("lora_weights.safetensors");
    lora.load(&tensors_path, device)
        .map_err(|e| FinetuneError::Checkpoint(format!("failed to load tensors: {e}")))?;

    let meta_path = dir.join("meta.json");
    let meta_json = std::fs::read_to_string(&meta_path)
        .map_err(|e| FinetuneError::Checkpoint(format!("failed to read meta: {e}")))?;
    let meta: CheckpointMeta = serde_json::from_str(&meta_json)
        .map_err(|e| FinetuneError::Checkpoint(format!("failed to parse meta: {e}")))?;

    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::LoraConfig;
    use candle_core::{DType, Tensor};
    use candle_nn::Linear;

    fn make_test_lora(device: &Device) -> LoraLinear {
        let weight = Tensor::zeros(&[4, 4], DType::F32, device).unwrap();
        let frozen = Linear::new(weight, None);
        let config = LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
        };
        LoraLinear::new(frozen, 4, 4, &config, device).unwrap()
    }

    #[test]
    fn test_checkpoint_round_trip() {
        let device = Device::Cpu;
        let lora = make_test_lora(&device);

        let meta = CheckpointMeta {
            epoch: 3,
            step: 42,
            loss: 0.25,
            lora_config: LoraConfig {
                rank: 2,
                alpha: 4.0,
                dropout: 0.0,
            },
        };

        let dir = tempfile::tempdir().unwrap();
        save_checkpoint(&lora, &meta, dir.path()).unwrap();

        // Get original tensors for comparison
        let orig_a = lora.lora_a().clone();
        let orig_b = lora.lora_b().clone();

        // Create a new LoRA and load checkpoint into it
        let mut lora2 = make_test_lora(&device);
        let loaded_meta = load_checkpoint(&mut lora2, dir.path(), &device).unwrap();

        assert_eq!(loaded_meta.epoch, 3);
        assert_eq!(loaded_meta.step, 42);
        assert!((loaded_meta.loss - 0.25).abs() < 1e-10);
        assert_eq!(loaded_meta.lora_config.rank, 2);

        // Verify tensor equality
        let loaded_a = lora2.lora_a();
        let loaded_b = lora2.lora_b();

        let diff_a = (orig_a - loaded_a)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff_b = (orig_b - loaded_b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(diff_a < 1e-6);
        assert!(diff_b < 1e-6);
    }
}
