use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::Linear;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 4.0,
            dropout: 0.0,
        }
    }
}

pub struct LoraLinear {
    frozen: Linear,
    lora_a: Tensor,
    lora_b: Tensor,
    scale: f64,
}

impl LoraLinear {
    pub fn new(frozen: Linear, in_features: usize, out_features: usize, config: &LoraConfig, device: &Device) -> Result<Self> {
        // Kaiming uniform init for A
        let bound = (1.0 / in_features as f64).sqrt();
        let lora_a = Tensor::rand(-bound as f32, bound as f32, &[config.rank, in_features], device)?
            .to_dtype(DType::F32)?;

        // Zero init for B
        let lora_b = Tensor::zeros(&[out_features, config.rank], DType::F32, device)?;

        let scale = config.alpha / config.rank as f64;

        Ok(Self {
            frozen,
            lora_a,
            lora_b,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let frozen_out = self.frozen.forward(x)?;

        // x @ A^T @ B^T * scale
        // Flatten to 2D for matmul (candle doesn't broadcast 3D @ 2D), then restore shape
        let dims = x.dims();
        let lora_out = if dims.len() == 3 {
            let (b, s, h) = (dims[0], dims[1], dims[2]);
            let out_2d = x.reshape((b * s, h))?
                .matmul(&self.lora_a.t()?)?
                .matmul(&self.lora_b.t()?)?;
            out_2d.reshape((b, s, out_2d.dims()[1]))?
        } else {
            x.matmul(&self.lora_a.t()?)?.matmul(&self.lora_b.t()?)?
        };
        let lora_out = (lora_out * self.scale)?;

        let out = (frozen_out + lora_out)?;
        Ok(out)
    }

    pub fn trainable_tensors(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut tensors = HashMap::new();
        tensors.insert("lora_a".to_string(), self.lora_a.clone());
        tensors.insert("lora_b".to_string(), self.lora_b.clone());
        candle_core::safetensors::save(&tensors, path.as_ref())
            .context("failed to save LoRA adapter weights")?;
        Ok(())
    }

    /// Forward pass using only the frozen linear layer (no LoRA contribution).
    /// Useful for DPO reference model computation.
    pub fn forward_frozen_only(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.frozen.forward(x)?;
        Ok(out)
    }

    /// Get a reference to the lora_a tensor.
    pub fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    /// Get a reference to the lora_b tensor.
    pub fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    /// Set the lora_a tensor.
    pub fn set_lora_a(&mut self, t: Tensor) {
        self.lora_a = t;
    }

    /// Set the lora_b tensor.
    pub fn set_lora_b(&mut self, t: Tensor) {
        self.lora_b = t;
    }

    /// Get the LoRA scaling factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn load<P: AsRef<Path>>(&mut self, path: P, device: &Device) -> Result<()> {
        let tensors = candle_core::safetensors::load(path.as_ref(), device)
            .context("failed to load LoRA adapter weights")?;
        self.lora_a = tensors
            .get("lora_a")
            .context("missing lora_a in checkpoint")?
            .clone();
        self.lora_b = tensors
            .get("lora_b")
            .context("missing lora_b in checkpoint")?
            .clone();
        Ok(())
    }
}

/// Multi-layer LoRA model for managing LoRA across multiple named layers.
pub struct LoraModel {
    pub layers: HashMap<String, LoraLinear>,
    pub config: LoraConfig,
}

impl LoraModel {
    pub fn new(config: LoraConfig) -> Self {
        Self {
            layers: HashMap::new(),
            config,
        }
    }

    pub fn add_layer(
        &mut self,
        name: String,
        frozen: candle_nn::Linear,
        in_features: usize,
        out_features: usize,
        device: &Device,
    ) -> Result<()> {
        let lora = LoraLinear::new(frozen, in_features, out_features, &self.config, device)?;
        self.layers.insert(name, lora);
        Ok(())
    }

    /// Save all LoRA layers to a single safetensors file with namespaced keys.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut tensors = HashMap::new();
        for (name, lora) in &self.layers {
            tensors.insert(format!("{name}.lora_a"), lora.lora_a().clone());
            tensors.insert(format!("{name}.lora_b"), lora.lora_b().clone());
        }
        candle_core::safetensors::save(&tensors, path.as_ref())
            .context("failed to save multi-layer LoRA adapter")?;
        Ok(())
    }

    /// Load all LoRA layers from a namespaced safetensors file.
    pub fn load<P: AsRef<Path>>(&mut self, path: P, device: &Device) -> Result<()> {
        let tensors = candle_core::safetensors::load(path.as_ref(), device)
            .context("failed to load multi-layer LoRA adapter")?;
        for (name, lora) in &mut self.layers {
            let a_key = format!("{name}.lora_a");
            let b_key = format!("{name}.lora_b");
            if let (Some(a), Some(b)) = (tensors.get(&a_key), tensors.get(&b_key)) {
                lora.set_lora_a(a.clone());
                lora.set_lora_b(b.clone());
            }
        }
        Ok(())
    }

    /// Collect all trainable tensors across all layers.
    pub fn trainable_tensors(&self) -> Vec<&Tensor> {
        let mut result = Vec::new();
        for lora in self.layers.values() {
            result.extend(lora.trainable_tensors());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{make_test_lora, tensor_abs_diff};
    use candle_core::Tensor;

    #[test]
    fn test_lora_linear_init_shapes() {
        let lora = make_test_lora(8, 16, 4, 16.0);
        assert_eq!(lora.lora_a().dims(), &[4, 8]); // [rank, in_features]
        assert_eq!(lora.lora_b().dims(), &[16, 4]); // [out_features, rank]
    }

    #[test]
    fn test_lora_linear_forward_dtype_preserved() {
        let lora = make_test_lora(8, 16, 4, 16.0);
        let x = Tensor::rand(0.0f32, 1.0f32, &[3, 8], &Device::Cpu).unwrap();
        assert_eq!(x.dtype(), DType::F32);
        let out = lora.forward(&x).unwrap();
        assert_eq!(out.dtype(), DType::F32, "F32 input must produce F32 output");
    }

    #[test]
    fn test_lora_b_initialized_to_zero() {
        let lora = make_test_lora(8, 16, 4, 16.0);
        let sum = lora
            .lora_b()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(sum, 0.0, "lora_b must be initialized to all zeros");
    }

    #[test]
    fn test_lora_forward_equals_frozen_at_init() {
        let lora = make_test_lora(8, 16, 4, 16.0);
        let x = Tensor::rand(0.0f32, 1.0f32, &[3, 8], &Device::Cpu).unwrap();
        let full_out = lora.forward(&x).unwrap();
        let frozen_out = lora.forward_frozen_only(&x).unwrap();
        let diff = tensor_abs_diff(&full_out, &frozen_out);
        assert!(
            diff < 1e-6,
            "with B=0, forward should equal forward_frozen_only, diff={diff}"
        );
    }

    #[test]
    fn test_lora_scale_computation() {
        let device = Device::Cpu;
        let weight = Tensor::zeros(&[16, 8], DType::F32, &device).unwrap();
        let frozen = Linear::new(weight, None);
        let config = LoraConfig {
            rank: 4,
            alpha: 32.0,
            dropout: 0.0,
        };
        let lora = LoraLinear::new(frozen, 8, 16, &config, &device).unwrap();
        assert!((lora.scale() - 8.0).abs() < 1e-10, "scale should be alpha/rank = 32/4 = 8");
    }

    #[test]
    fn test_lora_3d_forward_shape() {
        let lora = make_test_lora(8, 16, 4, 16.0);
        let x = Tensor::rand(0.0f32, 1.0f32, &[2, 5, 8], &Device::Cpu).unwrap();
        let out = lora.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 5, 16]);
    }

    #[test]
    fn test_lora_save_load_roundtrip_preserves_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");

        let mut lora = make_test_lora(8, 16, 4, 16.0);
        // Set lora_a to non-zero to have something meaningful
        let new_a = Tensor::rand(0.0f32, 1.0f32, &[4, 8], &Device::Cpu).unwrap();
        lora.set_lora_a(new_a.clone());
        lora.save(&path).unwrap();

        lora.load(&path, &Device::Cpu).unwrap();
        assert_eq!(lora.lora_a().dtype(), DType::F32);
        assert_eq!(lora.lora_b().dtype(), DType::F32);
        assert!(tensor_abs_diff(&new_a, lora.lora_a()) < 1e-6, "weights should survive save/load roundtrip");
    }

    #[test]
    fn test_lora_model_trainable_tensors_count() {
        let device = Device::Cpu;
        let config = LoraConfig {
            rank: 4,
            alpha: 16.0,
            dropout: 0.0,
        };
        let mut model = LoraModel::new(config);

        let n_layers = 3;
        for i in 0..n_layers {
            let weight = Tensor::rand(0.0f32, 1.0f32, &[16, 8], &device).unwrap();
            let frozen = Linear::new(weight, None);
            model
                .add_layer(format!("layer_{i}"), frozen, 8, 16, &device)
                .unwrap();
        }

        let tensors = model.trainable_tensors();
        assert_eq!(
            tensors.len(),
            n_layers * 2,
            "each layer contributes lora_a + lora_b"
        );
    }
}
