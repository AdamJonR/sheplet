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
            alpha: 16.0,
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
        let lora_out = x
            .matmul(&self.lora_a.t()?)?
            .matmul(&self.lora_b.t()?)?;
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
