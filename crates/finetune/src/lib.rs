pub mod checkpoint;
pub mod data;
pub mod dpo;
pub mod error;
pub mod gemma_lora;
pub mod llama_lora;
pub mod lora;
pub mod mistral_lora;
pub mod model_utils;
pub mod phi3_lora;
pub mod preflight;
pub mod qwen2_lora;
pub mod sft;

pub use checkpoint::{save_checkpoint, load_checkpoint, CheckpointMeta};
pub use data::{DpoExample, SftExample, load_dpo_data, load_sft_data};
pub use dpo::{DpoConfig, train_dpo, train_dpo_full};
pub use error::FinetuneError;
pub use gemma_lora::{GemmaLoraModel, GemmaLoraTrainer};
pub use llama_lora::{LlamaLoraModel, LlamaLoraTrainer};
pub use lora::{LoraConfig, LoraLinear, LoraModel};
pub use mistral_lora::{MistralLoraModel, MistralLoraTrainer};
pub use model_utils::LoraTrainable;
pub use phi3_lora::{Phi3LoraModel, Phi3LoraTrainer};
pub use preflight::{preflight_check, HardwareInfo, PreflightReport};
pub use qwen2_lora::{Qwen2LoraModel, Qwen2LoraTrainer};
pub use sft::{SftConfig, Tokenize, train_sft};

#[cfg(test)]
pub(crate) mod test_fixtures {
    use crate::lora::{LoraConfig, LoraLinear};
    use crate::sft::Tokenize;
    use candle_core::{Device, Tensor};
    use candle_nn::Linear;

    pub struct DummyTokenizer;
    impl Tokenize for DummyTokenizer {
        fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
            Ok(text
                .split_whitespace()
                .enumerate()
                .map(|(i, _)| (i % 4) as u32)
                .collect())
        }
    }

    pub fn make_test_lora(in_f: usize, out_f: usize, rank: usize, alpha: f64) -> LoraLinear {
        let device = Device::Cpu;
        let weight = Tensor::rand(0.0f32, 1.0f32, &[out_f, in_f], &device).unwrap();
        let frozen = Linear::new(weight, None);
        let config = LoraConfig {
            rank,
            alpha,
            dropout: 0.0,
        };
        LoraLinear::new(frozen, in_f, out_f, &config, &device).unwrap()
    }

    /// Compute the sum of absolute element-wise differences between two tensors.
    pub fn tensor_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }
}
