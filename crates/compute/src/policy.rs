use candle_core::Device;

use crate::detect::best_gpu_or_cpu;

/// The type of ML workload being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// LLM inference (3.8B params) — benefits greatly from GPU.
    Inference,
    /// LoRA fine-tuning (forward + backward passes) — benefits from GPU.
    Training,
    /// Sentence embedding (22M BERT) — GPU dispatch overhead negates speedup.
    Embedding,
    /// SafeTensors → GGUF quantization — one-time, integer-heavy.
    Quantization,
}

/// Select the appropriate device for the given workload.
///
/// - `Inference` and `Training` are routed to the best available GPU (or CPU fallback).
/// - `Embedding` and `Quantization` always use CPU.
pub fn device_for(workload: Workload) -> Device {
    match workload {
        Workload::Inference | Workload::Training => best_gpu_or_cpu(),
        Workload::Embedding | Workload::Quantization => Device::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_always_cpu() {
        let device = device_for(Workload::Embedding);
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn quantization_always_cpu() {
        let device = device_for(Workload::Quantization);
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn inference_returns_device() {
        // Without GPU features, falls back to CPU
        let device = device_for(Workload::Inference);
        let _ = device;
    }

    #[test]
    fn training_returns_device() {
        // Without GPU features, falls back to CPU
        let device = device_for(Workload::Training);
        let _ = device;
    }
}
