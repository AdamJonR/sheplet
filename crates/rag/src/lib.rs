pub mod config;
pub mod error;
pub mod inference;
pub mod pipeline;
pub mod prompt;
pub mod quantize;
pub mod quantized_phi3;

pub use config::RagConfig;
pub use error::{RagError, Result};
pub use inference::{detect_model_arch, ModelArch, PhiGenerator, TextGenerator};
pub use db::SearchResult;
pub use pipeline::{PreparedQuery, RagPipeline};
pub use quantize::quantize_safetensors_to_gguf;
