use serde::{Deserialize, Serialize};

/// Metadata about a `.sheplet` bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub course_name: String,
    pub model_name: String,
    pub quantization: String,
    /// ISO 8601 timestamp of when the bundle was built.
    pub build_timestamp: String,
    pub public_key_hex: String,
    pub public_key_fingerprint: String,
}
