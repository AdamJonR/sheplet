#[derive(Debug, thiserror::Error)]
pub enum BundleError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("signature verification failed")]
    SignatureInvalid,
    #[error("missing required file in bundle: {0}")]
    MissingEntry(String),
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
