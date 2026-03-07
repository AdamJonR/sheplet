use thiserror::Error;

/// Errors that can occur in the embeddings crate.
#[derive(Debug, Error)]
pub enum EmbeddingsError {
    /// Failed to load or parse the tokenizer.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Failed to load model weights or config.
    #[error("model loading error: {0}")]
    ModelLoad(String),

    /// Failed during model inference.
    #[error("inference error: {0}")]
    Inference(#[from] candle_core::Error),

    /// Failed during file I/O.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to download model files from Hugging Face Hub.
    #[error("download error: {0}")]
    Download(String),

    /// JSON deserialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Generic error wrapper.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Convenience alias for `Result<T, EmbeddingsError>`.
pub type Result<T> = std::result::Result<T, EmbeddingsError>;
