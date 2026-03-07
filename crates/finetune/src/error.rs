#[derive(Debug, thiserror::Error)]
pub enum FinetuneError {
    #[error("data loading error: {0}")]
    DataLoading(String),
    #[error("insufficient memory: available {available_gb:.1}GB, recommended {recommended_gb:.1}GB")]
    InsufficientMemory {
        available_gb: f64,
        recommended_gb: f64,
    },
    #[error("training error: {0}")]
    Training(String),
    #[error("checkpoint error: {0}")]
    Checkpoint(String),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
