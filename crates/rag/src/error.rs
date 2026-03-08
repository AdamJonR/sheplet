#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("embeddings error: {0}")]
    Embeddings(#[from] embeddings::EmbeddingsError),
    #[error("database error: {0}")]
    Database(#[from] db::DbError),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("no active model loaded")]
    NoModel,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, RagError>;
