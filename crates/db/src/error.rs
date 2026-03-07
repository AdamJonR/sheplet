use thiserror::Error;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("LanceDB error: {0}")]
    Lance(#[from] lancedb::error::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("No records provided")]
    EmptyInsert,

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DbError>;
