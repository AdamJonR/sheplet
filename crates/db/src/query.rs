use serde::{Deserialize, Serialize};

/// Determines how retrieval is performed against the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetrievalStrategy {
    /// Return the k nearest neighbors by vector distance.
    TopK { k: usize },
    /// Maximal Marginal Relevance: balances relevance and diversity.
    /// `lambda` in [0, 1]: 1.0 = pure relevance, 0.0 = pure diversity.
    Mmr { k: usize, lambda: f32 },
}
