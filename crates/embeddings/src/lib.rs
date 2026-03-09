//! Embedding model for Sheplet — wraps all-MiniLM-L6-v2 via Candle.
//!
//! Produces 384-dimensional L2-normalized sentence embeddings entirely on CPU.

pub mod download;
pub mod error;
pub mod model;
pub mod normalize;

pub use error::{EmbeddingsError, Result};
pub use model::EmbeddingModel;

/// The dimensionality of embeddings produced by all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: download the model, embed sentences, verify dimensions
    /// and cosine similarity ordering.
    ///
    /// This test is ignored by default because it downloads ~90MB from Hugging Face.
    /// Run with: `cargo test -p embeddings -- --ignored`
    #[test]
    #[ignore]
    fn test_embed_and_cosine_similarity() {
        let cache_dir = tempfile::tempdir().unwrap();
        let model = EmbeddingModel::download_and_load(cache_dir.path(), &candle_core::Device::Cpu).unwrap();

        let similar_a = "The cat sat on the mat.";
        let similar_b = "A cat was sitting on a rug.";
        let different = "Quantum computing uses qubits for parallel computation.";

        let embeddings = model
            .embed_batch(&[similar_a, similar_b, different])
            .unwrap();

        // Check dimensions
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), EMBEDDING_DIM);
        }

        // Cosine similarity (vectors are already L2-normalized, so dot product = cosine sim)
        let sim_ab = dot(&embeddings[0], &embeddings[1]);
        let sim_ac = dot(&embeddings[0], &embeddings[2]);
        let sim_bc = dot(&embeddings[1], &embeddings[2]);

        // Similar sentences should have higher cosine similarity than dissimilar ones
        assert!(
            sim_ab > sim_ac,
            "expected sim(a,b)={sim_ab} > sim(a,c)={sim_ac}"
        );
        assert!(
            sim_ab > sim_bc,
            "expected sim(a,b)={sim_ab} > sim(b,c)={sim_bc}"
        );
    }

    /// Integration test: single embed produces the same result as embed_batch with one item.
    #[test]
    #[ignore]
    fn test_embed_single_consistency() {
        let cache_dir = tempfile::tempdir().unwrap();
        let model = EmbeddingModel::download_and_load(cache_dir.path(), &candle_core::Device::Cpu).unwrap();

        let text = "Hello, world!";
        let single = model.embed(text).unwrap();
        let batch = model.embed_batch(&[text]).unwrap();

        assert_eq!(single.len(), EMBEDDING_DIM);
        assert_eq!(batch.len(), 1);

        // Values should be identical
        for (a, b) in single.iter().zip(batch[0].iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}
