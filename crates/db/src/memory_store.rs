use crate::error::Result;
use crate::query::RetrievalStrategy;
use crate::store::{SearchResult, VectorStore};

/// Metadata for a single chunk (everything except the vector).
struct ChunkMeta {
    text: String,
    source_file: String,
    chunk_index: u32,
}

/// In-memory vector store for fast brute-force search.
///
/// Vectors are stored in a flat `Vec<f32>` for cache-friendly sequential access
/// and compiler auto-vectorization. Intended for read-only student workloads
/// where the dataset is small (~1000 chunks) and never changes during a session.
pub struct InMemoryStore {
    /// Flat storage: `vectors[i * dim .. (i+1) * dim]` is the i-th vector.
    vectors: Vec<f32>,
    metadata: Vec<ChunkMeta>,
    dimension: usize,
}

impl InMemoryStore {
    /// Load all rows from a LanceDB `VectorStore` into memory.
    pub async fn from_store(store: &VectorStore) -> Result<Self> {
        let rows = store.read_all().await?;
        let dimension = if rows.is_empty() {
            0
        } else {
            rows[0].0.len()
        };

        let mut vectors = Vec::with_capacity(rows.len() * dimension);
        let mut metadata = Vec::with_capacity(rows.len());

        for (vec, text, source_file, chunk_index) in rows {
            vectors.extend_from_slice(&vec);
            metadata.push(ChunkMeta {
                text,
                source_file,
                chunk_index,
            });
        }

        Ok(Self {
            vectors,
            metadata,
            dimension,
        })
    }

    /// Construct directly from chunk records (for testing).
    pub fn from_records(records: &[(Vec<f32>, String, String, u32)]) -> Self {
        let dimension = if records.is_empty() {
            0
        } else {
            records[0].0.len()
        };

        let mut vectors = Vec::with_capacity(records.len() * dimension);
        let mut metadata = Vec::with_capacity(records.len());

        for (vec, text, source_file, chunk_index) in records {
            vectors.extend_from_slice(vec);
            metadata.push(ChunkMeta {
                text: text.clone(),
                source_file: source_file.clone(),
                chunk_index: *chunk_index,
            });
        }

        Self {
            vectors,
            metadata,
            dimension,
        }
    }

    /// Return the top-k most similar vectors by dot product (= cosine similarity
    /// for L2-normalized vectors).
    pub fn search_top_k(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let n = self.metadata.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        let mut scores: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let start = i * self.dimension;
                let vec = &self.vectors[start..start + self.dimension];
                (dot_product(query, vec), i)
            })
            .collect();

        let k = k.min(n);
        // Partial sort: put top-k at the front.
        scores.select_nth_unstable_by(k - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        scores
            .iter()
            .map(|&(score, idx)| {
                let meta = &self.metadata[idx];
                SearchResult {
                    text: meta.text.clone(),
                    source_file: meta.source_file.clone(),
                    chunk_index: meta.chunk_index,
                    score,
                }
            })
            .collect()
    }

    /// Search using Maximal Marginal Relevance for diverse results.
    ///
    /// Retrieves `3 * k` candidates, then greedily selects `k` results that
    /// maximize: `lambda * relevance - (1 - lambda) * max_similarity_to_selected`.
    pub fn search_mmr(&self, query: &[f32], k: usize, lambda: f32) -> Vec<SearchResult> {
        let n = self.metadata.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        // Score all vectors.
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let start = i * self.dimension;
                let vec = &self.vectors[start..start + self.dimension];
                (dot_product(query, vec), i)
            })
            .collect();

        // Take top 3*k candidates.
        let fetch_count = (3 * k).min(n);
        scored.select_nth_unstable_by(fetch_count - 1, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(fetch_count);

        // Greedy MMR selection.
        let mut selected: Vec<(f32, usize)> = Vec::with_capacity(k);
        let mut remaining = scored;

        while selected.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (i, &(relevance, vec_idx)) in remaining.iter().enumerate() {
                let start = vec_idx * self.dimension;
                let vec = &self.vectors[start..start + self.dimension];

                let max_sim = if selected.is_empty() {
                    0.0
                } else {
                    selected
                        .iter()
                        .map(|&(_, sel_idx)| {
                            let sel_start = sel_idx * self.dimension;
                            let sel_vec =
                                &self.vectors[sel_start..sel_start + self.dimension];
                            dot_product(vec, sel_vec)
                        })
                        .fold(f32::NEG_INFINITY, f32::max)
                };

                let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_idx = i;
                }
            }

            selected.push(remaining.swap_remove(best_idx));
        }

        selected
            .iter()
            .map(|&(score, idx)| {
                let meta = &self.metadata[idx];
                SearchResult {
                    text: meta.text.clone(),
                    source_file: meta.source_file.clone(),
                    chunk_index: meta.chunk_index,
                    score,
                }
            })
            .collect()
    }

    /// Dispatch search based on a `RetrievalStrategy`.
    pub fn search(&self, query: &[f32], strategy: &RetrievalStrategy) -> Vec<SearchResult> {
        match strategy {
            RetrievalStrategy::TopK { k } => self.search_top_k(query, *k),
            RetrievalStrategy::Mmr { k, lambda } => self.search_mmr(query, *k, *lambda),
        }
    }

    /// Number of stored vectors.
    pub fn count(&self) -> usize {
        self.metadata.len()
    }
}

/// Dot product of two vectors.
///
/// For L2-normalized vectors this equals cosine similarity. The flat slice
/// layout enables the compiler to auto-vectorize with SIMD (SSE/AVX/NEON).
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple vector with a known pattern.
    fn make_vector(dim: usize, base: f32) -> Vec<f32> {
        (0..dim).map(|i| base + i as f32 * 0.1).collect()
    }

    /// Helper: L2-normalize a vector in place.
    fn normalize(v: &mut Vec<f32>) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }

    fn normalized_vector(dim: usize, base: f32) -> Vec<f32> {
        let mut v = make_vector(dim, base);
        normalize(&mut v);
        v
    }

    fn make_test_store() -> InMemoryStore {
        let dim = 8;
        // "close" has base 0.1 (nearest direction to query base 0.0),
        // "medium" has base 5.0, "far" has base 10.0.
        // After normalization, cosine similarity = dot product.
        let records: Vec<(Vec<f32>, String, String, u32)> = vec![
            (normalized_vector(dim, 10.0), "far".into(), "a.txt".into(), 0),
            (normalized_vector(dim, 0.1), "close".into(), "b.txt".into(), 1),
            (normalized_vector(dim, 5.0), "medium".into(), "c.txt".into(), 2),
        ];
        InMemoryStore::from_records(&records)
    }

    #[test]
    fn test_top_k_ordering() {
        let store = make_test_store();
        let mut query = make_vector(8, 0.0);
        normalize(&mut query);
        let results = store.search_top_k(&query, 3);
        assert_eq!(results.len(), 3);
        // Closest direction first (highest cosine similarity).
        assert_eq!(results[0].text, "close");
        assert_eq!(results[1].text, "medium");
        assert_eq!(results[2].text, "far");
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[test]
    fn test_mmr_diversification() {
        let dim = 8;
        let records: Vec<(Vec<f32>, String, String, u32)> = vec![
            (make_vector(dim, 0.01), "similar_1".into(), "a.txt".into(), 0),
            (make_vector(dim, 0.02), "similar_2".into(), "a.txt".into(), 1),
            (make_vector(dim, 0.03), "similar_3".into(), "a.txt".into(), 2),
            (
                (0..dim).map(|i| 0.05 * (dim - i) as f32).collect(),
                "different".into(),
                "b.txt".into(),
                3,
            ),
        ];
        let store = InMemoryStore::from_records(&records);
        let query = make_vector(dim, 0.0);

        let results = store.search_mmr(&query, 2, 0.3);
        assert_eq!(results.len(), 2);
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        assert!(
            texts.contains(&"different"),
            "MMR should select the diverse vector; got {:?}",
            texts
        );
    }

    #[test]
    fn test_count() {
        let store = make_test_store();
        assert_eq!(store.count(), 3);
    }

    #[test]
    fn test_search_empty() {
        let store = InMemoryStore::from_records(&[]);
        let results = store.search_top_k(&[1.0, 0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_topk_greater_than_records() {
        let records: Vec<(Vec<f32>, String, String, u32)> = vec![
            (vec![1.0, 0.0, 0.0, 0.0], "one".into(), "f.txt".into(), 0),
            (vec![0.0, 1.0, 0.0, 0.0], "two".into(), "f.txt".into(), 1),
        ];
        let store = InMemoryStore::from_records(&records);
        let results = store.search_top_k(&[1.0, 0.0, 0.0, 0.0], 100);
        assert_eq!(results.len(), 2, "should return all available records");
    }

    #[test]
    fn test_normalized_cosine_similarity() {
        // With L2-normalized vectors, dot product should equal cosine similarity.
        let mut v1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut v2 = vec![1.0, 0.0, 0.0, 0.0];
        normalize(&mut v1);
        normalize(&mut v2);

        let records: Vec<(Vec<f32>, String, String, u32)> =
            vec![(v2.clone(), "unit_x".into(), "f.txt".into(), 0)];
        let store = InMemoryStore::from_records(&records);
        let results = store.search_top_k(&v1, 1);
        assert_eq!(results.len(), 1);
        // Score should be v1·v2 which is v1[0] since v2 is unit x.
        let expected = v1[0];
        assert!(
            (results[0].score - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            results[0].score
        );
    }

    #[tokio::test]
    async fn test_round_trip_from_vector_store() {
        let dir = tempfile::TempDir::new().unwrap();
        let dim = 4;
        let lance_store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        let records = vec![
            crate::store::ChunkRecord {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                text: "alpha".into(),
                source_file: "a.txt".into(),
                chunk_index: 0,
            },
            crate::store::ChunkRecord {
                vector: vec![0.0, 1.0, 0.0, 0.0],
                text: "beta".into(),
                source_file: "b.txt".into(),
                chunk_index: 1,
            },
            crate::store::ChunkRecord {
                vector: vec![0.0, 0.0, 1.0, 0.0],
                text: "gamma".into(),
                source_file: "c.txt".into(),
                chunk_index: 2,
            },
        ];
        lance_store.insert(&records).await.unwrap();

        let mem_store = InMemoryStore::from_store(&lance_store).await.unwrap();
        assert_eq!(mem_store.count(), 3);

        // Search should find "alpha" closest to [1,0,0,0].
        let results = mem_store.search_top_k(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "alpha");

        // Score for exact match should be 1.0 (dot product of identical unit vectors).
        assert!(
            (results[0].score - 1.0).abs() < 1e-6,
            "exact match score should be ~1.0, got {}",
            results[0].score
        );
    }
}
