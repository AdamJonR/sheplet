use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
    UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::Table;
use tracing::warn;
use serde::{Deserialize, Serialize};

use crate::error::{DbError, Result};
use crate::query::RetrievalStrategy;

/// Number of IVF partitions to probe during vector search.
/// Higher values improve recall at the cost of latency. Default LanceDB is 1,
/// which misses results when an IVF index is present.
const SEARCH_NPROBES: usize = 10;

/// A record to be stored in the vector database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub vector: Vec<f32>,
    pub text: String,
    pub source_file: String,
    pub chunk_index: u32,
}

/// A result returned from a vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub text: String,
    pub source_file: String,
    pub chunk_index: u32,
    /// Cosine similarity score (higher = more similar, range [-1, 1]).
    pub score: f32,
}

/// LanceDB-backed vector store.
pub struct VectorStore {
    db: lancedb::Connection,
    table_name: String,
    dimension: usize,
}

impl VectorStore {
    /// Open an existing table or create a new one at the given path.
    pub async fn open_or_create(
        path: impl AsRef<Path>,
        table_name: &str,
        dimension: usize,
    ) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let db = lancedb::connect(&path_str).execute().await?;

        // Check if the table already exists.
        let existing_tables = db.table_names().execute().await?;
        let table_exists = existing_tables.iter().any(|n| n == table_name);

        if !table_exists {
            // Create the table with an empty batch that carries the schema.
            let schema = Self::arrow_schema(dimension);
            let batch = Self::empty_batch(&schema, dimension)?;
            let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
            db.create_table(table_name, Box::new(batches))
                .execute()
                .await?;
        }

        Ok(Self {
            db,
            table_name: table_name.to_string(),
            dimension,
        })
    }

    /// Insert chunk records into the table.
    pub async fn insert(&self, records: &[ChunkRecord]) -> Result<()> {
        if records.is_empty() {
            return Err(DbError::EmptyInsert);
        }

        // Validate dimensions.
        for record in records {
            if record.vector.len() != self.dimension {
                return Err(DbError::DimensionMismatch {
                    expected: self.dimension,
                    got: record.vector.len(),
                });
            }
        }

        let table = self.open_table().await?;
        let schema = Self::arrow_schema(self.dimension);
        let batch = Self::records_to_batch(records, &schema, self.dimension)?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table.add(Box::new(batches)).execute().await?;
        Ok(())
    }

    /// Search for the top-k nearest neighbors of the query vector.
    pub async fn search_top_k(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let table = self.open_table().await?;
        let mut stream = table
            .vector_search(query_vector)?
            .limit(k)
            .nprobes(SEARCH_NPROBES)
            .select(Select::columns(&["text", "source_file", "chunk_index"]))
            .execute()
            .await?;

        let batches = Self::collect_batches(&mut stream).await?;
        let results = Self::batches_to_results(&batches)?;
        Ok(results)
    }

    /// Search using Maximal Marginal Relevance for diverse results.
    ///
    /// Retrieves `3 * k` candidates via vector search, then greedily selects
    /// `k` results that maximize:
    ///   `lambda * relevance - (1 - lambda) * max_similarity_to_selected`
    pub async fn search_mmr(
        &self,
        query_vector: &[f32],
        k: usize,
        lambda: f32,
    ) -> Result<Vec<SearchResult>> {
        let fetch_count = 3 * k;
        let table = self.open_table().await?;
        let mut stream = table
            .vector_search(query_vector)?
            .limit(fetch_count)
            .nprobes(SEARCH_NPROBES)
            .execute()
            .await?;

        let batches = Self::collect_batches(&mut stream).await?;

        // Collect candidates with their vectors.
        let mut candidates: Vec<(SearchResult, Vec<f32>)> = Vec::new();
        for batch in &batches {
            let text_arr = string_column(batch, "text");
            let source_arr = string_column(batch, "source_file");
            let chunk_arr = uint32_column(batch, "chunk_index");
            let dist_arr = float_column(batch, "_distance");
            let vector_col = batch
                .column_by_name("vector")
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let result = SearchResult {
                    text: text_arr.value(i).to_string(),
                    source_file: source_arr.value(i).to_string(),
                    chunk_index: chunk_arr.value(i),
                    score: 1.0 - dist_arr.value(i) / 2.0,
                };
                let vec_arr = vector_col
                    .value(i)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values()
                    .to_vec();
                candidates.push((result, vec_arr));
            }
        }

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Greedy MMR selection.
        let mut selected: Vec<(SearchResult, Vec<f32>)> = Vec::with_capacity(k);
        let mut remaining: Vec<(SearchResult, Vec<f32>)> = candidates;

        while selected.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (i, (result, vec)) in remaining.iter().enumerate() {
                let relevance = result.score;

                // Max similarity to any already-selected result.
                // Vectors are L2-normalized, so dot product = cosine similarity.
                let max_sim = if selected.is_empty() {
                    0.0
                } else {
                    selected
                        .iter()
                        .map(|(_, sel_vec)| dot_product(vec, sel_vec))
                        .fold(f32::NEG_INFINITY, f32::max)
                };

                let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
                if mmr > best_score {
                    best_score = mmr;
                    best_idx = i;
                }
            }

            selected.push(remaining.swap_remove(best_idx));
        }

        Ok(selected.into_iter().map(|(r, _)| r).collect())
    }

    /// Dispatch a search based on the given retrieval strategy.
    pub async fn search(
        &self,
        query_vector: &[f32],
        strategy: &RetrievalStrategy,
    ) -> Result<Vec<SearchResult>> {
        match strategy {
            RetrievalStrategy::TopK { k } => self.search_top_k(query_vector, *k).await,
            RetrievalStrategy::Mmr { k, lambda } => {
                self.search_mmr(query_vector, *k, *lambda).await
            }
        }
    }

    /// Return the number of rows in the table.
    pub async fn count(&self) -> Result<usize> {
        let table = self.open_table().await?;
        let count = table.count_rows(None).await?;
        Ok(count)
    }

    /// Read all rows from the table for bulk loading into memory.
    ///
    /// Returns `(vector, text, source_file, chunk_index)` tuples.
    pub async fn read_all(&self) -> Result<Vec<(Vec<f32>, String, String, u32)>> {
        let table = self.open_table().await?;
        let mut stream = table
            .query()
            .select(Select::columns(&[
                "vector",
                "text",
                "source_file",
                "chunk_index",
            ]))
            .execute()
            .await?;

        let batches = Self::collect_batches(&mut stream).await?;
        let mut rows = Vec::new();
        for batch in &batches {
            let text_arr = string_column(batch, "text");
            let source_arr = string_column(batch, "source_file");
            let chunk_arr = uint32_column(batch, "chunk_index");
            let vector_col = batch
                .column_by_name("vector")
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();

            for i in 0..batch.num_rows() {
                let vec_arr = vector_col
                    .value(i)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values()
                    .to_vec();
                rows.push((
                    vec_arr,
                    text_arr.value(i).to_string(),
                    source_arr.value(i).to_string(),
                    chunk_arr.value(i),
                ));
            }
        }
        Ok(rows)
    }

    /// Delete all rows from the table.
    pub async fn clear(&self) -> Result<()> {
        let table = self.open_table().await?;
        table.delete("true").await?;
        Ok(())
    }

    /// Create an ANN index if the table has at least `min_rows` rows.
    ///
    /// Best-effort: logs a warning on failure rather than returning an error,
    /// since brute-force search still works without an index.
    pub async fn create_index_if_needed(&self, min_rows: usize) {
        let Ok(table) = self.open_table().await else {
            return;
        };
        let Ok(count) = table.count_rows(None).await else {
            return;
        };
        if count >= min_rows
            && let Err(e) = table
                .create_index(&["vector"], lancedb::index::Index::Auto)
                .execute()
                .await
            {
                warn!("Failed to create ANN index (brute-force will be used): {e}");
            }
    }

    // ---- internal helpers ----

    async fn collect_batches(
        stream: &mut Pin<Box<dyn lancedb::arrow::RecordBatchStream + Send>>,
    ) -> Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        while let Some(result) = stream.next().await {
            batches.push(result.map_err(|e| DbError::Other(e.to_string()))?);
        }
        Ok(batches)
    }

    async fn open_table(&self) -> Result<Table> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        Ok(table)
    }

    fn arrow_schema(dimension: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                true,
            ),
            Field::new("text", DataType::Utf8, false),
            Field::new("source_file", DataType::Utf8, false),
            Field::new("chunk_index", DataType::UInt32, false),
        ]))
    }

    fn empty_batch(schema: &Arc<Schema>, dimension: usize) -> Result<RecordBatch> {
        let float_array = Float32Array::from(Vec::<f32>::new());
        let vector_array = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimension as i32,
            Arc::new(float_array),
            None,
        )?;
        let text_array = StringArray::from(Vec::<&str>::new());
        let source_array = StringArray::from(Vec::<&str>::new());
        let chunk_array = UInt32Array::from(Vec::<u32>::new());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(vector_array) as ArrayRef,
                Arc::new(text_array) as ArrayRef,
                Arc::new(source_array) as ArrayRef,
                Arc::new(chunk_array) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    fn records_to_batch(
        records: &[ChunkRecord],
        schema: &Arc<Schema>,
        dimension: usize,
    ) -> Result<RecordBatch> {
        let all_floats: Vec<f32> = records
            .iter()
            .flat_map(|r| r.vector.iter().copied())
            .collect();
        let float_array = Float32Array::from(all_floats);
        let vector_array = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dimension as i32,
            Arc::new(float_array),
            None,
        )?;

        let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
        let text_array = StringArray::from(texts);

        let sources: Vec<&str> = records.iter().map(|r| r.source_file.as_str()).collect();
        let source_array = StringArray::from(sources);

        let chunks: Vec<u32> = records.iter().map(|r| r.chunk_index).collect();
        let chunk_array = UInt32Array::from(chunks);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(vector_array) as ArrayRef,
                Arc::new(text_array) as ArrayRef,
                Arc::new(source_array) as ArrayRef,
                Arc::new(chunk_array) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    fn batches_to_results(batches: &[RecordBatch]) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        for batch in batches {
            let text_arr = string_column(batch, "text");
            let source_arr = string_column(batch, "source_file");
            let chunk_arr = uint32_column(batch, "chunk_index");
            let dist_arr = float_column(batch, "_distance");

            for i in 0..batch.num_rows() {
                results.push(SearchResult {
                    text: text_arr.value(i).to_string(),
                    source_file: source_arr.value(i).to_string(),
                    chunk_index: chunk_arr.value(i),
                    score: 1.0 - dist_arr.value(i) / 2.0,
                });
            }
        }
        Ok(results)
    }
}

fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a StringArray {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
}

fn float_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float32Array {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
}

fn uint32_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a UInt32Array {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
}

/// Dot product of two vectors.
///
/// For L2-normalized vectors this equals cosine similarity, avoiding
/// the redundant norm computations of a full cosine_similarity call.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_vector(dimension: usize, base: f32) -> Vec<f32> {
        (0..dimension).map(|i| base + i as f32 * 0.1).collect()
    }

    #[tokio::test]
    async fn test_top_k_ordering() {
        let dir = TempDir::new().unwrap();
        let dim = 8;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        // Insert three vectors at different distances from our query.
        let query = make_vector(dim, 0.0);
        let records = vec![
            ChunkRecord {
                vector: make_vector(dim, 10.0), // far
                text: "far".into(),
                source_file: "a.txt".into(),
                chunk_index: 0,
            },
            ChunkRecord {
                vector: make_vector(dim, 0.1), // close
                text: "close".into(),
                source_file: "b.txt".into(),
                chunk_index: 1,
            },
            ChunkRecord {
                vector: make_vector(dim, 5.0), // medium
                text: "medium".into(),
                source_file: "c.txt".into(),
                chunk_index: 2,
            },
        ];
        store.insert(&records).await.unwrap();

        let results = store.search_top_k(&query, 3).await.unwrap();
        assert_eq!(results.len(), 3);
        // Closest first (highest similarity).
        assert_eq!(results[0].text, "close");
        assert_eq!(results[1].text, "medium");
        assert_eq!(results[2].text, "far");
        // Scores should be in descending order (cosine similarity).
        assert!(results[0].score >= results[1].score);
        assert!(results[1].score >= results[2].score);
    }

    #[tokio::test]
    async fn test_mmr_diversification() {
        let dir = TempDir::new().unwrap();
        let dim = 8;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        let query = make_vector(dim, 0.0);
        // Three nearly identical vectors and one that's close in distance but
        // orthogonally different, so MMR's diversity term should prefer it.
        let records = vec![
            ChunkRecord {
                vector: make_vector(dim, 0.01),
                text: "similar_1".into(),
                source_file: "a.txt".into(),
                chunk_index: 0,
            },
            ChunkRecord {
                vector: make_vector(dim, 0.02),
                text: "similar_2".into(),
                source_file: "a.txt".into(),
                chunk_index: 1,
            },
            ChunkRecord {
                vector: make_vector(dim, 0.03),
                text: "similar_3".into(),
                source_file: "a.txt".into(),
                chunk_index: 2,
            },
            ChunkRecord {
                // Same magnitude range but reversed direction — close distance to query
                // but very different from the cluster of similar vectors.
                vector: (0..dim).map(|i| 0.05 * (dim - i) as f32).collect(),
                text: "different".into(),
                source_file: "b.txt".into(),
                chunk_index: 3,
            },
        ];
        store.insert(&records).await.unwrap();

        // With MMR (low lambda = strong diversity), the different one should appear
        // among the top 2 results because it's dissimilar to the first selected.
        let results = store.search_mmr(&query, 2, 0.3).await.unwrap();
        assert_eq!(results.len(), 2);
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        // The first result should be one of the similar vectors (closest),
        // and the second should be "different" due to diversity.
        assert!(
            texts.contains(&"different"),
            "MMR should select the diverse vector; got {:?}",
            texts
        );
    }

    #[tokio::test]
    async fn test_round_trip_metadata() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        let records = vec![ChunkRecord {
            vector: vec![1.0, 2.0, 3.0, 4.0],
            text: "hello world".into(),
            source_file: "notes.pdf".into(),
            chunk_index: 42,
        }];
        store.insert(&records).await.unwrap();

        let results = store.search_top_k(&[1.0, 2.0, 3.0, 4.0], 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "hello world");
        assert_eq!(results[0].source_file, "notes.pdf");
        assert_eq!(results[0].chunk_index, 42);
    }

    #[tokio::test]
    async fn test_count() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        assert_eq!(store.count().await.unwrap(), 0);

        let records = vec![
            ChunkRecord {
                vector: vec![1.0, 2.0, 3.0, 4.0],
                text: "a".into(),
                source_file: "f.txt".into(),
                chunk_index: 0,
            },
            ChunkRecord {
                vector: vec![5.0, 6.0, 7.0, 8.0],
                text: "b".into(),
                source_file: "f.txt".into(),
                chunk_index: 1,
            },
        ];
        store.insert(&records).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_search_empty_table() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        let query = vec![1.0f32, 0.0, 0.0, 0.0];
        let results = store.search_top_k(&query, 5).await.unwrap();
        assert!(results.is_empty(), "search on empty table should return no results");
    }

    #[tokio::test]
    async fn test_clear_and_recount() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        let records = vec![
            ChunkRecord {
                vector: vec![1.0, 2.0, 3.0, 4.0],
                text: "a".into(),
                source_file: "f.txt".into(),
                chunk_index: 0,
            },
            ChunkRecord {
                vector: vec![5.0, 6.0, 7.0, 8.0],
                text: "b".into(),
                source_file: "f.txt".into(),
                chunk_index: 1,
            },
        ];
        store.insert(&records).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_topk_greater_than_records() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        let records = vec![
            ChunkRecord {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                text: "one".into(),
                source_file: "f.txt".into(),
                chunk_index: 0,
            },
            ChunkRecord {
                vector: vec![0.0, 1.0, 0.0, 0.0],
                text: "two".into(),
                source_file: "f.txt".into(),
                chunk_index: 1,
            },
        ];
        store.insert(&records).await.unwrap();

        // Ask for 100 results when only 2 exist
        let results = store.search_top_k(&[1.0, 0.0, 0.0, 0.0], 100).await.unwrap();
        assert_eq!(results.len(), 2, "should return all available records");
    }

    #[tokio::test]
    async fn test_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        let store = VectorStore::open_or_create(dir.path(), "test", dim)
            .await
            .unwrap();

        let records = vec![ChunkRecord {
            vector: vec![1.0, 2.0, 3.0], // wrong: 3 instead of 4
            text: "oops".into(),
            source_file: "f.txt".into(),
            chunk_index: 0,
        }];
        let err = store.insert(&records).await.unwrap_err();
        assert!(
            matches!(err, DbError::DimensionMismatch { expected: 4, got: 3 }),
            "Expected DimensionMismatch, got: {:?}",
            err
        );
    }
}
