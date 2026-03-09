use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::Device;
use conversations::{Citation, Message};
use db::VectorStore;
use embeddings::EmbeddingModel;
use lru::LruCache;
use tracing::debug;

use crate::config::RagConfig;
use crate::error::Result;
use crate::inference::ModelArch;
use crate::prompt::assemble_prompt_for_arch;

pub enum PreparedQuery {
    Ready {
        prompt: String,
        citations: Vec<Citation>,
    },
    Blocked {
        message: String,
    },
}

pub struct RagPipeline {
    embedder: EmbeddingModel,
    store: VectorStore,
    config: RagConfig,
    model_arch: ModelArch,
    query_cache: Mutex<LruCache<String, Arc<[f32]>>>,
}

impl RagPipeline {
    pub async fn new(
        embeddings_dir: impl AsRef<Path>,
        database_dir: impl AsRef<Path>,
        config: RagConfig,
        model_arch: ModelArch,
        device: &Device,
    ) -> Result<Self> {
        let embedder = EmbeddingModel::from_local(embeddings_dir, device)?;
        embedder.warmup();
        let store =
            VectorStore::open_or_create(database_dir, "chunks", embeddings::EMBEDDING_DIM).await?;
        store.create_index_if_needed(256).await;
        Ok(Self {
            embedder,
            store,
            config,
            model_arch,
            query_cache: Mutex::new(LruCache::new(NonZeroUsize::new(128).unwrap())),
        })
    }

    /// Embed a query string into a vector (cached for repeated queries).
    pub fn embed_query(&self, question: &str) -> Result<Arc<[f32]>> {
        // Check cache first (brief lock).
        if let Some(cached) = self.query_cache.lock().unwrap().get(question) {
            return Ok(Arc::clone(cached));
        }
        // Compute outside the lock.
        let embedding: Arc<[f32]> = self.embedder.embed(question)?.into();
        self.query_cache
            .lock()
            .unwrap()
            .put(question.to_string(), Arc::clone(&embedding));
        Ok(embedding)
    }

    /// Search the vector store for relevant chunks.
    pub async fn search_chunks(&self, query_vec: &[f32]) -> Result<Vec<db::SearchResult>> {
        let strategy = self.config.to_retrieval_strategy();
        Ok(self.store.search(query_vec, &strategy).await?)
    }

    /// Check relevance, build citations, and assemble prompt from search results.
    pub fn assemble_from_results(
        &self,
        results: &[db::SearchResult],
        history: &[Message],
        question: &str,
    ) -> PreparedQuery {
        let any_relevant = results.iter().any(|r| {
            let similarity = r.score as f64;
            debug!(
                source = %r.source_file,
                chunk = r.chunk_index,
                similarity,
                threshold = self.config.relevance_threshold,
                "retrieval score"
            );
            similarity >= self.config.relevance_threshold
        });

        if !any_relevant && !results.is_empty() {
            return PreparedQuery::Blocked {
                message: "I don't have enough relevant course materials to answer this question. Please ask something related to the course content.".to_string(),
            };
        }

        let citations: Vec<Citation> = results
            .iter()
            .map(|r| Citation {
                source_file: r.source_file.clone(),
                chunk_index: r.chunk_index,
                text_snippet: r.text.clone(),
            })
            .collect();

        let prompt = assemble_prompt_for_arch(
            self.model_arch,
            &self.config.system_prompt,
            results,
            history,
            question,
        );

        PreparedQuery::Ready { prompt, citations }
    }

    pub async fn prepare_prompt(
        &self,
        question: &str,
        history: &[Message],
    ) -> Result<PreparedQuery> {
        let query_vec = self.embed_query(question)?;
        let results = self.search_chunks(&query_vec).await?;
        Ok(self.assemble_from_results(&results, history, question))
    }

    pub fn update_settings(
        &mut self,
        strategy: Option<String>,
        k: Option<usize>,
        threshold: Option<f64>,
        lambda: Option<f32>,
    ) {
        if let Some(s) = strategy {
            self.config.retrieval_strategy = s;
        }
        if let Some(k) = k {
            self.config.top_k = k;
        }
        if let Some(t) = threshold {
            self.config.relevance_threshold = t;
        }
        if let Some(l) = lambda {
            self.config.mmr_lambda = l;
        }
    }

    pub fn config(&self) -> &RagConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_is_cosine_similarity() {
        // Verify the conversion formula: cosine_sim = 1.0 - squared_l2 / 2.0
        // Distance 0.0 → similarity 1.0
        assert!((1.0 - 0.0_f64 / 2.0 - 1.0).abs() < 1e-10);
        // Distance 2.0 → similarity 0.0
        assert!((1.0 - 2.0_f64 / 2.0 - 0.0).abs() < 1e-10);
        // Distance 1.0 → similarity 0.5
        assert!((1.0 - 1.0_f64 / 2.0 - 0.5).abs() < 1e-10);
        // Distance 4.0 → similarity -1.0
        assert!((1.0 - 4.0_f64 / 2.0 - (-1.0)).abs() < 1e-10);
    }
}
