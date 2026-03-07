use std::path::Path;

use conversations::{Citation, Message};
use db::VectorStore;
use embeddings::EmbeddingModel;

use crate::config::RagConfig;
use crate::error::Result;
use crate::prompt::assemble_prompt;

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
}

impl RagPipeline {
    pub async fn new(
        embeddings_dir: impl AsRef<Path>,
        database_dir: impl AsRef<Path>,
        config: RagConfig,
    ) -> Result<Self> {
        let embedder = EmbeddingModel::from_local(embeddings_dir)?;
        let store =
            VectorStore::open_or_create(database_dir, "chunks", embeddings::EMBEDDING_DIM).await?;
        Ok(Self {
            embedder,
            store,
            config,
        })
    }

    pub async fn prepare_prompt(
        &self,
        question: &str,
        history: &[Message],
    ) -> Result<PreparedQuery> {
        let query_vec = self.embedder.embed(question)?;
        let strategy = self.config.to_retrieval_strategy();
        let results = self.store.search(&query_vec, &strategy).await?;

        // Check relevance: LanceDB returns L2 distance (lower = closer)
        // Convert: similarity = 1.0 / (1.0 + distance)
        let any_relevant = results.iter().any(|r| {
            let similarity = 1.0 / (1.0 + r.score as f64);
            similarity >= self.config.relevance_threshold
        });

        if !any_relevant && !results.is_empty() {
            return Ok(PreparedQuery::Blocked {
                message: "I don't have enough relevant course materials to answer this question. Please ask something related to the course content.".to_string(),
            });
        }

        let citations: Vec<Citation> = results
            .iter()
            .map(|r| Citation {
                source_file: r.source_file.clone(),
                chunk_index: r.chunk_index,
                text_snippet: r.text.chars().take(200).collect(),
            })
            .collect();

        let prompt = assemble_prompt(&self.config.system_prompt, &results, history, question);

        Ok(PreparedQuery::Ready { prompt, citations })
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
