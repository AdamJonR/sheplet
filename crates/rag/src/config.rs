use serde::{Deserialize, Serialize};

/// Mirrors the CourseConfig from sheplet-instructor for JSON compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    pub system_prompt: String,
    pub retrieval_strategy: String,
    pub top_k: usize,
    pub relevance_threshold: f64,
    pub mmr_lambda: f32,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful tutor. Answer only from the provided course materials.".to_string(),
            retrieval_strategy: "top-k".to_string(),
            top_k: 5,
            relevance_threshold: 0.7,
            mmr_lambda: 0.5,
        }
    }
}

impl RagConfig {
    pub fn to_retrieval_strategy(&self) -> db::RetrievalStrategy {
        match self.retrieval_strategy.as_str() {
            "mmr" => db::RetrievalStrategy::Mmr {
                k: self.top_k,
                lambda: self.mmr_lambda,
            },
            _ => db::RetrievalStrategy::TopK { k: self.top_k },
        }
    }
}
