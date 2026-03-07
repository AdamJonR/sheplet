use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, put};
use axum::{Json, Router};
use serde::Deserialize;

use crate::app_state::AppState;
use crate::project::CourseConfig;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/config", get(get_config))
        .route("/api/config", put(update_config))
}

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.to_string() }))
}

async fn get_config(
    State(state): State<Arc<AppState>>,
) -> Result<Json<CourseConfig>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?;

    let config = CourseConfig::load(path).unwrap_or_default();
    Ok(Json(config))
}

#[derive(Deserialize)]
struct ConfigUpdate {
    system_prompt: Option<String>,
    retrieval_strategy: Option<String>,
    top_k: Option<usize>,
    relevance_threshold: Option<f64>,
    mmr_lambda: Option<f32>,
}

async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ConfigUpdate>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?;

    let mut config = CourseConfig::load(path).unwrap_or_default();

    if let Some(prompt) = body.system_prompt {
        config.system_prompt = prompt;
    }
    if let Some(strategy) = &body.retrieval_strategy {
        match strategy.as_str() {
            "top-k" | "mmr" => config.retrieval_strategy = strategy.clone(),
            _ => return Err(err(StatusCode::BAD_REQUEST, "Invalid retrieval strategy. Use 'top-k' or 'mmr'.")),
        }
    }
    if let Some(k) = body.top_k {
        if !(1..=100).contains(&k) {
            return Err(err(StatusCode::BAD_REQUEST, "top_k must be between 1 and 100"));
        }
        config.top_k = k;
    }
    if let Some(threshold) = body.relevance_threshold {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(err(StatusCode::BAD_REQUEST, "relevance_threshold must be between 0.0 and 1.0"));
        }
        config.relevance_threshold = threshold;
    }
    if let Some(lambda) = body.mmr_lambda {
        if !(0.0..=1.0).contains(&lambda) {
            return Err(err(StatusCode::BAD_REQUEST, "mmr_lambda must be between 0.0 and 1.0"));
        }
        config.mmr_lambda = lambda;
    }

    config
        .save(path)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    Ok(Json(serde_json::json!({ "message": "Configuration updated" })))
}
