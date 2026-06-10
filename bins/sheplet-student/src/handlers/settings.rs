use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::handlers::bundles::ErrorResponse;

#[derive(Serialize)]
pub struct SettingsResponse {
    retrieval_strategy: String,
    top_k: usize,
    relevance_threshold: f64,
    mmr_lambda: f32,
    /// Instructor-set floor: the threshold cannot be lowered below this.
    min_relevance_threshold: f64,
}

#[derive(Deserialize)]
pub struct UpdateSettingsRequest {
    retrieval_strategy: Option<String>,
    top_k: Option<usize>,
    relevance_threshold: Option<f64>,
    mmr_lambda: Option<f32>,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/settings", get(get_settings))
        .route("/api/settings", put(update_settings))
}

async fn get_settings(
    State(state): State<Arc<AppState>>,
) -> Result<Json<SettingsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let courses = state.courses.read().await;
    let active = courses.active().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No active course".to_string(),
            }),
        )
    })?;
    let pipeline = active.pipeline.read().await;
    let config = pipeline.config();
    Ok(Json(SettingsResponse {
        retrieval_strategy: config.retrieval_strategy.clone(),
        top_k: config.top_k,
        relevance_threshold: config.relevance_threshold,
        mmr_lambda: config.mmr_lambda,
        min_relevance_threshold: active.metadata.config.relevance_threshold,
    }))
}

async fn update_settings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateSettingsRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // Validate inputs
    if let Some(top_k) = req.top_k
        && (top_k == 0 || top_k > 100) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "top_k must be between 1 and 100".to_string(),
                }),
            ));
        }
    if let Some(threshold) = req.relevance_threshold
        && !(0.0..=1.0).contains(&threshold) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "relevance_threshold must be between 0.0 and 1.0".to_string(),
                }),
            ));
        }
    if let Some(lambda) = req.mmr_lambda
        && !(0.0..=1.0).contains(&lambda) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "mmr_lambda must be between 0.0 and 1.0".to_string(),
                }),
            ));
        }
    if let Some(ref strategy) = req.retrieval_strategy
        && !["similarity", "mmr"].contains(&strategy.as_str()) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: "retrieval_strategy must be 'similarity' or 'mmr'".to_string(),
                }),
            ));
        }

    let courses = state.courses.read().await;
    let active = courses.active().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No active course".to_string(),
            }),
        )
    })?;

    // Academic-integrity lock: the relevance threshold is set by the
    // instructor in the bundle config and may only be raised, never lowered —
    // lowering it would let off-syllabus queries through the blocking gate.
    let instructor_threshold = active.metadata.config.relevance_threshold;
    if let Some(threshold) = req.relevance_threshold
        && threshold < instructor_threshold {
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: format!(
                        "relevance_threshold is locked by your instructor at a minimum of {instructor_threshold}"
                    ),
                }),
            ));
        }

    active.pipeline.write().await.update_settings(
        req.retrieval_strategy,
        req.top_k,
        req.relevance_threshold,
        req.mmr_lambda,
    );
    Ok(Json(serde_json::json!({"message": "Settings updated"})))
}
