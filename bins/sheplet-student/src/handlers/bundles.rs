use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;

#[derive(Deserialize)]
pub struct LoadBundleRequest {
    path: String,
    trusted_fingerprint: String,
}

#[derive(Serialize)]
pub struct LoadBundleResponse {
    course_id: String,
    message: String,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/api/bundles/load", post(load_bundle))
}

async fn load_bundle(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadBundleRequest>,
) -> Result<Json<LoadBundleResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate fingerprint format: must be exactly 16 hex characters
    if req.trusted_fingerprint.len() != 16
        || !req
            .trusted_fingerprint
            .chars()
            .all(|c| c.is_ascii_hexdigit())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid fingerprint: must be exactly 16 hex characters".to_string(),
            }),
        ));
    }

    let mut courses = state.courses.write().await;
    match courses
        .load_bundle(&req.path, &state.base_dir, &req.trusted_fingerprint, state.no_adapter)
        .await
    {
        Ok(course_id) => Ok(Json(LoadBundleResponse {
            message: format!("Bundle loaded successfully as '{course_id}'"),
            course_id,
        })),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}
