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
    let mut courses = state.courses.write().await;
    match courses.load_bundle(&req.path, &state.base_dir).await {
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
