use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::course::CourseInfo;
use crate::handlers::bundles::ErrorResponse;

#[derive(Deserialize)]
pub struct SwitchRequest {
    course_id: String,
}

#[derive(Serialize)]
pub struct ActiveCourseResponse {
    pub course: Option<CourseInfo>,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/courses", get(list_courses))
        .route("/api/courses/switch", post(switch_course))
        .route("/api/courses/active", get(active_course))
}

async fn list_courses(State(state): State<Arc<AppState>>) -> Json<Vec<CourseInfo>> {
    let courses = state.courses.read().await;
    Json(courses.list_courses())
}

async fn switch_course(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SwitchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let mut courses = state.courses.write().await;
    match courses.switch_course(&req.course_id).await {
        Ok(()) => Ok(Json(serde_json::json!({
            "message": format!("Switched to course '{}'", req.course_id)
        }))),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

async fn active_course(State(state): State<Arc<AppState>>) -> Json<ActiveCourseResponse> {
    let courses = state.courses.read().await;
    let course = courses.active().map(|active| {
        let id = courses.active_id.clone().unwrap_or_default();
        CourseInfo {
            id,
            course_name: active.metadata.manifest.course_name.clone(),
            version: active.metadata.manifest.version.clone(),
            model_name: active.metadata.manifest.model_name.clone(),
            quantization: active.metadata.manifest.quantization.clone(),
            is_active: true,
        }
    });
    Json(ActiveCourseResponse { course })
}
