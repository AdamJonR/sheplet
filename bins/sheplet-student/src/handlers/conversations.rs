use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;

use conversations::{ConversationSummary, export_as_txt};

use crate::app_state::AppState;
use crate::handlers::bundles::ErrorResponse;

#[derive(Deserialize)]
pub struct ListQuery {
    course_id: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateConversationRequest {
    course_id: String,
    title: Option<String>,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/conversations", get(list_conversations))
        .route("/api/conversations", post(create_conversation))
        .route("/api/conversations/{id}", get(get_conversation))
        .route("/api/conversations/{id}", delete(delete_conversation))
        .route(
            "/api/conversations/course/{course_id}",
            delete(clear_course),
        )
        .route(
            "/api/conversations/{id}/export",
            get(export_conversation),
        )
}

async fn list_conversations(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<ConversationSummary>>, (StatusCode, Json<ErrorResponse>)> {
    let result = if let Some(course_id) = &query.course_id {
        state.conversations.list_by_course(course_id)
    } else {
        state.conversations.list_all()
    };
    result
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })
}

async fn create_conversation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateConversationRequest>,
) -> Result<Json<conversations::Conversation>, (StatusCode, Json<ErrorResponse>)> {
    let title = req.title.as_deref().unwrap_or("New conversation");
    state
        .conversations
        .create_conversation(&req.course_id, title)
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })
}

async fn get_conversation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<conversations::Conversation>, (StatusCode, Json<ErrorResponse>)> {
    match state.conversations.get(&id) {
        Ok(Some(conv)) => Ok(Json(conv)),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Conversation not found".to_string(),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state
        .conversations
        .delete(&id)
        .map(|_| Json(serde_json::json!({"message": "Deleted"})))
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })
}

async fn clear_course(
    State(state): State<Arc<AppState>>,
    Path(course_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    state
        .conversations
        .clear_course(&course_id)
        .map(|_| Json(serde_json::json!({"message": "Cleared"})))
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })
}

async fn export_conversation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    match state.conversations.get(&id) {
        Ok(Some(conv)) => {
            let txt = export_as_txt(&conv);
            let filename = format!("{}.txt", conv.title.replace(' ', "_"));
            Ok((
                StatusCode::OK,
                [
                    (
                        axum::http::header::CONTENT_TYPE,
                        "text/plain; charset=utf-8".to_string(),
                    ),
                    (
                        axum::http::header::CONTENT_DISPOSITION,
                        format!("attachment; filename=\"{filename}\""),
                    ),
                ],
                txt,
            ))
        }
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Conversation not found".to_string(),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}
