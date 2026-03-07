use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::routing::get;
use axum::{Json, Router};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::app_state::AppState;
use crate::task_manager::{TaskEvent, TaskInfo};

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/tasks", get(list_tasks))
        .route("/api/tasks/{id}", get(get_task))
        .route("/api/tasks/{id}/stream", get(stream_task))
}

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.to_string() }))
}

async fn list_tasks(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<TaskInfo>> {
    Json(state.tasks.list_tasks().await)
}

async fn get_task(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<TaskInfo>, (StatusCode, Json<ErrorResponse>)> {
    state
        .tasks
        .get_task(&id)
        .await
        .map(Json)
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "Task not found"))
}

async fn stream_task(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)>
{
    let rx = state
        .tasks
        .subscribe(&id)
        .await
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "Task not found"))?;

    let stream = BroadcastStream::new(rx).filter_map(|result| match result {
        Ok(event) => {
            let sse_event = match &event {
                TaskEvent::StepStarted { .. } => Event::default()
                    .event("step")
                    .data(serde_json::to_string(&event).unwrap_or_default()),
                TaskEvent::StepCompleted { .. } => Event::default()
                    .event("step_done")
                    .data(serde_json::to_string(&event).unwrap_or_default()),
                TaskEvent::Progress { .. } => Event::default()
                    .event("progress")
                    .data(serde_json::to_string(&event).unwrap_or_default()),
                TaskEvent::Log { message } => {
                    Event::default().event("log").data(message.clone())
                }
                TaskEvent::Done { .. } => Event::default()
                    .event("done")
                    .data(serde_json::to_string(&event).unwrap_or_default()),
            };
            Some(Ok(sse_event))
        }
        Err(_) => None,
    });

    Ok(Sse::new(stream))
}
