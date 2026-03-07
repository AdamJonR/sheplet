use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use conversations::{Citation, Message, Role};
use rag::{PreparedQuery, TextGenerator};

use crate::app_state::AppState;
use crate::handlers::bundles::ErrorResponse;

#[derive(Deserialize)]
pub struct ChatRequest {
    message: String,
    conversation_id: Option<String>,
}

#[derive(Serialize)]
pub struct ChatSyncResponse {
    response: String,
    conversation_id: String,
    citations: Vec<Citation>,
    blocked: bool,
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/chat", post(chat_stream))
        .route("/api/chat/sync", post(chat_sync))
}

fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    let days = secs / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{mins:02}:{s:02}Z")
}

async fn get_or_create_conversation(
    state: &AppState,
    conversation_id: Option<&str>,
    course_id: &str,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    if let Some(id) = conversation_id {
        if state.conversations.get(id).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?.is_some() {
            return Ok(id.to_string());
        }
    }
    let conv = state
        .conversations
        .create_conversation(course_id, "New conversation")
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;
    Ok(conv.id)
}

async fn chat_sync(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatSyncResponse>, (StatusCode, Json<ErrorResponse>)> {
    let courses = state.courses.read().await;
    let active = courses.active().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No active course. Load a bundle first.".to_string(),
            }),
        )
    })?;
    let course_id = courses.active_id.clone().unwrap_or_default();

    let conv_id =
        get_or_create_conversation(&state, req.conversation_id.as_deref(), &course_id).await?;

    // Save user message
    let user_msg = Message {
        role: Role::User,
        content: req.message.clone(),
        timestamp: now_iso(),
        citations: vec![],
    };
    state
        .conversations
        .append_message(&conv_id, user_msg)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    // Get conversation history
    let conv = state.conversations.get(&conv_id).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?.ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Conversation not found after creation".to_string(),
            }),
        )
    })?;

    // Prepare prompt
    let prepared = active
        .pipeline
        .prepare_prompt(&req.message, &conv.messages[..conv.messages.len().saturating_sub(1)])
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    match prepared {
        PreparedQuery::Blocked { message } => {
            let assistant_msg = Message {
                role: Role::Assistant,
                content: message.clone(),
                timestamp: now_iso(),
                citations: vec![],
            };
            let _ = state.conversations.append_message(&conv_id, assistant_msg);

            Ok(Json(ChatSyncResponse {
                response: message,
                conversation_id: conv_id,
                citations: vec![],
                blocked: true,
            }))
        }
        PreparedQuery::Ready { prompt, citations } => {
            let mut generator = active.generator.lock().unwrap();
            let response = generator.generate(&prompt, 512).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;

            let assistant_msg = Message {
                role: Role::Assistant,
                content: response.clone(),
                timestamp: now_iso(),
                citations: citations.clone(),
            };
            let _ = state.conversations.append_message(&conv_id, assistant_msg);

            Ok(Json(ChatSyncResponse {
                response,
                conversation_id: conv_id,
                citations,
                blocked: false,
            }))
        }
    }
}

async fn chat_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ErrorResponse>)> {
    let courses = state.courses.read().await;
    let active = courses.active().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "No active course. Load a bundle first.".to_string(),
            }),
        )
    })?;
    let course_id = courses.active_id.clone().unwrap_or_default();

    let conv_id =
        get_or_create_conversation(&state, req.conversation_id.as_deref(), &course_id).await?;

    // Save user message
    let user_msg = Message {
        role: Role::User,
        content: req.message.clone(),
        timestamp: now_iso(),
        citations: vec![],
    };
    state
        .conversations
        .append_message(&conv_id, user_msg)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let conv = state.conversations.get(&conv_id).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?.ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Conversation not found after creation".to_string(),
            }),
        )
    })?;

    let prepared = active
        .pipeline
        .prepare_prompt(&req.message, &conv.messages[..conv.messages.len().saturating_sub(1)])
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let (citations, receiver) = match prepared {
        PreparedQuery::Blocked { message } => {
            let assistant_msg = Message {
                role: Role::Assistant,
                content: message.clone(),
                timestamp: now_iso(),
                citations: vec![],
            };
            let _ = state.conversations.append_message(&conv_id, assistant_msg);

            // Send blocked message as a single SSE event
            let (tx, rx) = std::sync::mpsc::channel();
            let _ = tx.send(Ok(message));
            drop(tx);
            return Ok(Sse::new(sse_from_receiver(rx, vec![], conv_id, state.clone(), true)));
        }
        PreparedQuery::Ready { prompt, citations } => {
            let mut generator = active.generator.lock().unwrap();
            let rx = generator.generate_stream(&prompt, 512).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            })?;
            (citations, rx)
        }
    };

    Ok(Sse::new(sse_from_receiver(receiver, citations, conv_id, state.clone(), false)))
}

fn sse_from_receiver(
    receiver: std::sync::mpsc::Receiver<rag::Result<String>>,
    citations: Vec<Citation>,
    conv_id: String,
    state: Arc<AppState>,
    blocked: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel(100);

    tokio::spawn(async move {
        let mut full_response = String::new();
        while let Ok(result) = receiver.recv() {
            match result {
                Ok(token) => {
                    full_response.push_str(&token);
                    let event = Event::default().data(&token);
                    if tx.send(Ok(event)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let event = Event::default()
                        .event("error")
                        .data(e.to_string());
                    let _ = tx.send(Ok(event)).await;
                    break;
                }
            }
        }

        // Send citations as a final event
        if !citations.is_empty() {
            let citations_json = serde_json::to_string(&citations).unwrap_or_default();
            let event = Event::default()
                .event("citations")
                .data(citations_json);
            let _ = tx.send(Ok(event)).await;
        }

        // Send done event
        let done_data = serde_json::json!({
            "conversation_id": conv_id,
            "blocked": blocked,
        });
        let event = Event::default()
            .event("done")
            .data(done_data.to_string());
        let _ = tx.send(Ok(event)).await;

        // Save assistant message
        let assistant_msg = Message {
            role: Role::Assistant,
            content: full_response,
            timestamp: now_iso(),
            citations,
        };
        let _ = state.conversations.append_message(&conv_id, assistant_msg);
    });

    tokio_stream::wrappers::ReceiverStream::new(rx)
}
