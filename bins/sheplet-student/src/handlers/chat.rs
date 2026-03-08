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

    // Embed query off the async runtime
    let embed_pipeline = active.pipeline.clone();
    let embed_message = req.message.clone();
    let query_vec = tokio::task::spawn_blocking(move || {
        embed_pipeline.blocking_read().embed_query(&embed_message)
    })
    .await
    .unwrap()
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Search and assemble prompt
    let pipeline = active.pipeline.read().await;
    let history = &conv.messages[..conv.messages.len().saturating_sub(1)];
    let results = pipeline.search_chunks(&query_vec).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    let prepared = pipeline.assemble_from_results(&results, history, &req.message);
    drop(pipeline);

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

    // Clone what we need, then drop the read lock so SSE can start immediately
    let pipeline = active.pipeline.clone();
    let generator = active.generator.clone();
    let message = req.message.clone();
    let history: Vec<Message> = conv.messages[..conv.messages.len().saturating_sub(1)].to_vec();
    drop(courses);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(100);

    tokio::spawn(async move {
        // Status: embedding
        let _ = tx.send(Ok(Event::default().event("status").data("embedding"))).await;

        let embed_pipeline = pipeline.clone();
        let embed_message = message.clone();
        let query_vec = match tokio::task::spawn_blocking(move || {
            embed_pipeline.blocking_read().embed_query(&embed_message)
        }).await.unwrap() {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await;
                let done_data = serde_json::json!({
                    "conversation_id": conv_id,
                    "blocked": false,
                });
                let _ = tx.send(Ok(Event::default().event("done").data(done_data.to_string()))).await;
                return;
            }
        };
        let pipeline_guard = pipeline.read().await;

        // Status: searching
        let _ = tx.send(Ok(Event::default().event("status").data("searching"))).await;

        let results = match pipeline_guard.search_chunks(&query_vec).await {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await;
                let done_data = serde_json::json!({
                    "conversation_id": conv_id,
                    "blocked": false,
                });
                let _ = tx.send(Ok(Event::default().event("done").data(done_data.to_string()))).await;
                return;
            }
        };

        let prepared = pipeline_guard.assemble_from_results(&results, &history, &message);
        drop(pipeline_guard);

        match prepared {
            PreparedQuery::Blocked { message: blocked_msg } => {
                let assistant_msg = Message {
                    role: Role::Assistant,
                    content: blocked_msg.clone(),
                    timestamp: now_iso(),
                    citations: vec![],
                };
                let _ = state.conversations.append_message(&conv_id, assistant_msg);

                let _ = tx.send(Ok(Event::default().event("status").data("generating"))).await;
                let _ = tx.send(Ok(Event::default().data(&blocked_msg))).await;
                let done_data = serde_json::json!({
                    "conversation_id": conv_id,
                    "blocked": true,
                });
                let _ = tx.send(Ok(Event::default().event("done").data(done_data.to_string()))).await;
            }
            PreparedQuery::Ready { prompt, citations } => {
                // Status: generating
                let _ = tx.send(Ok(Event::default().event("status").data("generating"))).await;

                // Send citations early
                if !citations.is_empty() {
                    let citations_json = serde_json::to_string(&citations).unwrap_or_default();
                    let _ = tx.send(Ok(Event::default().event("citations").data(citations_json))).await;
                }

                // Generate tokens
                let (token_tx, token_rx) = std::sync::mpsc::channel::<rag::Result<String>>();
                let gen_clone = generator.clone();
                tokio::task::spawn_blocking(move || {
                    let mut locked = gen_clone.lock().unwrap();
                    let _ = locked.generate_to_sender(&prompt, 512, token_tx);
                });

                let mut full_response = String::new();
                while let Ok(result) = token_rx.recv() {
                    match result {
                        Ok(token) => {
                            full_response.push_str(&token);
                            if tx.send(Ok(Event::default().data(&token))).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Ok(Event::default().event("error").data(e.to_string()))).await;
                            break;
                        }
                    }
                }

                // Done event
                let done_data = serde_json::json!({
                    "conversation_id": conv_id,
                    "blocked": false,
                });
                let _ = tx.send(Ok(Event::default().event("done").data(done_data.to_string()))).await;

                // Save assistant message
                let assistant_msg = Message {
                    role: Role::Assistant,
                    content: full_response,
                    timestamp: now_iso(),
                    citations,
                };
                let _ = state.conversations.append_message(&conv_id, assistant_msg);
            }
        }
    });

    Ok(Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
}
