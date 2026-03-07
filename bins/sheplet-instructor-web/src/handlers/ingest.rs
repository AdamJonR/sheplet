use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;

use crate::app_state::AppState;
use crate::project::{project_dirs, require_init};
use crate::task_manager::TaskEvent;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/api/ingest", post(start_ingest))
}

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.to_string() }))
}

#[derive(Deserialize)]
struct IngestRequest {
    sources_path: String,
}

async fn start_ingest(
    State(state): State<Arc<AppState>>,
    Json(body): Json<IngestRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let project_path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?
        .clone();
    drop(active);

    require_init(&project_path)
        .map_err(|e| err(StatusCode::BAD_REQUEST, &e.to_string()))?;

    let sources = std::path::PathBuf::from(&body.sources_path);
    if !sources.exists() {
        return Err(err(StatusCode::BAD_REQUEST, "Sources directory does not exist"));
    }

    let (task_id, tx) = state.tasks.create_task("ingest").await;
    let tasks = state.tasks.clone();
    let tid = task_id.clone();

    tokio::spawn(async move {
        let result = run_ingest(&project_path, &sources, &tx).await;
        match result {
            Ok(()) => {
                let _ = tx.send(TaskEvent::Done {
                    success: true,
                    error: None,
                });
                tasks.complete_task(&tid).await;
            }
            Err(e) => {
                let msg = format!("{e:#}");
                let _ = tx.send(TaskEvent::Done {
                    success: false,
                    error: Some(msg.clone()),
                });
                tasks.fail_task(&tid, msg).await;
            }
        }
    });

    Ok(Json(serde_json::json!({ "task_id": task_id })))
}

async fn run_ingest(
    project_path: &std::path::Path,
    sources: &std::path::Path,
    tx: &tokio::sync::broadcast::Sender<TaskEvent>,
) -> anyhow::Result<()> {
    let dirs = project_dirs(project_path);

    // Step 1: Parse documents
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Parsing documents".to_string(),
    });
    let sources_owned = sources.to_path_buf();
    let (chunks, warnings) = tokio::task::spawn_blocking(move || {
        let chunk_config = parser::ChunkConfig::default();
        parser::parse_directory(&sources_owned, &chunk_config)
    })
    .await??;

    for warning in &warnings {
        let _ = tx.send(TaskEvent::Log {
            message: format!("Warning: {warning}"),
        });
    }
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Parsing documents".to_string(),
        detail: format!("{} chunks extracted", chunks.len()),
    });

    if chunks.is_empty() {
        let _ = tx.send(TaskEvent::Log {
            message: "No chunks extracted. Nothing to ingest.".to_string(),
        });
        return Ok(());
    }

    // Step 2: Load embedding model
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Loading embedding model".to_string(),
    });
    let emb_dir = dirs.embeddings.clone();
    let embedding_model = tokio::task::spawn_blocking(move || {
        embeddings::EmbeddingModel::download_and_load(&emb_dir)
    })
    .await??;
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Loading embedding model".to_string(),
        detail: "Model ready".to_string(),
    });

    // Step 3: Embed chunks
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Embedding chunks".to_string(),
    });
    let total = chunks.len() as u64;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vectors = tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        embedding_model.embed_batch(&refs)
    })
    .await??;
    let _ = tx.send(TaskEvent::Progress {
        step: "Embedding chunks".to_string(),
        current: total,
        total,
    });
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Embedding chunks".to_string(),
        detail: format!("{total} chunks embedded"),
    });

    // Step 4: Store in vector database
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Storing in database".to_string(),
    });
    let store = db::VectorStore::open_or_create(
        &dirs.database,
        "chunks",
        embeddings::EMBEDDING_DIM,
    )
    .await?;

    let records: Vec<db::ChunkRecord> = chunks
        .iter()
        .zip(vectors.iter())
        .map(|(chunk, vector)| db::ChunkRecord {
            vector: vector.clone(),
            text: chunk.text.clone(),
            source_file: chunk.source.file_path.clone(),
            chunk_index: chunk.source.chunk_index as u32,
        })
        .collect();

    store.insert(&records).await?;
    let count = store.count().await?;
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Storing in database".to_string(),
        detail: format!("{count} total chunks in database"),
    });

    Ok(())
}
