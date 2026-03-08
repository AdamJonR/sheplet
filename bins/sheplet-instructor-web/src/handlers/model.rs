use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;

use crate::app_state::AppState;
use crate::project::{project_dirs, require_init, ProjectManifest};
use crate::response::{err, ErrorResponse};
use crate::task_manager::TaskEvent;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/api/model/download", post(start_model_download))
}

#[derive(Deserialize)]
struct ModelRequest {
    name: Option<String>,
    quantization: Option<String>,
}

async fn start_model_download(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ModelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let project_path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?
        .clone();
    drop(active);

    require_init(&project_path)
        .map_err(|e| err(StatusCode::BAD_REQUEST, &e.to_string()))?;

    let name = body.name.unwrap_or_else(|| "phi-4-mini-instruct".to_string());
    let quantization = body.quantization.unwrap_or_else(|| "q4-k-m".to_string());

    let (task_id, tx) = state.tasks.create_task("model_download").await;
    let rx = tx.subscribe();

    tokio::task::spawn_blocking(move || {
        let result = run_model_download(&project_path, &name, &quantization, &tx);
        let success = result.is_ok();
        let error = result.err().map(|e| format!("{e:#}"));
        let _ = tx.send(TaskEvent::Done { success, error });
    });

    super::spawn_task_listener(state.tasks.clone(), task_id.clone(), rx);

    Ok(Json(serde_json::json!({ "task_id": task_id })))
}

fn run_model_download(
    project_path: &std::path::Path,
    name: &str,
    quantization: &str,
    tx: &tokio::sync::broadcast::Sender<TaskEvent>,
) -> anyhow::Result<()> {
    let mut manifest = ProjectManifest::load(project_path)?;
    let dirs = project_dirs(project_path);

    let repo_id = match name {
        "phi-4-mini-instruct" => "microsoft/Phi-4-mini-instruct",
        "gemma-3-1b-it" => "google/gemma-3-1b-it",
        other => other,
    };

    // Step 1: Download model files
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Downloading model".to_string(),
    });
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.model(repo_id.to_string());
    let model_dir = &dirs.model;
    std::fs::create_dir_all(model_dir)?;

    for filename in &["config.json", "tokenizer.json", "tokenizer_config.json"] {
        match repo.get(filename) {
            Ok(src_path) => {
                let dest = model_dir.join(filename);
                if src_path != dest {
                    std::fs::copy(&src_path, &dest)?;
                }
            }
            Err(e) => {
                let _ = tx.send(TaskEvent::Log {
                    message: format!("Warning: could not download {filename}: {e}"),
                });
            }
        }
    }

    // Download model weights
    match repo.get("model.safetensors") {
        Ok(src_path) => {
            let dest = model_dir.join("model.safetensors");
            if src_path != dest {
                std::fs::copy(&src_path, &dest)?;
            }
        }
        Err(_) => {
            match repo.get("model.safetensors.index.json") {
                Ok(index_path) => {
                    let dest = model_dir.join("model.safetensors.index.json");
                    if index_path != dest {
                        std::fs::copy(&index_path, &dest)?;
                    }
                    let index_content = std::fs::read_to_string(&index_path)?;
                    let index: serde_json::Value = serde_json::from_str(&index_content)?;
                    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                        let shard_files: std::collections::HashSet<&str> =
                            weight_map.values().filter_map(|v| v.as_str()).collect();
                        for shard in shard_files {
                            match repo.get(shard) {
                                Ok(src_path) => {
                                    let dest = model_dir.join(shard);
                                    if src_path != dest {
                                        std::fs::copy(&src_path, &dest)?;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(TaskEvent::Log {
                                        message: format!("Warning: could not download shard {shard}: {e}"),
                                    });
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(TaskEvent::Log {
                        message: format!("Warning: could not find model weights: {e}"),
                    });
                }
            }
        }
    }
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Downloading model".to_string(),
        detail: format!("Model {repo_id} downloaded"),
    });

    // Step 2: Download embedding model
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Downloading embedding model".to_string(),
    });
    let _embedding_model = embeddings::EmbeddingModel::download_and_load(&dirs.embeddings)?;
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Downloading embedding model".to_string(),
        detail: "Embedding model ready".to_string(),
    });

    // Step 3: Quantize (skip for "none")
    if quantization != "none" {
        let _ = tx.send(TaskEvent::StepStarted {
            step: "Quantizing model".to_string(),
        });
        let gguf_path = model_dir.join("model.gguf");
        let progress_cb = |current: usize, total: usize| {
            let _ = tx.send(TaskEvent::Progress {
                step: "Quantizing model".to_string(),
                current: current as u64,
                total: total as u64,
            });
        };
        rag::quantize_safetensors_to_gguf(model_dir, &gguf_path, quantization, Some(&progress_cb))?;
        let _ = tx.send(TaskEvent::StepCompleted {
            step: "Quantizing model".to_string(),
            detail: format!("Quantized to {quantization}"),
        });

        // Clean up SafeTensors files
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                std::fs::remove_file(&path)?;
            }
            if path
                .file_name()
                .is_some_and(|n| n == "model.safetensors.index.json")
            {
                std::fs::remove_file(&path)?;
            }
        }
    } else {
        let _ = tx.send(TaskEvent::StepCompleted {
            step: "Quantizing model".to_string(),
            detail: "Skipped (full precision)".to_string(),
        });
    }

    // Update manifest
    manifest.model_name = Some(name.to_string());
    manifest.quantization = Some(quantization.to_string());
    manifest.save(&dirs.root)?;

    Ok(())
}
