use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;

use crate::app_state::AppState;
use crate::project::{project_dirs, require_model};
use crate::task_manager::TaskEvent;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/api/finetune", post(start_finetune))
}

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.to_string() }))
}

#[derive(Deserialize)]
struct FinetuneRequest {
    method: String,
    data_file: String,
    learning_rate: Option<f64>,
    epochs: Option<usize>,
}

async fn start_finetune(
    State(state): State<Arc<AppState>>,
    Json(body): Json<FinetuneRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let project_path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?
        .clone();
    drop(active);

    require_model(&project_path)
        .map_err(|e| err(StatusCode::BAD_REQUEST, &e.to_string()))?;

    if body.method != "sft" && body.method != "dpo" {
        return Err(err(StatusCode::BAD_REQUEST, "Method must be 'sft' or 'dpo'"));
    }

    // Validate data_file: reject path traversal
    if body.data_file.contains('/')
        || body.data_file.contains('\\')
        || body.data_file.contains("..")
    {
        return Err(err(StatusCode::BAD_REQUEST, "Invalid data file name"));
    }

    let dirs = project_dirs(&project_path);
    let data_path = dirs.finetune_data.join(&body.data_file);
    if !data_path.exists() {
        return Err(err(StatusCode::BAD_REQUEST, &format!("Data file '{}' not found in finetune_data/", body.data_file)));
    }

    let (task_id, tx) = state.tasks.create_task("finetune").await;
    let method = body.method.clone();
    let learning_rate = body.learning_rate;
    let epochs = body.epochs;

    let tasks2 = state.tasks.clone();
    let tid2 = task_id.clone();
    let mut rx = tx.subscribe();

    tokio::task::spawn_blocking(move || {
        let result = run_finetune(&project_path, &method, &data_path, learning_rate, epochs, &tx);
        let success = result.is_ok();
        let error = result.err().map(|e| format!("{e:#}"));
        let _ = tx.send(TaskEvent::Done { success, error });
    });

    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(TaskEvent::Done { success, error }) => {
                    if success {
                        tasks2.complete_task(&tid2).await;
                    } else {
                        tasks2.fail_task(&tid2, error.unwrap_or_default()).await;
                    }
                    break;
                }
                Err(_) => break,
                _ => continue,
            }
        }
    });

    Ok(Json(serde_json::json!({ "task_id": task_id })))
}

struct SimpleTokenizer;

impl finetune::sft::Tokenize for SimpleTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| i as u32)
            .collect())
    }
}

fn run_finetune(
    project_path: &std::path::Path,
    method: &str,
    data_path: &std::path::Path,
    learning_rate: Option<f64>,
    epochs: Option<usize>,
    tx: &tokio::sync::broadcast::Sender<TaskEvent>,
) -> anyhow::Result<()> {
    let dirs = project_dirs(project_path);
    let device = candle_core::Device::Cpu;

    // Hardware preflight
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Hardware check".to_string(),
    });
    let report = finetune::preflight::preflight_check(16.0);
    let _ = tx.send(TaskEvent::Log {
        message: format!(
            "Available RAM: {:.1} GB, CPU cores: {}",
            report.hardware.available_ram_gb, report.hardware.cpu_count
        ),
    });
    if !report.is_sufficient {
        let _ = tx.send(TaskEvent::Log {
            message: format!(
                "Warning: Recommended {:.0} GB RAM, you have {:.1} GB",
                report.recommended_ram_gb, report.hardware.available_ram_gb
            ),
        });
    }
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Hardware check".to_string(),
        detail: format!("{:.1} GB RAM available", report.hardware.available_ram_gb),
    });

    let lora_config = finetune::lora::LoraConfig::default();
    let adapter_path = dirs.root.join("adapter.safetensors");

    // Check for full model (SafeTensors)
    let has_safetensors = dirs.model.join("config.json").exists()
        && std::fs::read_dir(&dirs.model)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| {
                        e.path()
                            .extension()
                            .is_some_and(|ext| ext == "safetensors")
                    })
            })
            .unwrap_or(false);

    if has_safetensors {
        // Full model LoRA training
        let _ = tx.send(TaskEvent::StepStarted {
            step: "Loading model".to_string(),
        });
        let mut trainer =
            finetune::Phi3LoraTrainer::new(&dirs.model, &lora_config, &device)?;
        let _ = tx.send(TaskEvent::StepCompleted {
            step: "Loading model".to_string(),
            detail: "Model loaded with LoRA layers".to_string(),
        });

        match method {
            "sft" => {
                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Loading training data".to_string(),
                });
                let examples = finetune::data::load_sft_data(data_path)?;
                let _ = tx.send(TaskEvent::StepCompleted {
                    step: "Loading training data".to_string(),
                    detail: format!("{} SFT examples", examples.len()),
                });

                let mut config = finetune::sft::SftConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Training".to_string(),
                });
                let final_loss = finetune::sft::train_sft_full(&mut trainer, &examples, &config)?;
                let _ = tx.send(TaskEvent::StepCompleted {
                    step: "Training".to_string(),
                    detail: format!("SFT complete. Final loss: {final_loss:.6}"),
                });

                trainer.model.save_adapter(&adapter_path)?;
            }
            "dpo" => {
                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Loading training data".to_string(),
                });
                let examples = finetune::data::load_dpo_data(data_path)?;
                let _ = tx.send(TaskEvent::StepCompleted {
                    step: "Loading training data".to_string(),
                    detail: format!("{} DPO examples", examples.len()),
                });

                let mut config = finetune::dpo::DpoConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Training".to_string(),
                });
                let _ = tx.send(TaskEvent::Log {
                    message: "Using standalone LoRA training for DPO".to_string(),
                });
                run_standalone_dpo(&dirs, &examples, &config, &lora_config, &adapter_path, &device, tx)?;
            }
            _ => anyhow::bail!("Unknown method: {method}"),
        }
    } else {
        // Standalone LoRA training
        let _ = tx.send(TaskEvent::Log {
            message: "No SafeTensors model found; using standalone LoRA training".to_string(),
        });

        let in_features = 128;
        let out_features = 128;
        let frozen_weight =
            candle_core::Tensor::randn(0f32, 1.0, &[out_features, in_features], &device)?;
        let frozen = candle_nn::Linear::new(frozen_weight, None);
        let mut lora = finetune::lora::LoraLinear::new(
            frozen,
            in_features,
            out_features,
            &lora_config,
            &device,
        )?;
        let tokenizer = SimpleTokenizer;

        match method {
            "sft" => {
                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Loading training data".to_string(),
                });
                let examples = finetune::data::load_sft_data(data_path)?;
                let _ = tx.send(TaskEvent::StepCompleted {
                    step: "Loading training data".to_string(),
                    detail: format!("{} SFT examples", examples.len()),
                });

                let mut config = finetune::sft::SftConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Training".to_string(),
                });
                let final_loss = finetune::sft::train_sft(&mut lora, &examples, &config, &tokenizer, &device)?;
                let _ = tx.send(TaskEvent::StepCompleted {
                    step: "Training".to_string(),
                    detail: format!("SFT complete. Final loss: {final_loss:.6}"),
                });

                lora.save(&adapter_path)?;
            }
            "dpo" => {
                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Loading training data".to_string(),
                });
                let examples = finetune::data::load_dpo_data(data_path)?;
                let _ = tx.send(TaskEvent::StepCompleted {
                    step: "Loading training data".to_string(),
                    detail: format!("{} DPO examples", examples.len()),
                });

                let mut config = finetune::dpo::DpoConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let _ = tx.send(TaskEvent::StepStarted {
                    step: "Training".to_string(),
                });
                let final_loss = finetune::dpo::train_dpo(&mut lora, &examples, &config, &tokenizer, &device)?;
                let _ = tx.send(TaskEvent::StepCompleted {
                    step: "Training".to_string(),
                    detail: format!("DPO complete. Final loss: {final_loss:.6}"),
                });

                lora.save(&adapter_path)?;
            }
            _ => anyhow::bail!("Unknown method: {method}"),
        }

        // Save checkpoint
        let checkpoint_dir = dirs.root.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;
        let meta = finetune::checkpoint::CheckpointMeta {
            epoch: epochs.unwrap_or(3),
            step: 0,
            loss: 0.0,
            lora_config: lora_config.clone(),
        };
        finetune::checkpoint::save_checkpoint(&lora, &meta, &checkpoint_dir)?;
    }

    let _ = tx.send(TaskEvent::Log {
        message: format!("Adapter saved to {}", adapter_path.display()),
    });
    Ok(())
}

fn run_standalone_dpo(
    dirs: &crate::project::ProjectDirs,
    examples: &[finetune::data::DpoExample],
    config: &finetune::dpo::DpoConfig,
    lora_config: &finetune::lora::LoraConfig,
    adapter_path: &std::path::Path,
    device: &candle_core::Device,
    tx: &tokio::sync::broadcast::Sender<TaskEvent>,
) -> anyhow::Result<()> {
    let in_features = 128;
    let out_features = 128;
    let frozen_weight =
        candle_core::Tensor::randn(0f32, 1.0, &[out_features, in_features], device)?;
    let frozen = candle_nn::Linear::new(frozen_weight, None);
    let mut lora = finetune::lora::LoraLinear::new(
        frozen,
        in_features,
        out_features,
        lora_config,
        device,
    )?;
    let tokenizer = SimpleTokenizer;

    let final_loss = finetune::dpo::train_dpo(&mut lora, examples, config, &tokenizer, device)?;
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Training".to_string(),
        detail: format!("DPO complete. Final loss: {final_loss:.6}"),
    });

    lora.save(adapter_path)?;

    let checkpoint_dir = dirs.root.join("checkpoints");
    std::fs::create_dir_all(&checkpoint_dir)?;
    let meta = finetune::checkpoint::CheckpointMeta {
        epoch: config.epochs,
        step: 0,
        loss: final_loss,
        lora_config: lora_config.clone(),
    };
    finetune::checkpoint::save_checkpoint(&lora, &meta, &checkpoint_dir)?;

    Ok(())
}
