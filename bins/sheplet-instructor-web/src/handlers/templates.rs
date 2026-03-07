use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::app_state::AppState;
use crate::project::project_dirs;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/templates/generate", post(generate_templates))
        .route("/api/templates/files", get(list_files))
}

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.to_string() }))
}

const DPO_TEMPLATE: &str = r#"{"prompt": "What is photosynthesis?", "chosen": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.", "rejected": "I think it has something to do with plants and light, maybe they eat the sun or something."}
{"prompt": "Explain the water cycle.", "chosen": "The water cycle describes the continuous movement of water within the Earth and atmosphere. It involves evaporation from surface water, transpiration from plants, condensation into clouds, and precipitation back to the surface as rain or snow.", "rejected": "Water goes up and comes back down."}
"#;

const SFT_TEMPLATE: &str = r#"{"input": "What is photosynthesis?", "output": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct."}
{"input": "Explain the water cycle.", "output": "The water cycle describes the continuous movement of water within the Earth and atmosphere. It involves evaporation from surface water, transpiration from plants, condensation into clouds, and precipitation back to the surface as rain or snow."}
"#;

async fn generate_templates(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?;

    let dirs = project_dirs(path);
    std::fs::create_dir_all(&dirs.finetune_data)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    let dpo_path = dirs.finetune_data.join("dpo_template.jsonl");
    let sft_path = dirs.finetune_data.join("sft_template.jsonl");

    std::fs::write(&dpo_path, DPO_TEMPLATE)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    std::fs::write(&sft_path, SFT_TEMPLATE)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    Ok(Json(serde_json::json!({
        "message": "Template files generated",
        "files": ["dpo_template.jsonl", "sft_template.jsonl"],
    })))
}

async fn list_files(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?;

    let dirs = project_dirs(path);
    let mut files = Vec::new();
    if dirs.finetune_data.exists() {
        if let Ok(entries) = std::fs::read_dir(&dirs.finetune_data) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".jsonl") {
                    files.push(name);
                }
            }
        }
    }
    Ok(Json(files))
}
