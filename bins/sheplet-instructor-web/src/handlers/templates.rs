use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::app_state::AppState;
use crate::project::project_dirs;
use crate::response::{err, ErrorResponse};

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/templates/generate", post(generate_templates))
        .route("/api/templates/files", get(list_files))
}

const DPO_TEMPLATE: &str = r#"{"prompt": "When was Rome traditionally founded?", "chosen": "Rome was traditionally founded in 753 BC, a date calculated by the Roman scholar Varro. According to legend, the city was established by Romulus, who became its first king, though archaeology shows settlements existed on the site from at least the 8th century BC.", "rejected": "I think Rome was founded sometime in the Middle Ages, maybe by some merchants or something."}
{"prompt": "What does SPQR stand for?", "chosen": "SPQR stands for Senatus Populusque Romanus, meaning the Senate and the People of Rome. It appeared on official documents, monuments, and standards, symbolizing the shared authority of the governing council and the citizens.", "rejected": "It's some kind of Roman abbreviation."}
"#;

const SFT_TEMPLATE: &str = r#"{"input": "When was Rome traditionally founded?", "output": "Rome was traditionally founded in 753 BC, a date calculated by the Roman scholar Varro. According to legend, the city was established by Romulus, who became its first king, though archaeology shows settlements existed on the site from at least the 8th century BC."}
{"input": "What does SPQR stand for?", "output": "SPQR stands for Senatus Populusque Romanus, meaning the Senate and the People of Rome. It appeared on official documents, monuments, and standards, symbolizing the shared authority of the governing council and the citizens."}
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
    if dirs.finetune_data.exists()
        && let Ok(entries) = std::fs::read_dir(&dirs.finetune_data) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".jsonl") {
                    files.push(name);
                }
            }
        }
    Ok(Json(files))
}
