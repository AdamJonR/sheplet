use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::app_state::AppState;
use crate::project::{self, CourseConfig, ProjectManifest};
use crate::response::{err, ErrorResponse};
use crate::validation::validate_safe_name;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/projects", get(list_projects))
        .route("/api/projects", post(create_project))
        .route("/api/projects/select", post(select_project))
        .route("/api/projects/active", get(active_project))
}

#[derive(Serialize)]
struct ProjectInfo {
    name: String,
    course_name: String,
    version: String,
    is_active: bool,
    status: ProjectStatusInfo,
}

#[derive(Serialize)]
struct ProjectStatusInfo {
    has_config: bool,
    has_model: bool,
    has_database: bool,
    has_embeddings: bool,
    has_adapter: bool,
    has_finetune_data: bool,
    model_name: Option<String>,
    finetune_files: Vec<String>,
    build_timestamp: Option<String>,
}

async fn list_projects(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ProjectInfo>>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let active_path = active.as_ref();

    let entries = std::fs::read_dir(&state.base_dir).map_err(|e| {
        err(StatusCode::INTERNAL_SERVER_ERROR, &format!("Failed to read projects directory: {e}"))
    })?;

    let mut projects = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if let Ok(manifest) = ProjectManifest::load(&path) {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let is_active = active_path.is_some_and(|ap| ap == &path);
            let status = build_status(&path, &manifest);
            projects.push(ProjectInfo {
                name,
                course_name: manifest.course_name,
                version: manifest.version,
                is_active,
                status,
            });
        }
    }

    Ok(Json(projects))
}

fn build_status(path: &std::path::Path, manifest: &ProjectManifest) -> ProjectStatusInfo {
    let dirs = project::project_dirs(path);
    let has_model = dirs.model.join("config.json").exists()
        && std::fs::read_dir(&dirs.model)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            })
            .unwrap_or(false);
    let has_database = dirs.database.exists()
        && std::fs::read_dir(&dirs.database)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);
    let has_embeddings = dirs.embeddings.exists()
        && std::fs::read_dir(&dirs.embeddings)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);
    let has_adapter = path.join("adapter.safetensors").exists();
    let has_config = path.join("config.json").exists();

    let mut finetune_files = Vec::new();
    let has_finetune_data = if dirs.finetune_data.exists() {
        if let Ok(entries) = std::fs::read_dir(&dirs.finetune_data) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.ends_with(".jsonl") {
                    finetune_files.push(name);
                }
            }
        }
        !finetune_files.is_empty()
    } else {
        false
    };

    ProjectStatusInfo {
        has_config,
        has_model,
        has_database,
        has_embeddings,
        has_adapter,
        has_finetune_data,
        model_name: manifest.model_name.clone(),
        finetune_files,
        build_timestamp: manifest.build_timestamp.clone(),
    }
}

#[derive(Deserialize)]
struct CreateProject {
    course_name: String,
    directory_name: String,
}

async fn create_project(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CreateProject>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // Validate directory name: reject path separators and traversal
    validate_safe_name(&body.directory_name)
        .map_err(|e| err(StatusCode::BAD_REQUEST, &e.to_string()))?;

    let project_path = state.base_dir.join(&body.directory_name);
    if project_path.join("manifest.json").exists() {
        return Err(err(StatusCode::CONFLICT, "Project already exists"));
    }

    let dirs = project::project_dirs(&project_path);
    std::fs::create_dir_all(&dirs.root)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    std::fs::create_dir_all(&dirs.model)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    std::fs::create_dir_all(&dirs.embeddings)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    std::fs::create_dir_all(&dirs.database)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    std::fs::create_dir_all(&dirs.finetune_data)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    let manifest = ProjectManifest::new(&body.course_name);
    manifest
        .save(&dirs.root)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    let config = CourseConfig::default();
    config
        .save(&dirs.root)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    // Ensure keypair exists
    let fingerprint = if let Some(keypair_path) = bundle::keys::Keypair::default_keypair_path() {
        bundle::keys::Keypair::load_or_create(&keypair_path)
            .ok()
            .map(|kp| kp.fingerprint())
    } else {
        None
    };

    // Auto-select the new project
    *state.active_project.write().await = Some(project_path);

    Ok(Json(serde_json::json!({
        "message": format!("Project '{}' created", body.course_name),
        "name": body.directory_name,
        "fingerprint": fingerprint,
    })))
}

#[derive(Deserialize)]
struct SelectProject {
    name: String,
}

async fn select_project(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SelectProject>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    // Validate name: reject path traversal
    validate_safe_name(&body.name)
        .map_err(|_| err(StatusCode::BAD_REQUEST, "Invalid project name"))?;

    let project_path = state.base_dir.join(&body.name);
    if !project_path.join("manifest.json").exists() {
        return Err(err(StatusCode::NOT_FOUND, "Project not found"));
    }

    *state.active_project.write().await = Some(project_path);

    Ok(Json(serde_json::json!({
        "message": format!("Switched to project '{}'", body.name),
    })))
}

async fn active_project(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    match active.as_ref() {
        Some(path) => {
            let manifest = ProjectManifest::load(path)
                .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let status = build_status(path, &manifest);
            Ok(Json(serde_json::json!({
                "project": {
                    "name": name,
                    "course_name": manifest.course_name,
                    "version": manifest.version,
                    "is_active": true,
                    "status": status,
                }
            })))
        }
        None => Ok(Json(serde_json::json!({ "project": null }))),
    }
}
