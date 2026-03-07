use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;

use crate::app_state::AppState;
use crate::project::{require_bundleable, ProjectManifest};
use crate::task_manager::TaskEvent;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/api/bundle", post(start_bundle))
}

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: msg.to_string() }))
}

#[derive(Deserialize)]
struct BundleRequest {
    output_path: String,
    bump_version: Option<bool>,
}

async fn start_bundle(
    State(state): State<Arc<AppState>>,
    Json(body): Json<BundleRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let active = state.active_project.read().await;
    let project_path = active
        .as_ref()
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "No active project"))?
        .clone();
    drop(active);

    require_bundleable(&project_path)
        .map_err(|e| err(StatusCode::BAD_REQUEST, &e.to_string()))?;

    let output = std::path::PathBuf::from(&body.output_path);
    let bump = body.bump_version.unwrap_or(false);

    let (task_id, tx) = state.tasks.create_task("bundle").await;

    let tasks2 = state.tasks.clone();
    let tid2 = task_id.clone();
    let mut rx = tx.subscribe();

    tokio::task::spawn_blocking(move || {
        let result = run_bundle(&project_path, &output, bump, &tx);
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

fn run_bundle(
    project_path: &std::path::Path,
    output: &std::path::Path,
    bump: bool,
    tx: &tokio::sync::broadcast::Sender<TaskEvent>,
) -> anyhow::Result<()> {
    let mut manifest = ProjectManifest::load(project_path)?;

    if bump {
        manifest.bump_version();
        manifest.save(project_path)?;
        let _ = tx.send(TaskEvent::Log {
            message: format!("Version bumped to {}", manifest.version),
        });
    }

    // Update build timestamp
    manifest.build_timestamp = Some(timestamp());
    manifest.save(project_path)?;

    // Load keypair
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Loading keypair".to_string(),
    });
    let keypair_path = bundle::keys::Keypair::default_keypair_path()
        .ok_or_else(|| anyhow::anyhow!("Could not determine keypair path"))?;
    let keypair = bundle::keys::Keypair::load_or_create(&keypair_path)?;
    let pub_key_hex = hex::encode(keypair.public_key_bytes());
    let fingerprint = keypair.fingerprint();
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Loading keypair".to_string(),
        detail: format!("Fingerprint: {fingerprint}"),
    });

    // Write bundle manifest
    let _ = tx.send(TaskEvent::StepStarted {
        step: "Packaging bundle".to_string(),
    });
    let bundle_manifest = bundle::manifest::Manifest {
        version: manifest.version.clone(),
        course_name: manifest.course_name.clone(),
        model_name: manifest.model_name.clone().unwrap_or_default(),
        quantization: manifest.quantization.clone().unwrap_or_default(),
        build_timestamp: manifest.build_timestamp.clone().unwrap_or_default(),
        public_key_hex: pub_key_hex,
        public_key_fingerprint: fingerprint.clone(),
    };

    let manifest_content = serde_json::to_string_pretty(&bundle_manifest)?;
    std::fs::write(project_path.join("manifest.json"), &manifest_content)?;

    bundle::pack::pack(project_path, output, &keypair)?;
    let _ = tx.send(TaskEvent::StepCompleted {
        step: "Packaging bundle".to_string(),
        detail: format!("Bundle created at {}", output.display()),
    });

    let _ = tx.send(TaskEvent::Log {
        message: format!(
            "Course: {}, Version: {}, Fingerprint: {}",
            manifest.course_name, manifest.version, fingerprint
        ),
    });

    Ok(())
}

fn timestamp() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}
