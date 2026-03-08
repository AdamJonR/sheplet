pub mod bundle;
pub mod config;
pub mod finetune;
pub mod frontend;
pub mod ingest;
pub mod model;
pub mod projects;
pub mod tasks;
pub mod templates;

use std::sync::Arc;

use crate::task_manager::{TaskEvent, TaskManager};

/// Spawn a background listener that completes/fails a task based on the Done event.
pub fn spawn_task_listener(
    tasks: Arc<TaskManager>,
    task_id: String,
    mut rx: tokio::sync::broadcast::Receiver<TaskEvent>,
) {
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(TaskEvent::Done { success, error }) => {
                    if success {
                        tasks.complete_task(&task_id).await;
                    } else {
                        tasks.fail_task(&task_id, error.unwrap_or_default()).await;
                    }
                    break;
                }
                Err(_) => break,
                _ => continue,
            }
        }
    });
}
