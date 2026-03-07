use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::task_manager::TaskManager;

pub struct AppState {
    pub base_dir: PathBuf,
    pub active_project: RwLock<Option<PathBuf>>,
    pub tasks: Arc<TaskManager>,
}
