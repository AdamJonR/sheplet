use std::path::PathBuf;
use std::sync::Arc;

use conversations::ConversationStore;
use tokio::sync::RwLock;

use crate::course::CourseManager;

pub struct AppState {
    pub courses: RwLock<CourseManager>,
    pub conversations: Arc<ConversationStore>,
    pub base_dir: PathBuf,
    /// Debug flag: skip loading LoRA adapter to test base model in isolation.
    pub no_adapter: bool,
}
