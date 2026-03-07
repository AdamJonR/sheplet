use serde::Serialize;
use std::collections::HashMap;
use std::time::SystemTime;
use tokio::sync::{broadcast, RwLock};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum TaskEvent {
    #[serde(rename = "step_started")]
    StepStarted { step: String },
    #[serde(rename = "step_completed")]
    StepCompleted { step: String, detail: String },
    #[serde(rename = "progress")]
    Progress {
        step: String,
        current: u64,
        total: u64,
    },
    #[serde(rename = "log")]
    Log { message: String },
    #[serde(rename = "done")]
    Done {
        success: bool,
        error: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum TaskStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskInfo {
    pub id: String,
    pub kind: String,
    pub status: TaskStatus,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub error: Option<String>,
}

struct TaskEntry {
    info: TaskInfo,
    sender: broadcast::Sender<TaskEvent>,
}

pub struct TaskManager {
    tasks: RwLock<HashMap<String, TaskEntry>>,
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
        }
    }

    pub async fn create_task(&self, kind: &str) -> (String, broadcast::Sender<TaskEvent>) {
        let id = format!("{:016x}", rand_id());
        let (tx, _) = broadcast::channel(256);
        let entry = TaskEntry {
            info: TaskInfo {
                id: id.clone(),
                kind: kind.to_string(),
                status: TaskStatus::Running,
                started_at: timestamp(),
                finished_at: None,
                error: None,
            },
            sender: tx.clone(),
        };
        self.tasks.write().await.insert(id.clone(), entry);
        (id, tx)
    }

    pub async fn complete_task(&self, id: &str) {
        if let Some(entry) = self.tasks.write().await.get_mut(id) {
            entry.info.status = TaskStatus::Completed;
            entry.info.finished_at = Some(timestamp());
        }
    }

    pub async fn fail_task(&self, id: &str, error: String) {
        if let Some(entry) = self.tasks.write().await.get_mut(id) {
            entry.info.status = TaskStatus::Failed;
            entry.info.finished_at = Some(timestamp());
            entry.info.error = Some(error);
        }
    }

    pub async fn subscribe(&self, id: &str) -> Option<broadcast::Receiver<TaskEvent>> {
        self.tasks
            .read()
            .await
            .get(id)
            .map(|entry| entry.sender.subscribe())
    }

    pub async fn get_task(&self, id: &str) -> Option<TaskInfo> {
        self.tasks.read().await.get(id).map(|e| e.info.clone())
    }

    pub async fn list_tasks(&self) -> Vec<TaskInfo> {
        self.tasks
            .read()
            .await
            .values()
            .map(|e| e.info.clone())
            .collect()
    }
}

fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

fn rand_id() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    hasher.finish()
}
