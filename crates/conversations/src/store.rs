use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{ConversationError, Result};
use crate::types::{Conversation, ConversationSummary, Message};

pub struct ConversationStore {
    db: sled::Db,
}

fn now_iso() -> String {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    // Simple ISO-like timestamp without pulling in chrono
    let days_since_epoch = secs / 86400;
    // Approximate date calculation
    let (year, month, day) = days_to_date(days_since_epoch);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{s:02}Z")
}

fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn generate_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let random: u64 = rand::random();
    format!("{millis:x}-{random:016x}")
}

impl ConversationStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn create_conversation(&self, course_id: &str, title: &str) -> Result<Conversation> {
        let id = generate_id();
        let now = now_iso();
        let conversation = Conversation {
            id: id.clone(),
            course_id: course_id.to_string(),
            title: title.to_string(),
            created_at: now.clone(),
            updated_at: now,
            messages: Vec::new(),
        };
        let key = format!("{course_id}:{id}");
        let value = serde_json::to_vec(&conversation)?;
        self.db.insert(key.as_bytes(), value)?;
        Ok(conversation)
    }

    pub fn get(&self, id: &str) -> Result<Option<Conversation>> {
        // We need to scan since we don't know the course_id from just the id
        for item in self.db.iter() {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            // Split on first ':' and match the ID portion exactly
            if key_str.split_once(':').is_some_and(|(_, key_id)| key_id == id) {
                let conv: Conversation = serde_json::from_slice(&value)?;
                return Ok(Some(conv));
            }
        }
        Ok(None)
    }

    pub fn append_message(&self, id: &str, msg: Message) -> Result<()> {
        // Find the conversation
        for item in self.db.iter() {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            // Split on first ':' and match the ID portion exactly
            if key_str.split_once(':').is_some_and(|(_, key_id)| key_id == id) {
                let mut conv: Conversation = serde_json::from_slice(&value)?;
                conv.updated_at = now_iso();
                conv.messages.push(msg);
                let new_value = serde_json::to_vec(&conv)?;
                self.db.insert(key.as_ref(), new_value)?;
                return Ok(());
            }
        }
        Err(ConversationError::NotFound(id.to_string()))
    }

    pub fn list_by_course(&self, course_id: &str) -> Result<Vec<ConversationSummary>> {
        let prefix = format!("{course_id}:");
        let mut summaries = Vec::new();
        for item in self.db.scan_prefix(prefix.as_bytes()) {
            let (_key, value) = item?;
            let conv: Conversation = serde_json::from_slice(&value)?;
            summaries.push(conv.summary());
        }
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(summaries)
    }

    pub fn list_all(&self) -> Result<Vec<ConversationSummary>> {
        let mut summaries = Vec::new();
        for item in self.db.iter() {
            let (_key, value) = item?;
            let conv: Conversation = serde_json::from_slice(&value)?;
            summaries.push(conv.summary());
        }
        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(summaries)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        for item in self.db.iter() {
            let (key, _value) = item?;
            let key_str = String::from_utf8_lossy(&key);
            // Split on first ':' and match the ID portion exactly
            if key_str.split_once(':').is_some_and(|(_, key_id)| key_id == id) {
                self.db.remove(key)?;
                return Ok(());
            }
        }
        Err(ConversationError::NotFound(id.to_string()))
    }

    pub fn clear_course(&self, course_id: &str) -> Result<()> {
        let prefix = format!("{course_id}:");
        let keys: Vec<_> = self
            .db
            .scan_prefix(prefix.as_bytes())
            .filter_map(|item| item.ok().map(|(k, _)| k))
            .collect();
        for key in keys {
            self.db.remove(key)?;
        }
        Ok(())
    }
}
