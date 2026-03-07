use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{ConversationError, Result};
use crate::types::{Conversation, ConversationSummary, Message};

pub struct ConversationStore {
    db: sled::Db,
    /// Secondary index: conversation id → course_id for O(1) lookups.
    id_index: sled::Tree,
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
        let id_index = db.open_tree("id_index")?;

        // Migrate: if the main tree has data but the index is empty, populate it.
        if id_index.is_empty() && !db.is_empty() {
            for item in db.iter() {
                let (key, _) = item?;
                let key_str = String::from_utf8_lossy(&key);
                if let Some((course_id, id)) = key_str.split_once(':') {
                    id_index.insert(id.as_bytes(), course_id.as_bytes())?;
                }
            }
        }

        Ok(Self { db, id_index })
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
        self.id_index.insert(id.as_bytes(), course_id.as_bytes())?;
        Ok(conversation)
    }

    /// Build the full key `{course_id}:{id}` using the secondary index.
    fn resolve_key(&self, id: &str) -> Result<Option<String>> {
        if let Some(course_id_bytes) = self.id_index.get(id.as_bytes())? {
            let course_id = String::from_utf8_lossy(&course_id_bytes);
            return Ok(Some(format!("{course_id}:{id}")));
        }
        // Fallback: scan (shouldn't happen if index is populated).
        for item in self.db.iter() {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);
            if key_str.split_once(':').is_some_and(|(_, key_id)| key_id == id) {
                return Ok(Some(key_str.into_owned()));
            }
        }
        Ok(None)
    }

    pub fn get(&self, id: &str) -> Result<Option<Conversation>> {
        let Some(key) = self.resolve_key(id)? else {
            return Ok(None);
        };
        match self.db.get(key.as_bytes())? {
            Some(value) => Ok(Some(serde_json::from_slice(&value)?)),
            None => Ok(None),
        }
    }

    pub fn append_message(&self, id: &str, msg: Message) -> Result<()> {
        let key = self
            .resolve_key(id)?
            .ok_or_else(|| ConversationError::NotFound(id.to_string()))?;
        let value = self
            .db
            .get(key.as_bytes())?
            .ok_or_else(|| ConversationError::NotFound(id.to_string()))?;
        let mut conv: Conversation = serde_json::from_slice(&value)?;
        conv.updated_at = now_iso();
        conv.messages.push(msg);
        let new_value = serde_json::to_vec(&conv)?;
        self.db.insert(key.as_bytes(), new_value)?;
        Ok(())
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
        let key = self
            .resolve_key(id)?
            .ok_or_else(|| ConversationError::NotFound(id.to_string()))?;
        self.db.remove(key.as_bytes())?;
        self.id_index.remove(id.as_bytes())?;
        Ok(())
    }

    pub fn clear_course(&self, course_id: &str) -> Result<()> {
        let prefix = format!("{course_id}:");
        let keys: Vec<_> = self
            .db
            .scan_prefix(prefix.as_bytes())
            .filter_map(|item| item.ok().map(|(k, _)| k))
            .collect();
        for key in &keys {
            let key_str = String::from_utf8_lossy(key);
            if let Some((_, id)) = key_str.split_once(':') {
                self.id_index.remove(id.as_bytes())?;
            }
            self.db.remove(key)?;
        }
        Ok(())
    }
}
