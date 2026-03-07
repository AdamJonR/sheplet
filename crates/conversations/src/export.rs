use crate::types::{Conversation, Role};

pub fn export_as_txt(conversation: &Conversation) -> String {
    let mut out = String::new();
    out.push_str(&format!("Conversation: {}\n", conversation.title));
    out.push_str(&format!("Course: {}\n", conversation.course_id));
    out.push_str(&format!("Created: {}\n", conversation.created_at));
    out.push_str(&format!("Exported: {}\n", conversation.updated_at));
    out.push_str(&"=".repeat(60));
    out.push('\n');
    out.push('\n');

    for msg in &conversation.messages {
        let role_label = match msg.role {
            Role::User => "Student",
            Role::Assistant => "Tutor",
        };
        out.push_str(&format!("[{}] {}:\n", msg.timestamp, role_label));
        out.push_str(&msg.content);
        out.push('\n');

        if !msg.citations.is_empty() {
            out.push_str("\n  Sources:\n");
            for cite in &msg.citations {
                out.push_str(&format!(
                    "    - {} (chunk {}): {}\n",
                    cite.source_file,
                    cite.chunk_index,
                    truncate(&cite.text_snippet, 80),
                ));
            }
        }
        out.push('\n');
    }
    out
}

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..s.floor_char_boundary(max_len)]
    }
}
