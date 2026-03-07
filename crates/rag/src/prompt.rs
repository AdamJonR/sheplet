use conversations::{Message, Role};
use db::SearchResult;

const MAX_HISTORY_TURNS: usize = 10;

pub fn assemble_prompt(
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> String {
    let mut prompt = String::new();

    // System section with retrieved context
    prompt.push_str("<|system|>\n");
    prompt.push_str(system_prompt);

    if !results.is_empty() {
        prompt.push_str("\n\nContext from course materials:\n");
        for (i, r) in results.iter().enumerate() {
            prompt.push_str(&format!(
                "[{}] {} (Source: {})\n",
                i + 1,
                r.text,
                r.source_file
            ));
        }
    }
    prompt.push_str("<|end|>\n");

    // Conversation history (last N turns)
    let history_start = if history.len() > MAX_HISTORY_TURNS * 2 {
        history.len() - MAX_HISTORY_TURNS * 2
    } else {
        0
    };
    for msg in &history[history_start..] {
        match msg.role {
            Role::User => {
                prompt.push_str("<|user|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|end|>\n");
            }
            Role::Assistant => {
                prompt.push_str("<|assistant|>\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|end|>\n");
            }
        }
    }

    // Current question
    prompt.push_str("<|user|>\n");
    prompt.push_str(question);
    prompt.push_str("<|end|>\n");
    prompt.push_str("<|assistant|>\n");

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemble_prompt_basic() {
        let results = vec![SearchResult {
            text: "Mitosis is cell division.".to_string(),
            source_file: "ch3.pdf".to_string(),
            chunk_index: 5,
            score: 0.9,
        }];
        let prompt = assemble_prompt("You are a tutor.", &results, &[], "What is mitosis?");
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("You are a tutor."));
        assert!(prompt.contains("[1] Mitosis is cell division. (Source: ch3.pdf)"));
        assert!(prompt.contains("<|user|>\nWhat is mitosis?<|end|>"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_assemble_prompt_with_history() {
        let history = vec![
            Message {
                role: Role::User,
                content: "Hello".to_string(),
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                citations: vec![],
            },
            Message {
                role: Role::Assistant,
                content: "Hi there!".to_string(),
                timestamp: "2026-01-01T00:00:01Z".to_string(),
                citations: vec![],
            },
        ];
        let prompt = assemble_prompt("Tutor.", &[], &history, "Next question");
        assert!(prompt.contains("<|user|>\nHello<|end|>"));
        assert!(prompt.contains("<|assistant|>\nHi there!<|end|>"));
        assert!(prompt.contains("<|user|>\nNext question<|end|>"));
    }

    #[test]
    fn test_assemble_prompt_no_context() {
        let prompt = assemble_prompt("System.", &[], &[], "Question?");
        assert!(!prompt.contains("Context from course materials"));
    }
}
