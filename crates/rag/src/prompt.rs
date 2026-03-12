use std::fmt::Write;

use conversations::{Message, Role};
use db::SearchResult;

use crate::inference::ModelArch;

const MAX_HISTORY_TURNS: usize = 10;

/// Overhead per search result: "[1] " + " (Source: " + ")\n" ≈ 20 bytes of markup.
const MARKUP_BYTES_PER_RESULT: usize = 20;
/// Overhead per history message: role tag + end tag ≈ 30 bytes of markup.
const MARKUP_BYTES_PER_MESSAGE: usize = 30;
/// Fixed overhead for system framing, special tokens, and safety margin.
const PROMPT_FRAME_OVERHEAD: usize = 256;

pub fn assemble_prompt(
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> String {
    let estimated_size = estimate_prompt_size(system_prompt, results, history, question);
    let mut prompt = String::with_capacity(estimated_size);

    // System section with retrieved context
    prompt.push_str("<|system|>\n");
    prompt.push_str(system_prompt);

    if !results.is_empty() {
        prompt.push_str("\n\nContext from course materials:\n");
        for (i, r) in results.iter().enumerate() {
            let _ = write!(prompt, "[{}] {} (Source: {})\n", i + 1, r.text, r.source_file);
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

pub fn assemble_prompt_gemma(
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> String {
    let estimated_size = estimate_prompt_size(system_prompt, results, history, question);
    let mut prompt = String::with_capacity(estimated_size);

    // Conversation history (last N turns)
    let history_start = if history.len() > MAX_HISTORY_TURNS * 2 {
        history.len() - MAX_HISTORY_TURNS * 2
    } else {
        0
    };
    for msg in &history[history_start..] {
        match msg.role {
            Role::User => {
                prompt.push_str("<start_of_turn>user\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
            Role::Assistant => {
                prompt.push_str("<start_of_turn>model\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
        }
    }

    // Current question — system prompt folded into first user turn (Gemma has no system role)
    prompt.push_str("<start_of_turn>user\n");
    prompt.push_str(system_prompt);

    if !results.is_empty() {
        prompt.push_str("\n\nContext from course materials:\n");
        for (i, r) in results.iter().enumerate() {
            let _ = write!(prompt, "[{}] {} (Source: {})\n", i + 1, r.text, r.source_file);
        }
    }

    prompt.push_str("\n\n");
    prompt.push_str(question);
    prompt.push_str("<end_of_turn>\n");
    prompt.push_str("<start_of_turn>model\n");

    prompt
}

fn estimate_prompt_size(
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> usize {
    let context_size: usize = results.iter().map(|r| r.text.len() + r.source_file.len() + MARKUP_BYTES_PER_RESULT).sum();
    let history_size: usize = history.iter().map(|m| m.content.len() + MARKUP_BYTES_PER_MESSAGE).sum();
    system_prompt.len() + context_size + history_size + question.len() + PROMPT_FRAME_OVERHEAD
}

pub fn assemble_prompt_llama(
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> String {
    let estimated_size = estimate_prompt_size(system_prompt, results, history, question);
    let mut prompt = String::with_capacity(estimated_size);

    // System turn
    prompt.push_str("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n");
    prompt.push_str(system_prompt);

    if !results.is_empty() {
        prompt.push_str("\n\nContext from course materials:\n");
        for (i, r) in results.iter().enumerate() {
            let _ = write!(prompt, "[{}] {} (Source: {})\n", i + 1, r.text, r.source_file);
        }
    }
    prompt.push_str("<|eot_id|>");

    // Conversation history (last N turns)
    let history_start = if history.len() > MAX_HISTORY_TURNS * 2 {
        history.len() - MAX_HISTORY_TURNS * 2
    } else {
        0
    };
    for msg in &history[history_start..] {
        match msg.role {
            Role::User => {
                prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|eot_id|>");
            }
            Role::Assistant => {
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|eot_id|>");
            }
        }
    }

    // Current question
    prompt.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
    prompt.push_str(question);
    prompt.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");

    prompt
}

pub fn assemble_prompt_for_arch(
    arch: ModelArch,
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> String {
    match arch {
        ModelArch::Phi3 => assemble_prompt(system_prompt, results, history, question),
        ModelArch::Gemma3 => assemble_prompt_gemma(system_prompt, results, history, question),
        ModelArch::Llama => assemble_prompt_llama(system_prompt, results, history, question),
    }
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

    #[test]
    fn test_assemble_prompt_gemma_basic() {
        let results = vec![SearchResult {
            text: "Mitosis is cell division.".to_string(),
            source_file: "ch3.pdf".to_string(),
            chunk_index: 5,
            score: 0.9,
        }];
        let prompt = assemble_prompt_gemma("You are a tutor.", &results, &[], "What is mitosis?");
        assert!(prompt.contains("<start_of_turn>user\n"));
        assert!(prompt.contains("You are a tutor."));
        assert!(prompt.contains("[1] Mitosis is cell division. (Source: ch3.pdf)"));
        assert!(prompt.contains("What is mitosis?<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
        // Gemma has no system role
        assert!(!prompt.contains("<|system|>"));
    }

    #[test]
    fn test_assemble_prompt_gemma_with_history() {
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
        let prompt = assemble_prompt_gemma("Tutor.", &[], &history, "Next question");
        assert!(prompt.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(prompt.contains("<start_of_turn>model\nHi there!<end_of_turn>"));
        assert!(prompt.contains("Next question<end_of_turn>"));
    }

    #[test]
    fn test_assemble_prompt_llama_basic() {
        let results = vec![SearchResult {
            text: "Mitosis is cell division.".to_string(),
            source_file: "ch3.pdf".to_string(),
            chunk_index: 5,
            score: 0.9,
        }];
        let prompt = assemble_prompt_llama("You are a tutor.", &results, &[], "What is mitosis?");
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(prompt.contains("You are a tutor."));
        assert!(prompt.contains("[1] Mitosis is cell division. (Source: ch3.pdf)"));
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nWhat is mitosis?<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_assemble_prompt_llama_with_history() {
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
        let prompt = assemble_prompt_llama("Tutor.", &[], &history, "Next question");
        assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"));
        assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"));
        assert!(prompt.contains("Next question<|eot_id|>"));
    }

    #[test]
    fn test_assemble_prompt_llama_no_context() {
        let prompt = assemble_prompt_llama("System.", &[], &[], "Question?");
        assert!(!prompt.contains("Context from course materials"));
        assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
    }
}
