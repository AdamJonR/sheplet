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

pub fn assemble_prompt_qwen2(
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> String {
    let estimated_size = estimate_prompt_size(system_prompt, results, history, question);
    let mut prompt = String::with_capacity(estimated_size);

    // System turn (ChatML format)
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(system_prompt);

    if !results.is_empty() {
        prompt.push_str("\n\nContext from course materials:\n");
        for (i, r) in results.iter().enumerate() {
            let _ = write!(prompt, "[{}] {} (Source: {})\n", i + 1, r.text, r.source_file);
        }
    }
    prompt.push_str("<|im_end|>\n");

    // Conversation history (last N turns)
    let history_start = if history.len() > MAX_HISTORY_TURNS * 2 {
        history.len() - MAX_HISTORY_TURNS * 2
    } else {
        0
    };
    for msg in &history[history_start..] {
        match msg.role {
            Role::User => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<|im_end|>\n");
            }
        }
    }

    // Current question
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(question);
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");

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

    // Gemma has no system role — fold system prompt into first user turn
    prompt.push_str("<start_of_turn>user\n");
    prompt.push_str(system_prompt);

    if !results.is_empty() {
        prompt.push_str("\n\nContext from course materials:\n");
        for (i, r) in results.iter().enumerate() {
            let _ = write!(prompt, "[{}] {} (Source: {})\n", i + 1, r.text, r.source_file);
        }
    }

    // Conversation history (last N turns)
    let history_start = if history.len() > MAX_HISTORY_TURNS * 2 {
        history.len() - MAX_HISTORY_TURNS * 2
    } else {
        0
    };

    // If there's history, close the system-as-user turn, then add history
    if !history.is_empty() {
        // First user message in history gets merged with system context
        let mut first_user = true;
        for msg in &history[history_start..] {
            match msg.role {
                Role::User => {
                    if first_user {
                        // Append to the already-open user turn
                        prompt.push_str("\n\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str("<end_of_turn>\n");
                        first_user = false;
                    } else {
                        prompt.push_str("<start_of_turn>user\n");
                        prompt.push_str(&msg.content);
                        prompt.push_str("<end_of_turn>\n");
                    }
                }
                Role::Assistant => {
                    if first_user {
                        // Close system-as-user turn first
                        prompt.push_str("<end_of_turn>\n");
                        first_user = false;
                    }
                    prompt.push_str("<start_of_turn>model\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("<end_of_turn>\n");
                }
            }
        }
        // Current question
        prompt.push_str("<start_of_turn>user\n");
        prompt.push_str(question);
        prompt.push_str("<end_of_turn>\n");
    } else {
        // No history — add question to the system-as-user turn
        prompt.push_str("\n\n");
        prompt.push_str(question);
        prompt.push_str("<end_of_turn>\n");
    }

    prompt.push_str("<start_of_turn>model\n");

    prompt
}

pub fn assemble_prompt_mistral(
    system_prompt: &str,
    results: &[SearchResult],
    history: &[Message],
    question: &str,
) -> String {
    let estimated_size = estimate_prompt_size(system_prompt, results, history, question);
    let mut prompt = String::with_capacity(estimated_size);

    // Mistral v0.3 format: [INST] system + question [/INST]
    // For multi-turn: [INST] msg [/INST] response </s> [INST] msg [/INST]
    let mut system_with_context = system_prompt.to_string();
    if !results.is_empty() {
        system_with_context.push_str("\n\nContext from course materials:\n");
        for (i, r) in results.iter().enumerate() {
            let _ = write!(system_with_context, "[{}] {} (Source: {})\n", i + 1, r.text, r.source_file);
        }
    }

    // Conversation history (last N turns)
    let history_start = if history.len() > MAX_HISTORY_TURNS * 2 {
        history.len() - MAX_HISTORY_TURNS * 2
    } else {
        0
    };

    let mut first_inst = true;
    for msg in &history[history_start..] {
        match msg.role {
            Role::User => {
                if first_inst {
                    prompt.push_str("[INST] ");
                    prompt.push_str(&system_with_context);
                    prompt.push_str("\n\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str(" [/INST]");
                    first_inst = false;
                } else {
                    prompt.push_str("[INST] ");
                    prompt.push_str(&msg.content);
                    prompt.push_str(" [/INST]");
                }
            }
            Role::Assistant => {
                prompt.push_str(&msg.content);
                prompt.push_str("</s>");
            }
        }
    }

    // Current question
    if first_inst {
        prompt.push_str("[INST] ");
        prompt.push_str(&system_with_context);
        prompt.push_str("\n\n");
        prompt.push_str(question);
        prompt.push_str(" [/INST]");
    } else {
        prompt.push_str("[INST] ");
        prompt.push_str(question);
        prompt.push_str(" [/INST]");
    }

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
        ModelArch::Llama => assemble_prompt_llama(system_prompt, results, history, question),
        ModelArch::Qwen2 => assemble_prompt_qwen2(system_prompt, results, history, question),
        ModelArch::Gemma | ModelArch::Gemma2 => assemble_prompt_gemma(system_prompt, results, history, question),
        ModelArch::Mistral => assemble_prompt_mistral(system_prompt, results, history, question),
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

    #[test]
    fn test_assemble_prompt_qwen2_basic() {
        let results = vec![SearchResult {
            text: "Mitosis is cell division.".to_string(),
            source_file: "ch3.pdf".to_string(),
            chunk_index: 5,
            score: 0.9,
        }];
        let prompt = assemble_prompt_qwen2("You are a tutor.", &results, &[], "What is mitosis?");
        assert!(prompt.contains("<|im_start|>system\n"));
        assert!(prompt.contains("You are a tutor."));
        assert!(prompt.contains("[1] Mitosis is cell division. (Source: ch3.pdf)"));
        assert!(prompt.contains("<|im_start|>user\nWhat is mitosis?<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_assemble_prompt_qwen2_no_context() {
        let prompt = assemble_prompt_qwen2("System.", &[], &[], "Question?");
        assert!(!prompt.contains("Context from course materials"));
        assert!(prompt.contains("<|im_start|>system\n"));
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
        assert!(prompt.contains("What is mitosis?"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
        // Gemma has no system role
        assert!(!prompt.contains("system"));
    }

    #[test]
    fn test_assemble_prompt_gemma_no_context() {
        let prompt = assemble_prompt_gemma("System.", &[], &[], "Question?");
        assert!(!prompt.contains("Context from course materials"));
        assert!(prompt.contains("<start_of_turn>user\n"));
    }

    #[test]
    fn test_assemble_prompt_mistral_basic() {
        let results = vec![SearchResult {
            text: "Mitosis is cell division.".to_string(),
            source_file: "ch3.pdf".to_string(),
            chunk_index: 5,
            score: 0.9,
        }];
        let prompt = assemble_prompt_mistral("You are a tutor.", &results, &[], "What is mitosis?");
        assert!(prompt.contains("[INST] "));
        assert!(prompt.contains("You are a tutor."));
        assert!(prompt.contains("[1] Mitosis is cell division. (Source: ch3.pdf)"));
        assert!(prompt.contains("What is mitosis?"));
        assert!(prompt.contains(" [/INST]"));
    }

    #[test]
    fn test_assemble_prompt_mistral_no_context() {
        let prompt = assemble_prompt_mistral("System.", &[], &[], "Question?");
        assert!(!prompt.contains("Context from course materials"));
        assert!(prompt.contains("[INST] "));
    }

    #[test]
    fn test_assemble_prompt_qwen2_with_history() {
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
        let prompt = assemble_prompt_qwen2("Tutor.", &[], &history, "Next question");
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\nHi there!<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nNext question<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
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
        // First user message is merged with system-as-user turn
        assert!(prompt.contains("Tutor."));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains("<start_of_turn>model\nHi there!<end_of_turn>"));
        assert!(prompt.contains("<start_of_turn>user\nNext question<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
        // Gemma has no system role
        assert!(!prompt.contains("<start_of_turn>system"));
    }

    #[test]
    fn test_assemble_prompt_mistral_with_history() {
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
        let prompt = assemble_prompt_mistral("Tutor.", &[], &history, "Next question");
        // First [INST] includes system prompt + first user message
        assert!(prompt.contains("[INST] Tutor."));
        assert!(prompt.contains("Hello"));
        assert!(prompt.contains(" [/INST]"));
        assert!(prompt.contains("Hi there!</s>"));
        assert!(prompt.contains("Next question"));
    }
}
