use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::path::Path;

use crate::error::FinetuneError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoExample {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SftExample {
    pub input: String,
    pub output: String,
}

pub fn load_dpo_data(path: impl AsRef<Path>) -> Result<Vec<DpoExample>, FinetuneError> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| FinetuneError::DataLoading(format!("failed to open file: {e}")))?;
    let reader = std::io::BufReader::new(file);
    let mut examples = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|e| FinetuneError::DataLoading(format!("line {}: {e}", line_num + 1)))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let example: DpoExample = serde_json::from_str(&line)
            .map_err(|e| FinetuneError::DataLoading(format!("line {}: {e}", line_num + 1)))?;
        examples.push(example);
    }
    Ok(examples)
}

pub fn load_sft_data(path: impl AsRef<Path>) -> Result<Vec<SftExample>, FinetuneError> {
    let file = std::fs::File::open(path.as_ref())
        .map_err(|e| FinetuneError::DataLoading(format!("failed to open file: {e}")))?;
    let reader = std::io::BufReader::new(file);
    let mut examples = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|e| FinetuneError::DataLoading(format!("line {}: {e}", line_num + 1)))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let example: SftExample = serde_json::from_str(&line)
            .map_err(|e| FinetuneError::DataLoading(format!("line {}: {e}", line_num + 1)))?;
        examples.push(example);
    }
    Ok(examples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_dpo_data_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dpo.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"prompt":"q1","chosen":"a1","rejected":"b1"}}"#).unwrap();
        writeln!(f, r#"{{"prompt":"q2","chosen":"a2","rejected":"b2"}}"#).unwrap();

        let data = load_dpo_data(&path).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].prompt, "q1");
        assert_eq!(data[0].chosen, "a1");
        assert_eq!(data[0].rejected, "b1");
        assert_eq!(data[1].prompt, "q2");
    }

    #[test]
    fn test_load_sft_data_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sft.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"hello","output":"world"}}"#).unwrap();
        writeln!(f, r#"{{"input":"foo","output":"bar"}}"#).unwrap();

        let data = load_sft_data(&path).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].input, "hello");
        assert_eq!(data[1].output, "bar");
    }

    #[test]
    fn test_load_dpo_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "not json").unwrap();

        let result = load_dpo_data(&path);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("line 1"));
    }

    #[test]
    fn test_load_sft_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"wrong":"fields"}}"#).unwrap();

        let result = load_sft_data(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::File::create(&path).unwrap();

        let dpo = load_dpo_data(&path).unwrap();
        assert!(dpo.is_empty());

        let sft = load_sft_data(&path).unwrap();
        assert!(sft.is_empty());
    }

    #[test]
    fn test_load_with_empty_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gaps.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"input":"a","output":"b"}}"#).unwrap();
        writeln!(f).unwrap();
        writeln!(f, r#"{{"input":"c","output":"d"}}"#).unwrap();

        let data = load_sft_data(&path).unwrap();
        assert_eq!(data.len(), 2);
    }
}
