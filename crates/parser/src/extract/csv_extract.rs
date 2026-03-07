use crate::ParseWarning;
use crate::chunk::ChunkSource;
use crate::error::ParserError;
use crate::extract::FileFormat;
use crate::{Chunk, ChunkConfig};
use std::path::Path;

/// Extract text from a CSV file (for use with text-splitter chunking).
pub fn extract_csv_text(path: &Path) -> Result<(String, Vec<ParseWarning>), ParserError> {
    let mut reader =
        csv::ReaderBuilder::new()
            .from_path(path)
            .map_err(|e| ParserError::CsvError {
                path: path.display().to_string(),
                message: e.to_string(),
            })?;

    let mut all_text = String::new();

    // Read headers
    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| ParserError::CsvError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?
        .iter()
        .map(|h| h.to_string())
        .collect();

    all_text.push_str(&headers.join(" | "));
    all_text.push('\n');

    for record in reader.records() {
        let record = record.map_err(|e| ParserError::CsvError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let fields: Vec<&str> = record.iter().collect();
        all_text.push_str(&fields.join(" | "));
        all_text.push('\n');
    }

    Ok((all_text, Vec::new()))
}

/// Extract rows from a CSV file as individual chunks.
/// Each row becomes its own chunk with headers prepended: "col1: val1 | col2: val2"
pub fn extract_csv_rows(
    path: &Path,
    _config: &ChunkConfig,
) -> Result<(Vec<Chunk>, Vec<ParseWarning>), ParserError> {
    let mut reader =
        csv::ReaderBuilder::new()
            .from_path(path)
            .map_err(|e| ParserError::CsvError {
                path: path.display().to_string(),
                message: e.to_string(),
            })?;

    let file_path_str = path.display().to_string();

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| ParserError::CsvError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?
        .iter()
        .map(|h| h.to_string())
        .collect();

    let mut chunks = Vec::new();

    for (row_idx, record) in reader.records().enumerate() {
        let record = record.map_err(|e| ParserError::CsvError {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;

        let parts: Vec<String> = headers
            .iter()
            .zip(record.iter())
            .filter(|(_, val)| !val.is_empty())
            .map(|(header, val)| format!("{}: {}", header, val))
            .collect();

        if parts.is_empty() {
            continue;
        }

        let text = parts.join(" | ");
        chunks.push(Chunk {
            text,
            source: ChunkSource {
                file_path: file_path_str.clone(),
                format: FileFormat::Csv,
                chunk_index: row_idx,
                sheet_name: None,
                row_number: Some(row_idx + 2), // 1-indexed, +1 for header row
            },
        });
    }

    Ok((chunks, Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_csv_nonexistent() {
        let result = extract_csv_text(Path::new("/nonexistent/file.csv"));
        assert!(result.is_err());
    }
}
