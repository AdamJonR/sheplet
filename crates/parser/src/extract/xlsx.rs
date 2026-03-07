use crate::ParseWarning;
use crate::chunk::ChunkSource;
use crate::error::ParserError;
use crate::extract::FileFormat;
use crate::{Chunk, ChunkConfig};
use calamine::{Reader, open_workbook_auto};
use std::path::Path;

/// Maximum number of rows to process from an Excel file.
const MAX_XLSX_ROWS: usize = 500_000;

/// Extract text from an Excel file (for use with text-splitter chunking).
/// This concatenates all sheets into a single text block.
pub fn extract_xlsx_text(path: &Path) -> Result<(String, Vec<ParseWarning>), ParserError> {
    let mut workbook = open_workbook_auto(path).map_err(|e| ParserError::XlsxError {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;

    let mut all_text = String::new();
    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();

    let mut total_rows = 0;
    for name in &sheet_names {
        if let Ok(range) = workbook.worksheet_range(name) {
            for row in range.rows() {
                if total_rows >= MAX_XLSX_ROWS {
                    break;
                }
                let cells: Vec<String> = row
                    .iter()
                    .map(|cell| format!("{}", cell))
                    .collect();
                all_text.push_str(&cells.join(" | "));
                all_text.push('\n');
                total_rows += 1;
            }
        }
        if total_rows >= MAX_XLSX_ROWS {
            break;
        }
    }

    Ok((all_text, Vec::new()))
}

/// Extract rows from an Excel file as individual chunks.
/// Each row becomes its own chunk with headers prepended: "col1: val1 | col2: val2"
pub fn extract_xlsx_rows(
    path: &Path,
    _config: &ChunkConfig,
) -> Result<(Vec<Chunk>, Vec<ParseWarning>), ParserError> {
    let mut workbook = open_workbook_auto(path).map_err(|e| ParserError::XlsxError {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;

    let mut chunks = Vec::new();
    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    let file_path_str = path.display().to_string();

    let mut total_rows = 0;
    for sheet_name in &sheet_names {
        if let Ok(range) = workbook.worksheet_range(sheet_name) {
            let rows: Vec<Vec<String>> = range
                .rows()
                .take(MAX_XLSX_ROWS + 1) // +1 for header
                .map(|row| row.iter().map(|cell| format!("{}", cell)).collect())
                .collect();

            if rows.is_empty() {
                continue;
            }

            let headers = &rows[0];
            let mut chunk_index = 0;

            for (row_idx, row) in rows.iter().enumerate().skip(1) {
                if total_rows >= MAX_XLSX_ROWS {
                    break;
                }
                total_rows += 1;
                let parts: Vec<String> = headers
                    .iter()
                    .zip(row.iter())
                    .filter(|(_, val)| !val.is_empty() && *val != "empty")
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
                        format: FileFormat::Xlsx,
                        chunk_index,
                        sheet_name: Some(sheet_name.clone()),
                        row_number: Some(row_idx + 1), // 1-indexed
                    },
                });
                chunk_index += 1;
            }
        }
    }

    Ok((chunks, Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_xlsx_nonexistent() {
        let result = extract_xlsx_text(Path::new("/nonexistent/file.xlsx"));
        assert!(result.is_err());
    }
}
