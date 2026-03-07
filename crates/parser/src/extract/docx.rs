use crate::ParseWarning;
use crate::error::ParserError;
use std::path::Path;

/// Extract text from a DOCX file using docx_rs.
pub fn extract_docx(path: &Path) -> Result<(String, Vec<ParseWarning>), ParserError> {
    let bytes = std::fs::read(path).map_err(|e| ParserError::ReadError {
        path: path.display().to_string(),
        source: e,
    })?;

    let docx = docx_rs::read_docx(&bytes).map_err(|e| ParserError::DocxError {
        path: path.display().to_string(),
        message: format!("{:?}", e),
    })?;

    let mut text_parts: Vec<String> = Vec::new();

    for child in &docx.document.children {
        match child {
            docx_rs::DocumentChild::Paragraph(paragraph) => {
                let para_text = extract_paragraph_text(paragraph);
                if !para_text.is_empty() {
                    text_parts.push(para_text);
                }
            }
            docx_rs::DocumentChild::Table(table) => {
                let table_text = extract_table_text(table);
                if !table_text.is_empty() {
                    text_parts.push(table_text);
                }
            }
            _ => {}
        }
    }

    let text = text_parts.join("\n\n");
    Ok((text, Vec::new()))
}

/// Extract text from a single paragraph.
fn extract_paragraph_text(paragraph: &docx_rs::Paragraph) -> String {
    let mut parts: Vec<String> = Vec::new();

    for child in &paragraph.children {
        match child {
            docx_rs::ParagraphChild::Run(run) => {
                for run_child in &run.children {
                    if let docx_rs::RunChild::Text(text) = run_child {
                        parts.push(text.text.clone());
                    }
                }
            }
            _ => {}
        }
    }

    parts.join("")
}

/// Extract text from a table, formatting each row.
fn extract_table_text(table: &docx_rs::Table) -> String {
    let mut rows: Vec<String> = Vec::new();

    for row in &table.rows {
        match row {
            docx_rs::TableChild::TableRow(tr) => {
                let mut cells: Vec<String> = Vec::new();
                for cell in &tr.cells {
                    match cell {
                        docx_rs::TableRowChild::TableCell(tc) => {
                            let mut cell_text_parts: Vec<String> = Vec::new();
                            for child in &tc.children {
                                match child {
                                    docx_rs::TableCellContent::Paragraph(p) => {
                                        let t = extract_paragraph_text(p);
                                        if !t.is_empty() {
                                            cell_text_parts.push(t);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            cells.push(cell_text_parts.join(" "));
                        }
                    }
                }
                rows.push(cells.join(" | "));
            }
        }
    }

    rows.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_docx_nonexistent() {
        let result = extract_docx(Path::new("/nonexistent/file.docx"));
        assert!(result.is_err());
    }
}
