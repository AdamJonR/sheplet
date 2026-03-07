use crate::ParseWarning;
use crate::error::ParserError;
use std::path::Path;

/// Extract text from a PDF file using pdf_extract.
/// Returns the extracted text and any quality warnings.
pub fn extract_pdf(path: &Path) -> Result<(String, Vec<ParseWarning>), ParserError> {
    let mut warnings = Vec::new();

    let text = pdf_extract::extract_text(path).map_err(|e| ParserError::PdfError {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;

    // Quality heuristic: check file size vs text length
    let file_size = std::fs::metadata(path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);

    if file_size > 0 {
        let text_ratio = text.len() as f64 / file_size as f64;
        if text_ratio < 0.1 {
            warnings.push(ParseWarning::PdfQualityDegraded {
                path: path.display().to_string(),
            });
        }
    }

    // Quality heuristic: check for non-printable characters
    if !text.is_empty() {
        let non_printable_count = text
            .chars()
            .filter(|c| !c.is_ascii_graphic() && !c.is_ascii_whitespace() && !c.is_alphanumeric())
            .count();
        let non_printable_ratio = non_printable_count as f64 / text.chars().count() as f64;
        if non_printable_ratio > 0.15 {
            // Only add if we haven't already added a quality warning
            if !warnings
                .iter()
                .any(|w| matches!(w, ParseWarning::PdfQualityDegraded { .. }))
            {
                warnings.push(ParseWarning::PdfQualityDegraded {
                    path: path.display().to_string(),
                });
            }
        }
    }

    Ok((text, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_pdf_nonexistent() {
        let result = extract_pdf(Path::new("/nonexistent/file.pdf"));
        assert!(result.is_err());
    }
}
