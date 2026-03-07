use crate::ParseWarning;
use crate::error::ParserError;
use std::path::Path;

/// Extract text from a plain text file.
pub fn extract_txt(path: &Path) -> Result<(String, Vec<ParseWarning>), ParserError> {
    let text = std::fs::read_to_string(path).map_err(|e| ParserError::ReadError {
        path: path.display().to_string(),
        source: e,
    })?;

    Ok((text, Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_txt_nonexistent() {
        let result = extract_txt(Path::new("/nonexistent/file.txt"));
        assert!(result.is_err());
    }
}
