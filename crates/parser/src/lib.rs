//! Document parsing and semantic chunking for the Sheplet platform.
//!
//! Supports PDF, DOCX, Excel, CSV, and plain text files. Text is extracted
//! and split into chunks suitable for embedding and retrieval.

pub mod chunk;
pub mod error;
pub mod extract;
pub mod walk;

pub use chunk::{Chunk, ChunkConfig, ChunkSource};
pub use error::ParserError;
pub use extract::FileFormat;

use std::path::Path;

/// Warnings generated during parsing that do not prevent extraction.
#[derive(Debug, Clone)]
pub enum ParseWarning {
    /// PDF text extraction quality may be degraded (scanned pages, images, etc.).
    PdfQualityDegraded { path: String },
    /// text_splitter produced no chunks; fell back to fixed-size chunking.
    FallbackToFixedChunking { path: String },
}

impl std::fmt::Display for ParseWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PdfQualityDegraded { path } => {
                write!(f, "PDF quality may be degraded: {}", path)
            }
            Self::FallbackToFixedChunking { path } => {
                write!(f, "Fell back to fixed-size chunking: {}", path)
            }
        }
    }
}

/// Detect the file format from a path's extension.
pub fn detect_format(path: &Path) -> Option<FileFormat> {
    extract::detect_format(path)
}

/// Parse all supported files in a directory recursively.
///
/// Walks the directory tree, extracts text from each supported file,
/// and chunks the text according to the provided configuration.
/// Hidden files and directories (starting with '.') are skipped.
pub fn parse_directory(
    dir: &Path,
    config: &ChunkConfig,
) -> Result<(Vec<Chunk>, Vec<ParseWarning>), ParserError> {
    let files = walk::walk_directory(dir)?;
    let mut all_chunks = Vec::new();
    let mut all_warnings = Vec::new();

    for file_path in files {
        match parse_file(&file_path, config) {
            Ok((chunks, warnings)) => {
                all_chunks.extend(chunks);
                all_warnings.extend(warnings);
            }
            Err(ParserError::EmptyExtraction { .. }) => {
                // Skip empty files silently
            }
            Err(e) => return Err(e),
        }
    }

    Ok((all_chunks, all_warnings))
}

/// Parse a single file and return chunks.
///
/// For CSV and Excel files, rows are returned as individual chunks with
/// headers prepended. For all other formats, text is extracted and split
/// using the text_splitter with fallback to fixed-size chunking.
pub fn parse_file(
    path: &Path,
    config: &ChunkConfig,
) -> Result<(Vec<Chunk>, Vec<ParseWarning>), ParserError> {
    let format = extract::detect_format(path).ok_or_else(|| ParserError::UnsupportedFormat {
        path: path.display().to_string(),
    })?;

    // Check file size before processing (prevents decompression bombs and memory exhaustion)
    extract::check_file_size(path)?;

    // CSV and Excel use row-based chunking
    match format {
        FileFormat::Csv => return extract::csv_extract::extract_csv_rows(path, config),
        FileFormat::Xlsx => return extract::xlsx::extract_xlsx_rows(path, config),
        _ => {}
    }

    // For other formats, extract text and chunk it
    let (text, mut warnings) = extract::extract_text(path, format)?;

    if text.trim().is_empty() {
        return Err(ParserError::EmptyExtraction {
            path: path.display().to_string(),
        });
    }

    let file_path_str = path.display().to_string();
    let (chunks, chunk_warnings) = chunk::chunk_text(&text, &file_path_str, format, config);
    warnings.extend(chunk_warnings);

    Ok((chunks, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_public_api() {
        assert_eq!(
            detect_format(Path::new("test.pdf")),
            Some(FileFormat::Pdf)
        );
        assert_eq!(detect_format(Path::new("test.xyz")), None);
    }

    #[test]
    fn test_parse_file_unsupported() {
        let result = parse_file(Path::new("test.xyz"), &ChunkConfig::default());
        assert!(matches!(result, Err(ParserError::UnsupportedFormat { .. })));
    }

    #[test]
    fn test_parse_file_nonexistent_txt() {
        let result = parse_file(Path::new("/nonexistent/file.txt"), &ChunkConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_file_txt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "Hello, world! This is a test document.").unwrap();

        let config = ChunkConfig::default();
        let (chunks, warnings) = parse_file(&path, &config).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].text.contains("Hello"));
        assert_eq!(chunks[0].source.format, FileFormat::Txt);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_parse_file_empty_txt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, "").unwrap();

        let result = parse_file(&path, &ChunkConfig::default());
        assert!(matches!(result, Err(ParserError::EmptyExtraction { .. })));
    }

    #[test]
    fn test_parse_directory_with_txt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("doc1.txt"),
            "First document with some text content.",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("doc2.txt"),
            "Second document with different content.",
        )
        .unwrap();

        let config = ChunkConfig::default();
        let (chunks, _) = parse_directory(dir.path(), &config).unwrap();
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_parse_directory_skips_hidden() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "visible content").unwrap();
        std::fs::write(dir.path().join(".hidden.txt"), "hidden content").unwrap();

        let config = ChunkConfig::default();
        let (chunks, _) = parse_directory(dir.path(), &config).unwrap();
        // Only the visible file should be parsed
        for chunk in &chunks {
            assert!(!chunk.source.file_path.contains(".hidden"));
        }
    }

    #[test]
    fn test_parse_directory_empty() {
        let dir = tempfile::tempdir().unwrap();
        let config = ChunkConfig::default();
        let (chunks, warnings) = parse_directory(dir.path(), &config).unwrap();
        assert!(chunks.is_empty());
        assert!(warnings.is_empty());
    }
}
