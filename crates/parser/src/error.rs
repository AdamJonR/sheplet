use thiserror::Error;

/// Errors that can occur during document parsing and chunking.
#[derive(Debug, Error)]
pub enum ParserError {
    #[error("unsupported file format: {path}")]
    UnsupportedFormat { path: String },

    #[error("failed to read file: {path}: {source}")]
    ReadError {
        path: String,
        source: std::io::Error,
    },

    #[error("PDF extraction failed for {path}: {message}")]
    PdfError { path: String, message: String },

    #[error("DOCX extraction failed for {path}: {message}")]
    DocxError { path: String, message: String },

    #[error("Excel extraction failed for {path}: {message}")]
    XlsxError { path: String, message: String },

    #[error("CSV extraction failed for {path}: {message}")]
    CsvError { path: String, message: String },

    #[error("directory walk failed: {path}: {source}")]
    WalkError {
        path: String,
        source: std::io::Error,
    },

    #[error("no text extracted from file: {path}")]
    EmptyExtraction { path: String },
}
