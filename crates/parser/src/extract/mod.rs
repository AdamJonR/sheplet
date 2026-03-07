pub mod csv_extract;
pub mod docx;
pub mod pdf;
pub mod txt;
pub mod xlsx;

use serde::{Deserialize, Serialize};

/// Supported file formats for document parsing.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FileFormat {
    Pdf,
    Docx,
    Xlsx,
    Csv,
    Txt,
}

impl FileFormat {
    /// Returns the typical file extensions for this format.
    pub fn extensions(&self) -> &[&str] {
        match self {
            FileFormat::Pdf => &["pdf"],
            FileFormat::Docx => &["docx"],
            FileFormat::Xlsx => &["xlsx", "xls", "ods"],
            FileFormat::Csv => &["csv", "tsv"],
            FileFormat::Txt => &["txt", "md", "rst", "text"],
        }
    }
}

/// Detect file format from path extension.
pub fn detect_format(path: &std::path::Path) -> Option<FileFormat> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    match ext.as_str() {
        "pdf" => Some(FileFormat::Pdf),
        "docx" => Some(FileFormat::Docx),
        "xlsx" | "xls" | "ods" => Some(FileFormat::Xlsx),
        "csv" | "tsv" => Some(FileFormat::Csv),
        "txt" | "md" | "rst" | "text" => Some(FileFormat::Txt),
        _ => None,
    }
}

/// Extract raw text from a file based on its format.
/// Returns the extracted text and any warnings.
pub fn extract_text(
    path: &std::path::Path,
    format: FileFormat,
) -> Result<(String, Vec<crate::ParseWarning>), crate::error::ParserError> {
    match format {
        FileFormat::Pdf => pdf::extract_pdf(path),
        FileFormat::Docx => docx::extract_docx(path),
        FileFormat::Txt => txt::extract_txt(path),
        // Xlsx and Csv are handled separately (row-based chunking)
        FileFormat::Xlsx => xlsx::extract_xlsx_text(path),
        FileFormat::Csv => csv_extract::extract_csv_text(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_detect_format_pdf() {
        assert_eq!(detect_format(Path::new("file.pdf")), Some(FileFormat::Pdf));
    }

    #[test]
    fn test_detect_format_docx() {
        assert_eq!(
            detect_format(Path::new("file.docx")),
            Some(FileFormat::Docx)
        );
    }

    #[test]
    fn test_detect_format_xlsx() {
        assert_eq!(
            detect_format(Path::new("file.xlsx")),
            Some(FileFormat::Xlsx)
        );
        assert_eq!(
            detect_format(Path::new("file.xls")),
            Some(FileFormat::Xlsx)
        );
        assert_eq!(
            detect_format(Path::new("file.ods")),
            Some(FileFormat::Xlsx)
        );
    }

    #[test]
    fn test_detect_format_csv() {
        assert_eq!(detect_format(Path::new("file.csv")), Some(FileFormat::Csv));
        assert_eq!(detect_format(Path::new("file.tsv")), Some(FileFormat::Csv));
    }

    #[test]
    fn test_detect_format_txt() {
        assert_eq!(detect_format(Path::new("file.txt")), Some(FileFormat::Txt));
        assert_eq!(detect_format(Path::new("file.md")), Some(FileFormat::Txt));
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format(Path::new("file.xyz")), None);
        assert_eq!(detect_format(Path::new("file")), None);
    }

    #[test]
    fn test_detect_format_case_insensitive() {
        assert_eq!(detect_format(Path::new("file.PDF")), Some(FileFormat::Pdf));
        assert_eq!(
            detect_format(Path::new("file.DOCX")),
            Some(FileFormat::Docx)
        );
    }
}
