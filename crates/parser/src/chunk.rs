use crate::ParseWarning;
use crate::extract::FileFormat;
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;

/// A document chunk ready for embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// The text content of this chunk.
    pub text: String,
    /// Metadata about where this chunk came from.
    pub source: ChunkSource,
}

/// Metadata about the origin of a chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSource {
    /// Path to the source file.
    pub file_path: String,
    /// Format of the source file.
    pub format: FileFormat,
    /// Index of this chunk within the file.
    pub chunk_index: usize,
    /// Sheet name (for Excel files).
    pub sheet_name: Option<String>,
    /// Row number (for CSV/Excel files, 1-indexed).
    pub row_number: Option<usize>,
}

/// Configuration for text chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size range in characters.
    pub target_size: RangeInclusive<usize>,
    /// Overlap percentage between consecutive chunks (0-100).
    pub overlap_percent: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_size: 800..=2000,
            overlap_percent: 10,
        }
    }
}

/// Split text into chunks using text_splitter, with fallback to fixed-size chunking.
/// Returns chunks and any warnings generated.
pub fn chunk_text(
    text: &str,
    file_path: &str,
    format: FileFormat,
    config: &ChunkConfig,
) -> (Vec<Chunk>, Vec<ParseWarning>) {
    let mut warnings = Vec::new();

    if text.trim().is_empty() {
        return (Vec::new(), warnings);
    }

    let max_size = *config.target_size.end();

    // Try text_splitter first
    let splitter = text_splitter::TextSplitter::new(max_size);
    let raw_chunks: Vec<&str> = splitter.chunks(text).collect();

    let chunk_texts = if raw_chunks.is_empty() {
        // Fallback to fixed-size chunking
        warnings.push(ParseWarning::FallbackToFixedChunking {
            path: file_path.to_string(),
        });
        fixed_size_chunks(text, config)
    } else {
        // Apply overlap if configured
        if config.overlap_percent > 0 {
            apply_overlap(&raw_chunks, config)
        } else {
            raw_chunks.iter().map(|s| s.to_string()).collect()
        }
    };

    let chunks = chunk_texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            text,
            source: ChunkSource {
                file_path: file_path.to_string(),
                format,
                chunk_index: i,
                sheet_name: None,
                row_number: None,
            },
        })
        .collect();

    (chunks, warnings)
}

/// Apply overlap between chunks by prepending tail of previous chunk.
fn apply_overlap(chunks: &[&str], config: &ChunkConfig) -> Vec<String> {
    if chunks.len() <= 1 {
        return chunks.iter().map(|s| s.to_string()).collect();
    }

    let mut result = Vec::with_capacity(chunks.len());
    result.push(chunks[0].to_string());

    for i in 1..chunks.len() {
        let prev = chunks[i - 1];
        let overlap_chars = prev.len() * config.overlap_percent / 100;
        if overlap_chars > 0 {
            // Take the last `overlap_chars` characters from the previous chunk
            let overlap_start = prev.len().saturating_sub(overlap_chars);
            // Find a char boundary
            let overlap_start = find_char_boundary(prev, overlap_start);
            let overlap = &prev[overlap_start..];
            result.push(format!("{}{}", overlap, chunks[i]));
        } else {
            result.push(chunks[i].to_string());
        }
    }

    result
}

/// Find the nearest char boundary at or after the given byte index.
fn find_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut i = idx;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Fixed-size chunking fallback.
fn fixed_size_chunks(text: &str, config: &ChunkConfig) -> Vec<String> {
    let max_size = *config.target_size.end();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let mut end = (start + max_size).min(text.len());
        // Adjust to char boundary
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        chunks.push(text[start..end].to_string());

        // Calculate next start with overlap
        let overlap_chars = max_size * config.overlap_percent / 100;
        let advance = if max_size > overlap_chars {
            max_size - overlap_chars
        } else {
            max_size
        };
        start += advance;
        // Adjust to char boundary
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_config_default() {
        let config = ChunkConfig::default();
        assert_eq!(*config.target_size.start(), 800);
        assert_eq!(*config.target_size.end(), 2000);
        assert_eq!(config.overlap_percent, 10);
    }

    #[test]
    fn test_chunk_empty_text() {
        let config = ChunkConfig::default();
        let (chunks, warnings) = chunk_text("", "test.txt", FileFormat::Txt, &config);
        assert!(chunks.is_empty());
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_chunk_whitespace_only() {
        let config = ChunkConfig::default();
        let (chunks, _) = chunk_text("   \n\n  ", "test.txt", FileFormat::Txt, &config);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_short_text() {
        let config = ChunkConfig {
            target_size: 100..=500,
            overlap_percent: 0,
        };
        let text = "This is a short piece of text.";
        let (chunks, _) = chunk_text(text, "test.txt", FileFormat::Txt, &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert_eq!(chunks[0].source.chunk_index, 0);
    }

    #[test]
    fn test_chunk_source_metadata() {
        let config = ChunkConfig {
            target_size: 100..=500,
            overlap_percent: 0,
        };
        let (chunks, _) = chunk_text("Hello world", "doc.pdf", FileFormat::Pdf, &config);
        assert_eq!(chunks[0].source.file_path, "doc.pdf");
        assert_eq!(chunks[0].source.format, FileFormat::Pdf);
        assert!(chunks[0].source.sheet_name.is_none());
        assert!(chunks[0].source.row_number.is_none());
    }

    #[test]
    fn test_fixed_size_chunks_no_overlap() {
        let config = ChunkConfig {
            target_size: 10..=10,
            overlap_percent: 0,
        };
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = fixed_size_chunks(text, &config);
        assert_eq!(chunks[0], "abcdefghij");
        assert_eq!(chunks[1], "klmnopqrst");
        assert_eq!(chunks[2], "uvwxyz");
    }

    #[test]
    fn test_fixed_size_chunks_with_overlap() {
        let config = ChunkConfig {
            target_size: 10..=10,
            overlap_percent: 20, // 2 chars overlap
        };
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = fixed_size_chunks(text, &config);
        // First chunk: 0..10 = "abcdefghij"
        // Second chunk starts at 10 - 2 = 8: "ijklmnopqr"
        assert_eq!(chunks[0], "abcdefghij");
        assert_eq!(chunks[1], "ijklmnopqr");
    }

    #[test]
    fn test_apply_overlap() {
        let chunks = vec!["hello world", "foo bar"];
        let config = ChunkConfig {
            target_size: 100..=200,
            overlap_percent: 50,
        };
        let result = apply_overlap(&chunks, &config);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "hello world");
        // 50% of "hello world" (11 chars) = 5 chars from the end = "world"
        assert_eq!(result[1], "worldfoo bar");
    }

    #[test]
    fn test_apply_overlap_single_chunk() {
        let chunks = vec!["only chunk"];
        let config = ChunkConfig {
            target_size: 100..=200,
            overlap_percent: 10,
        };
        let result = apply_overlap(&chunks, &config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "only chunk");
    }
}
