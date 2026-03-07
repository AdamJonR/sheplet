use parser::{ChunkConfig, FileFormat, detect_format, parse_directory, parse_file};
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
fn test_parse_txt_fixture() {
    let path = fixtures_dir().join("sample.txt");
    let config = ChunkConfig::default();
    let (chunks, warnings) = parse_file(&path, &config).unwrap();

    assert!(!chunks.is_empty(), "should produce at least one chunk");
    assert!(warnings.is_empty(), "txt files should produce no warnings");

    // All chunks should reference the correct file and format
    for chunk in &chunks {
        assert!(chunk.source.file_path.ends_with("sample.txt"));
        assert_eq!(chunk.source.format, FileFormat::Txt);
        assert!(chunk.source.sheet_name.is_none());
        assert!(chunk.source.row_number.is_none());
    }

    // The full text should be recoverable (approximately) from chunks
    let combined: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("");
    assert!(combined.contains("Machine learning"));
    assert!(combined.contains("Deep Learning") || combined.contains("deep learning"));
}

#[test]
fn test_parse_csv_fixture() {
    let path = fixtures_dir().join("sample.csv");
    let config = ChunkConfig::default();
    let (chunks, warnings) = parse_file(&path, &config).unwrap();

    // CSV should produce one chunk per data row (5 rows)
    assert_eq!(chunks.len(), 5, "CSV should produce 5 row chunks");
    assert!(warnings.is_empty());

    // Check format metadata
    for chunk in &chunks {
        assert_eq!(chunk.source.format, FileFormat::Csv);
        assert!(chunk.source.row_number.is_some());
    }

    // First row should have Alice's data with headers prepended
    assert!(chunks[0].text.contains("name: Alice Smith"));
    assert!(chunks[0].text.contains("course: CS101"));
    assert!(chunks[0].text.contains("grade: A"));

    // Row numbers should be 2-indexed (1 for header + 1-based)
    assert_eq!(chunks[0].source.row_number, Some(2));
    assert_eq!(chunks[4].source.row_number, Some(6));
}

#[test]
fn test_parse_directory_fixtures() {
    let config = ChunkConfig::default();
    let (chunks, _) = parse_directory(&fixtures_dir(), &config).unwrap();

    // Should have chunks from both sample.txt and sample.csv
    let txt_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.source.format == FileFormat::Txt)
        .collect();
    let csv_chunks: Vec<_> = chunks
        .iter()
        .filter(|c| c.source.format == FileFormat::Csv)
        .collect();

    assert!(!txt_chunks.is_empty(), "should have txt chunks");
    assert!(!csv_chunks.is_empty(), "should have csv chunks");
}

#[test]
fn test_detect_format_various() {
    assert_eq!(detect_format(Path::new("report.pdf")), Some(FileFormat::Pdf));
    assert_eq!(detect_format(Path::new("doc.docx")), Some(FileFormat::Docx));
    assert_eq!(detect_format(Path::new("data.xlsx")), Some(FileFormat::Xlsx));
    assert_eq!(detect_format(Path::new("data.csv")), Some(FileFormat::Csv));
    assert_eq!(detect_format(Path::new("notes.txt")), Some(FileFormat::Txt));
    assert_eq!(detect_format(Path::new("readme.md")), Some(FileFormat::Txt));
    assert_eq!(detect_format(Path::new("unknown.bin")), None);
}

#[test]
fn test_chunk_overlap_produces_overlapping_text() {
    let dir = tempfile::tempdir().unwrap();
    // Create a long enough text that it will be split into multiple chunks
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(200);
    let path = dir.path().join("long.txt");
    std::fs::write(&path, &long_text).unwrap();

    let config = ChunkConfig {
        target_size: 200..=500,
        overlap_percent: 20,
    };
    let (chunks, _) = parse_file(&path, &config).unwrap();

    if chunks.len() >= 2 {
        // With overlap, the end of chunk[0] should appear at the start of chunk[1]
        let c0 = &chunks[0].text;
        let c1 = &chunks[1].text;

        // The overlap region from c0's tail should be a prefix of c1
        let overlap_len = c0.len() * 20 / 100;
        if overlap_len > 0 {
            let c0_tail = &c0[c0.len() - overlap_len..];
            assert!(
                c1.starts_with(c0_tail),
                "chunk 1 should start with the overlap from chunk 0"
            );
        }
    }
}

#[test]
fn test_fallback_chunking_on_empty_splitter() {
    // A single very long line with no natural break points
    let dir = tempfile::tempdir().unwrap();
    let long_word = "x".repeat(5000);
    let path = dir.path().join("nobreaks.txt");
    std::fs::write(&path, &long_word).unwrap();

    let config = ChunkConfig {
        target_size: 800..=2000,
        overlap_percent: 0,
    };
    let (chunks, warnings) = parse_file(&path, &config).unwrap();

    // Should produce chunks via fallback
    assert!(!chunks.is_empty(), "fallback should produce chunks");

    // Check if fallback warning was emitted (it may or may not be, depending
    // on whether text_splitter handles the case)
    // The important thing is that we got chunks back
    let total_len: usize = chunks.iter().map(|c| c.text.len()).sum();
    assert!(total_len >= long_word.len(), "all text should be captured");
}

#[test]
fn test_parse_directory_with_mixed_and_hidden() {
    let dir = tempfile::tempdir().unwrap();

    // Create visible files
    std::fs::write(dir.path().join("notes.txt"), "Some notes here.").unwrap();
    std::fs::write(dir.path().join("data.csv"), "col1,col2\nval1,val2\n").unwrap();
    std::fs::write(dir.path().join("unknown.bin"), "binary data").unwrap();

    // Create hidden file
    std::fs::write(dir.path().join(".secret.txt"), "secret notes").unwrap();

    // Create hidden directory with a file
    let hidden_dir = dir.path().join(".hidden");
    std::fs::create_dir(&hidden_dir).unwrap();
    std::fs::write(hidden_dir.join("inner.txt"), "inner notes").unwrap();

    // Create a subdirectory with a file
    let sub = dir.path().join("subdir");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("deep.txt"), "Deep content.").unwrap();

    let config = ChunkConfig::default();
    let (chunks, _) = parse_directory(dir.path(), &config).unwrap();

    // Should include: notes.txt, data.csv, subdir/deep.txt
    // Should exclude: unknown.bin, .secret.txt, .hidden/inner.txt
    let file_paths: Vec<&str> = chunks.iter().map(|c| c.source.file_path.as_str()).collect();

    assert!(
        file_paths.iter().any(|p| p.contains("notes.txt")),
        "should include notes.txt"
    );
    assert!(
        file_paths.iter().any(|p| p.contains("data.csv")),
        "should include data.csv"
    );
    assert!(
        file_paths.iter().any(|p| p.contains("deep.txt")),
        "should include deep.txt"
    );
    assert!(
        !file_paths.iter().any(|p| p.contains(".secret")),
        "should not include hidden files"
    );
    assert!(
        !file_paths.iter().any(|p| p.contains(".hidden")),
        "should not include files in hidden dirs"
    );
    assert!(
        !file_paths.iter().any(|p| p.contains("unknown.bin")),
        "should not include unsupported formats"
    );
}

#[test]
fn test_csv_headers_in_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.csv");
    std::fs::write(&path, "name,age,city\nAlice,30,NYC\nBob,25,LA\n").unwrap();

    let config = ChunkConfig::default();
    let (chunks, _) = parse_file(&path, &config).unwrap();

    assert_eq!(chunks.len(), 2);
    // Each chunk should have "header: value" format
    assert!(chunks[0].text.contains("name: Alice"));
    assert!(chunks[0].text.contains("age: 30"));
    assert!(chunks[0].text.contains("city: NYC"));
    assert!(chunks[1].text.contains("name: Bob"));
}

#[test]
fn test_empty_csv() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.csv");
    std::fs::write(&path, "col1,col2\n").unwrap();

    let config = ChunkConfig::default();
    let (chunks, _) = parse_file(&path, &config).unwrap();
    assert!(chunks.is_empty(), "CSV with only headers should produce no chunks");
}

#[test]
fn test_chunk_indices_are_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let text = "Paragraph one. ".repeat(100) + "\n\n" + &"Paragraph two. ".repeat(100);
    let path = dir.path().join("multi.txt");
    std::fs::write(&path, &text).unwrap();

    let config = ChunkConfig {
        target_size: 200..=500,
        overlap_percent: 0,
    };
    let (chunks, _) = parse_file(&path, &config).unwrap();

    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.source.chunk_index, i, "chunk indices should be sequential");
    }
}
