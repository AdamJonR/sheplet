use anyhow::{Context, Result};
use std::path::Path;

use crate::progress;
use crate::project::{require_init, project_dirs};

pub async fn run(sources: &Path, project: &Path) -> Result<()> {
    let _manifest = require_init(project)?;
    let dirs = project_dirs(project);

    // Step 1: Parse and chunk documents
    let pb = progress::spinner("Parsing documents...");
    let chunk_config = parser::ChunkConfig::default();
    let (chunks, warnings) = parser::parse_directory(sources, &chunk_config)
        .context("failed to parse source documents")?;
    pb.finish_with_message(format!("Parsed {} chunks from source documents", chunks.len()));

    for warning in &warnings {
        println!("  Warning: {}", warning);
    }

    if chunks.is_empty() {
        println!("No chunks extracted from source documents. Nothing to ingest.");
        return Ok(());
    }

    // Step 2: Load or download embedding model
    let pb = progress::spinner("Loading embedding model...");
    let embedding_model = embeddings::EmbeddingModel::download_and_load(&dirs.embeddings)
        .context("failed to load embedding model")?;
    pb.finish_with_message("Embedding model loaded.");

    // Step 3: Embed chunks
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let pb = progress::progress_bar(texts.len() as u64, "Embedding chunks...");
    let vectors = embedding_model
        .embed_batch(&texts)
        .context("failed to embed chunks")?;
    pb.finish_with_message("Embedding complete.");

    // Step 4: Store in LanceDB
    let pb = progress::spinner("Storing in vector database...");
    let store = db::VectorStore::open_or_create(
        &dirs.database,
        "chunks",
        embeddings::EMBEDDING_DIM,
    )
    .await
    .context("failed to open vector database")?;

    let records: Vec<db::ChunkRecord> = chunks
        .iter()
        .zip(vectors.iter())
        .map(|(chunk, vector)| db::ChunkRecord {
            vector: vector.clone(),
            text: chunk.text.clone(),
            source_file: chunk.source.file_path.clone(),
            chunk_index: chunk.source.chunk_index as u32,
        })
        .collect();

    store
        .insert(&records)
        .await
        .context("failed to insert records into vector database")?;
    let count = store.count().await?;
    pb.finish_with_message(format!("Stored {} chunks in vector database.", count));

    println!("Ingestion complete.");
    Ok(())
}
