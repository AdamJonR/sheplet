use std::path::{Path, PathBuf};

use crate::error::{EmbeddingsError, Result};

/// The Hugging Face model identifier for all-MiniLM-L6-v2.
const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Files required to load the embedding model.
const REQUIRED_FILES: &[&str] = &["model.safetensors", "config.json", "tokenizer.json"];

/// Download the all-MiniLM-L6-v2 model files from Hugging Face Hub.
///
/// Files are cached in `cache_dir`. Returns the directory containing the
/// downloaded files (which may be inside the hf-hub cache structure).
///
/// # Arguments
/// * `cache_dir` - Directory to use as the hf-hub cache root.
///
/// # Returns
/// The path to the directory containing `model.safetensors`, `config.json`,
/// and `tokenizer.json`. Since hf-hub stores files as symlinks in its cache,
/// we return the individual file paths via [`download_model_files`] instead.
pub fn download_model_files(cache_dir: impl AsRef<Path>) -> Result<DownloadedFiles> {
    let cache_dir = cache_dir.as_ref();
    std::fs::create_dir_all(cache_dir).map_err(|e| {
        EmbeddingsError::Download(format!("failed to create cache dir: {e}"))
    })?;

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .build()
        .map_err(|e| EmbeddingsError::Download(format!("failed to build HF API: {e}")))?;

    let repo = api.model(MODEL_REPO.to_string());

    let mut paths = Vec::with_capacity(REQUIRED_FILES.len());
    for &file_name in REQUIRED_FILES {
        let path = repo.get(file_name).map_err(|e| {
            EmbeddingsError::Download(format!("failed to download {file_name}: {e}"))
        })?;
        paths.push(path);
    }

    Ok(DownloadedFiles {
        model_safetensors: paths[0].clone(),
        config_json: paths[1].clone(),
        tokenizer_json: paths[2].clone(),
    })
}

/// Resolve model files from an existing HF Hub cache directory (no download).
///
/// Walks the cache structure to find the snapshot directory containing the model files.
pub fn resolve_cached_files(cache_dir: impl AsRef<Path>) -> Result<DownloadedFiles> {
    let cache_dir = cache_dir.as_ref();

    // HF cache layout: <cache_dir>/models--<org>--<name>/snapshots/<hash>/
    let models_dir = cache_dir.join(format!(
        "models--{}",
        MODEL_REPO.replace('/', "--")
    ));

    if !models_dir.is_dir() {
        return Err(EmbeddingsError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("HF cache model dir not found: {}", models_dir.display()),
        )));
    }

    // Read the current ref to find the snapshot hash
    let ref_path = models_dir.join("refs").join("main");
    let snapshot_hash = std::fs::read_to_string(&ref_path)
        .map_err(|e| EmbeddingsError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to read refs/main at {}: {e}", ref_path.display()),
        )))?;
    let snapshot_dir = models_dir.join("snapshots").join(snapshot_hash.trim());

    for &file_name in REQUIRED_FILES {
        let p = snapshot_dir.join(file_name);
        if !p.exists() {
            return Err(EmbeddingsError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("missing {file_name} in {}", snapshot_dir.display()),
            )));
        }
    }

    Ok(DownloadedFiles {
        model_safetensors: snapshot_dir.join("model.safetensors"),
        config_json: snapshot_dir.join("config.json"),
        tokenizer_json: snapshot_dir.join("tokenizer.json"),
    })
}

/// Paths to the downloaded model files.
#[derive(Debug, Clone)]
pub struct DownloadedFiles {
    pub model_safetensors: PathBuf,
    pub config_json: PathBuf,
    pub tokenizer_json: PathBuf,
}
