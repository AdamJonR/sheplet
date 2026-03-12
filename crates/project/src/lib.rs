use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub version: String,
    pub course_name: String,
    pub model_name: Option<String>,
    pub quantization: Option<String>,
    pub build_timestamp: Option<String>,
}

impl ProjectManifest {
    pub fn new(course_name: &str) -> Self {
        Self {
            version: "0.1.0".to_string(),
            course_name: course_name.to_string(),
            model_name: None,
            quantization: None,
            build_timestamp: None,
        }
    }

    pub fn load(project_dir: &Path) -> Result<Self> {
        let path = project_dir.join("manifest.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read manifest at {}", path.display()))?;
        let manifest: Self = serde_json::from_str(&content)?;
        Ok(manifest)
    }

    pub fn save(&self, project_dir: &Path) -> Result<()> {
        let path = project_dir.join("manifest.json");
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }

    pub fn bump_version(&mut self) {
        let parts: Vec<&str> = self.version.split('.').collect();
        if parts.len() == 3 {
            if let Ok(patch) = parts[2].parse::<u32>() {
                self.version = format!("{}.{}.{}", parts[0], parts[1], patch + 1);
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseConfig {
    pub system_prompt: String,
    pub retrieval_strategy: String,
    pub top_k: usize,
    pub relevance_threshold: f64,
    pub mmr_lambda: f32,
}

impl Default for CourseConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful tutor. Answer only from the provided course materials.".to_string(),
            retrieval_strategy: "top-k".to_string(),
            top_k: 5,
            relevance_threshold: 0.3,
            mmr_lambda: 0.5,
        }
    }
}

impl CourseConfig {
    pub fn load(project_dir: &Path) -> Result<Self> {
        let path = project_dir.join("config.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read config at {}", path.display()))?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self, project_dir: &Path) -> Result<()> {
        let path = project_dir.join("config.json");
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }
}

pub fn require_init(project_dir: &Path) -> Result<ProjectManifest> {
    if !project_dir.join("manifest.json").exists() {
        bail!(
            "Project directory {} is not initialized. Run `sheplet-instructor init` first.",
            project_dir.display()
        );
    }
    ProjectManifest::load(project_dir)
}

pub fn require_model(project_dir: &Path) -> Result<ProjectManifest> {
    let manifest = require_init(project_dir)?;
    if !project_dir.join("model").exists() || manifest.model_name.is_none() {
        bail!(
            "No model found in project. Run `sheplet-instructor model` first."
        );
    }
    Ok(manifest)
}

pub fn require_bundleable(project_dir: &Path) -> Result<ProjectManifest> {
    let manifest = require_init(project_dir)?;
    if !project_dir.join("config.json").exists() {
        bail!("No config.json found. Run `sheplet-instructor config` first.");
    }
    Ok(manifest)
}

pub fn project_dirs(project_dir: &Path) -> ProjectDirs {
    ProjectDirs {
        root: project_dir.to_path_buf(),
        model: project_dir.join("model"),
        embeddings: project_dir.join("embeddings"),
        database: project_dir.join("database"),
        finetune_data: project_dir.join("finetune_data"),
    }
}

pub struct ProjectDirs {
    pub root: PathBuf,
    pub model: PathBuf,
    pub embeddings: PathBuf,
    pub database: PathBuf,
    pub finetune_data: PathBuf,
}

/// Maps model shortcut names to local directory names under `downloaded-models/`.
pub fn local_model_source(name: &str) -> Option<&'static str> {
    match name {
        "gemma270m" => Some("gemma-3-transformers-gemma-3-270m-it-v1"),
        "gemma1b" => Some("gemma-3-transformers-gemma-3-1b-it-v1"),
        "llama-3.2-1b" | "llama1b" => Some("meta-llama--Llama-3.2-1B-Instruct"),
        "llama-3.2-3b" | "llama3b" => Some("meta-llama--Llama-3.2-3B-Instruct"),
        _ => None,
    }
}

/// Copy model files from a local directory into the project model dir.
///
/// Handles both single-file weights (`model.safetensors`) and sharded weights
/// (`model.safetensors.index.json` + shard files).
pub fn copy_local_model(src_dir: &Path, dest_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dest_dir)?;

    // Required files
    for filename in &["config.json", "tokenizer.json"] {
        let src = src_dir.join(filename);
        let dest = dest_dir.join(filename);
        std::fs::copy(&src, &dest)
            .with_context(|| format!("required file {} not found in {}", filename, src_dir.display()))?;
    }

    // Optional metadata files
    for filename in &["tokenizer_config.json", "generation_config.json"] {
        let src = src_dir.join(filename);
        if src.exists() {
            std::fs::copy(&src, dest_dir.join(filename))?;
        }
    }

    // Weights: single file or sharded
    let single = src_dir.join("model.safetensors");
    let index = src_dir.join("model.safetensors.index.json");

    if single.exists() {
        std::fs::copy(&single, dest_dir.join("model.safetensors"))
            .context("failed to copy model.safetensors")?;
    } else if index.exists() {
        // Copy index file
        std::fs::copy(&index, dest_dir.join("model.safetensors.index.json"))
            .context("failed to copy model.safetensors.index.json")?;

        // Parse index to find shard filenames
        let index_content = std::fs::read_to_string(&index)?;
        let index_json: serde_json::Value = serde_json::from_str(&index_content)?;
        let weight_map = index_json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("model.safetensors.index.json missing weight_map")?;

        // Collect unique shard filenames
        let mut shards: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shards.sort();
        shards.dedup();

        for shard in &shards {
            let src = src_dir.join(shard);
            std::fs::copy(&src, dest_dir.join(shard))
                .with_context(|| format!("failed to copy shard {}", shard))?;
        }
    } else {
        bail!(
            "no model weights found in {} — expected model.safetensors or model.safetensors.index.json",
            src_dir.display()
        );
    }

    Ok(())
}

/// Returns true if the model name refers to a Gemma architecture.
///
/// Checks known shortcut names and HF repo ID patterns containing "gemma".
pub fn is_gemma_model(name: &str) -> bool {
    matches!(name, "gemma270m" | "gemma1b" | "gemma-3-1b-it") || name.contains("/gemma")
}

/// Returns true if the model name refers to a Llama architecture.
///
/// Checks known shortcut names and HF repo ID patterns containing "llama" or "Llama".
pub fn is_llama_model(name: &str) -> bool {
    matches!(name, "llama-3.2-1b" | "llama1b" | "llama-3.2-3b" | "llama3b")
        || name.contains("/Llama")
        || name.contains("/llama")
}

/// Generate a Unix timestamp (seconds since epoch) as a string.
pub fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = ProjectManifest {
            version: "1.2.3".to_string(),
            course_name: "Biology 101".to_string(),
            model_name: Some("phi-4-mini".to_string()),
            quantization: Some("Q4_K_M".to_string()),
            build_timestamp: Some("1234567890".to_string()),
        };
        manifest.save(dir.path()).unwrap();
        let loaded = ProjectManifest::load(dir.path()).unwrap();
        assert_eq!(loaded.version, "1.2.3");
        assert_eq!(loaded.course_name, "Biology 101");
        assert_eq!(loaded.model_name.as_deref(), Some("phi-4-mini"));
        assert_eq!(loaded.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(loaded.build_timestamp.as_deref(), Some("1234567890"));
    }

    #[test]
    fn test_config_default_values() {
        let config = CourseConfig::default();
        assert_eq!(config.retrieval_strategy, "top-k");
        assert_eq!(config.top_k, 5);
        assert!((config.relevance_threshold - 0.3).abs() < 1e-10);
        assert!((config.mmr_lambda - 0.5).abs() < 1e-6);
        assert!(!config.system_prompt.is_empty());
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let config = CourseConfig {
            system_prompt: "Custom prompt".to_string(),
            retrieval_strategy: "mmr".to_string(),
            top_k: 10,
            relevance_threshold: 0.5,
            mmr_lambda: 0.7,
        };
        config.save(dir.path()).unwrap();
        let loaded = CourseConfig::load(dir.path()).unwrap();
        assert_eq!(loaded.system_prompt, "Custom prompt");
        assert_eq!(loaded.retrieval_strategy, "mmr");
        assert_eq!(loaded.top_k, 10);
    }

    #[test]
    fn test_require_init_fails_on_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let result = require_init(dir.path());
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not initialized"));
    }

    #[test]
    fn test_project_dirs_paths() {
        let base = Path::new("/tmp/test_project");
        let dirs = project_dirs(base);
        assert_eq!(dirs.root, base);
        assert_eq!(dirs.model, base.join("model"));
        assert_eq!(dirs.embeddings, base.join("embeddings"));
        assert_eq!(dirs.database, base.join("database"));
        assert_eq!(dirs.finetune_data, base.join("finetune_data"));
    }

    #[test]
    fn test_manifest_bump_version() {
        let mut manifest = ProjectManifest::new("Test");
        assert_eq!(manifest.version, "0.1.0");
        manifest.bump_version();
        assert_eq!(manifest.version, "0.1.1");
        manifest.bump_version();
        assert_eq!(manifest.version, "0.1.2");
    }
}
