use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
            "Project directory {} is not initialized.",
            project_dir.display()
        );
    }
    ProjectManifest::load(project_dir)
}

pub fn require_model(project_dir: &Path) -> Result<ProjectManifest> {
    let manifest = require_init(project_dir)?;
    if !project_dir.join("model").exists() || manifest.model_name.is_none() {
        bail!("No model found in project.");
    }
    Ok(manifest)
}

pub fn require_bundleable(project_dir: &Path) -> Result<ProjectManifest> {
    let manifest = require_init(project_dir)?;
    if !project_dir.join("config.json").exists() {
        bail!("No config.json found.");
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
