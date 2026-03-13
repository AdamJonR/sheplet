use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use bundle::Manifest;
use rag::{PhiGenerator, RagConfig, RagPipeline};
use serde::Serialize;
use tokio::sync::RwLock;

pub struct CourseMetadata {
    pub manifest: Manifest,
    pub config: RagConfig,
    pub course_dir: PathBuf,
}

pub struct LoadedCourse {
    pub metadata: CourseMetadata,
    pub pipeline: Arc<RwLock<RagPipeline>>,
    pub generator: Arc<Mutex<PhiGenerator>>,
}

pub struct CourseManager {
    pub known_courses: HashMap<String, CourseMetadata>,
    pub active: Option<LoadedCourse>,
    pub active_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CourseInfo {
    pub id: String,
    pub course_name: String,
    pub version: String,
    pub model_name: String,
    pub quantization: String,
    pub is_active: bool,
}

impl CourseManager {
    pub fn new() -> Self {
        Self {
            known_courses: HashMap::new(),
            active: None,
            active_id: None,
        }
    }

    pub async fn load_bundle(
        &mut self,
        bundle_path: impl AsRef<Path>,
        base_dir: impl AsRef<Path>,
        trusted_fingerprint: &str,
        no_adapter: bool,
    ) -> Result<String> {
        let bundle_path = bundle_path.as_ref();
        let base_dir = base_dir.as_ref();

        // Derive course_id from bundle filename
        let course_id = bundle_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("course")
            .to_string();

        let course_dir = base_dir.join("courses").join(&course_id);
        std::fs::create_dir_all(&course_dir)?;

        let manifest = bundle::verify_and_unpack(bundle_path, &course_dir, trusted_fingerprint)
            .context("Failed to verify and unpack bundle")?;

        // Read config.json from extracted bundle
        let config_path = course_dir.join("config.json");
        let config: RagConfig = if config_path.exists() {
            let data = std::fs::read_to_string(&config_path)?;
            serde_json::from_str(&data)?
        } else {
            RagConfig::default()
        };

        let metadata = CourseMetadata {
            manifest,
            config,
            course_dir,
        };

        let is_first = self.known_courses.is_empty();
        self.known_courses.insert(course_id.clone(), metadata);

        // Auto-switch if this is the first/only course
        if is_first {
            self.switch_course_inner(&course_id, no_adapter).await?;
        }

        Ok(course_id)
    }

    pub async fn switch_course(&mut self, course_id: &str) -> Result<()> {
        self.switch_course_inner(course_id, false).await
    }

    pub async fn switch_course_no_adapter(&mut self, course_id: &str, no_adapter: bool) -> Result<()> {
        self.switch_course_inner(course_id, no_adapter).await
    }

    async fn switch_course_inner(&mut self, course_id: &str, no_adapter: bool) -> Result<()> {
        // Move current active course back into known_courses before loading new one
        if let (Some(active), Some(active_id)) = (self.active.take(), self.active_id.take()) {
            self.known_courses.insert(active_id, active.metadata);
        }

        let metadata = self
            .known_courses
            .remove(course_id)
            .context("Course not found")?;

        let course_dir = &metadata.course_dir;

        let model_arch = rag::detect_model_arch(course_dir.join("model"))
            .unwrap_or(rag::ModelArch::Phi3);

        let embedding_device = compute::device_for(compute::Workload::Embedding);
        let pipeline = RagPipeline::new(
            course_dir.join("embeddings"),
            course_dir.join("database"),
            metadata.config.clone(),
            model_arch,
            &embedding_device,
        )
        .await?;

        let adapter_path = course_dir.join("adapter.safetensors");
        let adapter = if no_adapter {
            println!("[debug] --no-adapter flag set: skipping LoRA adapter, using base model only");
            None
        } else if adapter_path.exists() {
            Some(adapter_path.as_path())
        } else {
            None
        };

        let device = compute::device_for(compute::Workload::Inference);
        let generator = PhiGenerator::load(course_dir.join("model"), adapter, &device)?;

        self.active = Some(LoadedCourse {
            metadata,
            pipeline: Arc::new(RwLock::new(pipeline)),
            generator: Arc::new(Mutex::new(generator)),
        });
        self.active_id = Some(course_id.to_string());

        Ok(())
    }

    pub fn list_courses(&self) -> Vec<CourseInfo> {
        let mut courses: Vec<CourseInfo> = self
            .known_courses
            .iter()
            .map(|(id, meta)| CourseInfo {
                id: id.clone(),
                course_name: meta.manifest.course_name.clone(),
                version: meta.manifest.version.clone(),
                model_name: meta.manifest.model_name.clone(),
                quantization: meta.manifest.quantization.clone(),
                is_active: self.active_id.as_deref() == Some(id),
            })
            .collect();

        // Also include the active course's metadata
        if let (Some(active), Some(active_id)) = (&self.active, &self.active_id) {
            if !courses.iter().any(|c| c.id == *active_id) {
                courses.push(CourseInfo {
                    id: active_id.clone(),
                    course_name: active.metadata.manifest.course_name.clone(),
                    version: active.metadata.manifest.version.clone(),
                    model_name: active.metadata.manifest.model_name.clone(),
                    quantization: active.metadata.manifest.quantization.clone(),
                    is_active: true,
                });
            }
        }

        courses
    }

    pub fn active(&self) -> Option<&LoadedCourse> {
        self.active.as_ref()
    }
}
