use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use bundle::Manifest;
use candle_core::Device;
use rag::{PhiGenerator, RagConfig, RagPipeline};
use serde::Serialize;

pub struct CourseMetadata {
    pub manifest: Manifest,
    pub config: RagConfig,
    pub course_dir: PathBuf,
}

pub struct LoadedCourse {
    pub metadata: CourseMetadata,
    pub pipeline: RagPipeline,
    pub generator: Mutex<PhiGenerator>,
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

        let manifest = bundle::verify_and_unpack(bundle_path, &course_dir)
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
            self.switch_course(&course_id).await?;
        }

        Ok(course_id)
    }

    pub async fn switch_course(&mut self, course_id: &str) -> Result<()> {
        // Move current active course back into known_courses before loading new one
        if let (Some(active), Some(active_id)) = (self.active.take(), self.active_id.take()) {
            self.known_courses.insert(active_id, active.metadata);
        }

        let metadata = self
            .known_courses
            .remove(course_id)
            .context("Course not found")?;

        let course_dir = &metadata.course_dir;

        let pipeline = RagPipeline::new(
            course_dir.join("embeddings"),
            course_dir.join("database"),
            metadata.config.clone(),
        )
        .await?;

        let adapter_path = course_dir.join("adapter.lora");
        let adapter = if adapter_path.exists() {
            Some(adapter_path.as_path())
        } else {
            None
        };

        let generator = PhiGenerator::load(course_dir.join("model"), adapter, &Device::Cpu)?;

        self.active = Some(LoadedCourse {
            metadata,
            pipeline,
            generator: Mutex::new(generator),
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

    pub fn active_mut(&mut self) -> Option<&mut LoadedCourse> {
        self.active.as_mut()
    }
}
