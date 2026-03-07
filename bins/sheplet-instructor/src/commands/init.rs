use anyhow::{bail, Result};
use std::path::Path;

use crate::project::{CourseConfig, ProjectManifest, project_dirs};

pub fn run(course: &str, output: &Path) -> Result<()> {
    if output.join("manifest.json").exists() {
        bail!("Project already exists at {}", output.display());
    }

    // Create project directory structure
    let dirs = project_dirs(output);
    std::fs::create_dir_all(&dirs.root)?;
    std::fs::create_dir_all(&dirs.model)?;
    std::fs::create_dir_all(&dirs.embeddings)?;
    std::fs::create_dir_all(&dirs.database)?;
    std::fs::create_dir_all(&dirs.finetune_data)?;

    // Write manifest
    let manifest = ProjectManifest::new(course);
    manifest.save(&dirs.root)?;

    // Write default config
    let config = CourseConfig::default();
    config.save(&dirs.root)?;

    // Ensure keypair exists
    if let Some(keypair_path) = bundle::keys::Keypair::default_keypair_path() {
        let keypair = bundle::keys::Keypair::load_or_create(&keypair_path)?;
        println!("Public key fingerprint: {}", keypair.fingerprint());
    }

    println!("Initialized course project: \"{}\"", course);
    println!("Project directory: {}", output.display());
    Ok(())
}
