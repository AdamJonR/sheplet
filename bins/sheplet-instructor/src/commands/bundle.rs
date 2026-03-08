use anyhow::{Context, Result};
use std::path::Path;

use crate::progress;
use crate::project::{self, require_bundleable};

pub fn run(project: &Path, output: &Path, bump_version: bool) -> Result<()> {
    let mut manifest = require_bundleable(project)?;

    if bump_version {
        manifest.bump_version();
        manifest.save(project)?;
        println!("Version bumped to {}", manifest.version);
    }

    // Update build timestamp
    manifest.build_timestamp = Some(project::timestamp());
    manifest.save(project)?;

    // Load keypair
    let keypair_path = bundle::keys::Keypair::default_keypair_path()
        .context("could not determine keypair path")?;
    let keypair = bundle::keys::Keypair::load_or_create(&keypair_path)?;

    // Update manifest with public key info
    let pub_key_hex = hex::encode(keypair.public_key_bytes());
    let fingerprint = keypair.fingerprint();

    // Write bundle manifest (different from project manifest)
    let bundle_manifest = bundle::manifest::Manifest {
        version: manifest.version.clone(),
        course_name: manifest.course_name.clone(),
        model_name: manifest.model_name.clone().unwrap_or_default(),
        quantization: manifest.quantization.clone().unwrap_or_default(),
        build_timestamp: manifest.build_timestamp.clone().unwrap_or_default(),
        public_key_hex: pub_key_hex,
        public_key_fingerprint: fingerprint.clone(),
    };

    // Write the bundle manifest into the project dir temporarily
    let manifest_content = serde_json::to_string_pretty(&bundle_manifest)?;
    std::fs::write(project.join("manifest.json"), &manifest_content)?;

    // Pack the bundle
    let pb = progress::spinner("Packaging bundle...");
    bundle::pack::pack(project, output, &keypair)
        .context("failed to pack bundle")?;
    pb.finish_with_message("Bundle packaged.");

    println!("Bundle created: {}", output.display());
    println!("  Course: {}", manifest.course_name);
    println!("  Version: {}", manifest.version);
    println!("  Fingerprint: {}", fingerprint);

    Ok(())
}

