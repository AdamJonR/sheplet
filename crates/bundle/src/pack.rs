use std::io::Write;
use std::path::Path;

use crate::error::BundleError;
use crate::keys::Keypair;

/// Pack a project directory into a `.sheplet` bundle.
///
/// Bundle format: `[zstd-compressed tar bytes][64-byte Ed25519 signature]`
///
/// The signature covers the compressed bytes (not the raw tar).
///
/// Required entries in `project_dir`:
/// - `manifest.json`
/// - `config.json`
///
/// Optional entries (included if present):
/// - `model/` directory
/// - `embeddings/` directory
/// - `database/` directory
/// - `adapter.safetensors` file
pub fn pack(
    project_dir: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    keypair: &Keypair,
) -> Result<(), BundleError> {
    let project_dir = project_dir.as_ref();
    let output_path = output_path.as_ref();

    // Verify required files exist
    let manifest_path = project_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(BundleError::MissingEntry("manifest.json".to_string()));
    }
    let config_path = project_dir.join("config.json");
    if !config_path.exists() {
        return Err(BundleError::MissingEntry("config.json".to_string()));
    }

    // Build tar archive in memory
    let tar_bytes = {
        let mut builder = tar::Builder::new(Vec::new());

        // Add required files
        builder.append_path_with_name(&manifest_path, "manifest.json")?;
        builder.append_path_with_name(&config_path, "config.json")?;

        // Add optional directories
        for dir_name in &["model", "embeddings", "database"] {
            let dir_path = project_dir.join(dir_name);
            if dir_path.is_dir() {
                builder.append_dir_all(*dir_name, &dir_path)?;
            }
        }

        // Add optional adapter file
        let adapter_path = project_dir.join("adapter.safetensors");
        if adapter_path.exists() {
            builder.append_path_with_name(&adapter_path, "adapter.safetensors")?;
        }

        builder.into_inner()?
    };

    // Compress with zstd
    let compressed = {
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3)?;
        encoder.write_all(&tar_bytes)?;
        encoder.finish()?
    };

    // Sign the compressed bytes
    let signature = keypair.sign(&compressed);

    // Write compressed data + 64-byte signature
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::fs::File::create(output_path)?;
    out.write_all(&compressed)?;
    out.write_all(&signature)?;

    Ok(())
}
