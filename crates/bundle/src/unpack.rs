use std::io::{Cursor, Read};
use std::path::Path;

use crate::error::BundleError;
use crate::keys::Keypair;
use crate::manifest::Manifest;

/// Verify the signature and extract a `.sheplet` bundle.
///
/// If `trusted_fingerprint` is `Some`, the bundle's public key fingerprint
/// must match exactly — this prevents an attacker from self-signing a bundle
/// with their own keypair. If `None`, the public key from the manifest is
/// trusted (backwards-compatible but insecure).
///
/// Steps:
/// 1. Read the bundle file.
/// 2. Split into compressed data (all but last 64 bytes) and signature (last 64 bytes).
/// 3. Decompress to get tar bytes and extract `manifest.json` to read the public key.
/// 4. Optionally verify the public key fingerprint against a trusted value.
/// 5. Verify the signature over the compressed data using the public key from the manifest.
/// 6. Validate tar entry paths to prevent path traversal.
/// 7. Extract all tar entries to `output_dir`.
/// 8. Return the parsed [`Manifest`].
pub fn verify_and_unpack(
    bundle_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<Manifest, BundleError> {
    verify_and_unpack_with_trust(bundle_path, output_dir, None)
}

/// Like [`verify_and_unpack`] but requires the bundle's public key fingerprint
/// to match `trusted_fingerprint`.
pub fn verify_and_unpack_trusted(
    bundle_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    trusted_fingerprint: &str,
) -> Result<Manifest, BundleError> {
    verify_and_unpack_with_trust(bundle_path, output_dir, Some(trusted_fingerprint))
}

fn verify_and_unpack_with_trust(
    bundle_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    trusted_fingerprint: Option<&str>,
) -> Result<Manifest, BundleError> {
    let bundle_path = bundle_path.as_ref();
    let output_dir = output_dir.as_ref();

    let bytes = std::fs::read(bundle_path)?;

    if bytes.len() < 64 {
        return Err(BundleError::SignatureInvalid);
    }

    let (compressed_data, signature) = bytes.split_at(bytes.len() - 64);

    // Decompress with size limit (4 GB)
    const MAX_DECOMPRESSED_SIZE: usize = 4 * 1024 * 1024 * 1024;
    let tar_bytes = {
        let decoder = zstd::stream::Decoder::new(Cursor::new(compressed_data))?;
        let mut buf = Vec::new();
        let n = decoder
            .take(MAX_DECOMPRESSED_SIZE as u64 + 1)
            .read_to_end(&mut buf)?;
        if n > MAX_DECOMPRESSED_SIZE {
            return Err(BundleError::InvalidManifest(
                "decompressed bundle exceeds 4 GB size limit".to_string(),
            ));
        }
        buf
    };

    // Extract manifest.json from the tar to read the public key
    let manifest = {
        let mut archive = tar::Archive::new(Cursor::new(&tar_bytes));
        let mut manifest_data: Option<Vec<u8>> = None;

        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;
            if path.as_ref() == Path::new("manifest.json") {
                let mut buf = Vec::new();
                entry.read_to_end(&mut buf)?;
                manifest_data = Some(buf);
                break;
            }
        }

        let data = manifest_data
            .ok_or_else(|| BundleError::MissingEntry("manifest.json".to_string()))?;
        let manifest: Manifest = serde_json::from_slice(&data)?;
        manifest
    };

    // Verify the public key fingerprint against trusted value
    if let Some(expected) = trusted_fingerprint {
        if manifest.public_key_fingerprint != expected {
            return Err(BundleError::UntrustedKey {
                expected: expected.to_string(),
                actual: manifest.public_key_fingerprint.clone(),
            });
        }
    }

    // Verify signature
    let public_key_bytes = hex::decode(&manifest.public_key_hex)
        .map_err(|e| BundleError::InvalidManifest(format!("invalid public key hex: {e}")))?;
    Keypair::verify(&public_key_bytes, compressed_data, signature)?;

    // Validate tar entry paths before extraction (prevent path traversal)
    std::fs::create_dir_all(output_dir)?;
    let canonical_output = output_dir.canonicalize()?;
    {
        let mut archive = tar::Archive::new(Cursor::new(&tar_bytes));
        for entry in archive.entries()? {
            let entry = entry?;
            let entry_path = entry.path()?;
            // Reject absolute paths and paths with ".." components
            if entry_path.is_absolute() {
                return Err(BundleError::InvalidManifest(format!(
                    "tar entry has absolute path: {}",
                    entry_path.display()
                )));
            }
            for component in entry_path.components() {
                if matches!(component, std::path::Component::ParentDir) {
                    return Err(BundleError::InvalidManifest(format!(
                        "tar entry contains path traversal: {}",
                        entry_path.display()
                    )));
                }
            }
            // Verify resolved path stays within output_dir
            let resolved = canonical_output.join(&entry_path);
            if !resolved.starts_with(&canonical_output) {
                return Err(BundleError::InvalidManifest(format!(
                    "tar entry escapes output directory: {}",
                    entry_path.display()
                )));
            }
        }
    }

    // Extract all entries to output_dir
    let mut archive = tar::Archive::new(Cursor::new(&tar_bytes));
    archive.unpack(output_dir)?;

    Ok(manifest)
}
