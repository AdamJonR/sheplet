use std::io::{Cursor, Read};
use std::path::Path;

use crate::error::BundleError;
use crate::keys::Keypair;
use crate::manifest::Manifest;

/// Verify the signature and extract a `.sheplet` bundle.
///
/// 1. Read the bundle file.
/// 2. Split into compressed data (all but last 64 bytes) and signature (last 64 bytes).
/// 3. Decompress to get tar bytes and extract `manifest.json` to read the public key.
/// 4. Verify the signature over the compressed data using the public key from the manifest.
/// 5. Extract all tar entries to `output_dir`.
/// 6. Return the parsed [`Manifest`].
pub fn verify_and_unpack(
    bundle_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
) -> Result<Manifest, BundleError> {
    let bundle_path = bundle_path.as_ref();
    let output_dir = output_dir.as_ref();

    let bytes = std::fs::read(bundle_path)?;

    if bytes.len() < 64 {
        return Err(BundleError::SignatureInvalid);
    }

    let (compressed_data, signature) = bytes.split_at(bytes.len() - 64);

    // Decompress to get tar bytes
    let tar_bytes = {
        let mut decoder = zstd::stream::Decoder::new(Cursor::new(compressed_data))?;
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;
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

    // Verify signature
    let public_key_bytes = hex::decode(&manifest.public_key_hex)
        .map_err(|e| BundleError::InvalidManifest(format!("invalid public key hex: {e}")))?;
    Keypair::verify(&public_key_bytes, compressed_data, signature)?;

    // Extract all entries to output_dir
    std::fs::create_dir_all(output_dir)?;
    let mut archive = tar::Archive::new(Cursor::new(&tar_bytes));
    archive.unpack(output_dir)?;

    Ok(manifest)
}
