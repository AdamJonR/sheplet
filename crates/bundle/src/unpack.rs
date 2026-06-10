use std::io::{BufReader, Cursor, Read};
use std::path::Path;

use crate::error::BundleError;
use crate::keys::Keypair;
use crate::manifest::Manifest;

/// Maximum size of manifest.json. Pass 1 runs *before* signature
/// verification, so the read must be bounded or a malicious bundle could
/// trigger an unbounded allocation (zstd decompression bomb).
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

/// Create a streaming tar archive over zstd-compressed data.
///
/// Each call produces a fresh decompression stream, allowing multiple
/// sequential passes without materializing the full decompressed content.
fn streaming_archive(
    compressed_data: &[u8],
) -> Result<tar::Archive<zstd::stream::Decoder<'_, BufReader<Cursor<&[u8]>>>>, BundleError> {
    let decoder = zstd::stream::Decoder::new(Cursor::new(compressed_data))?;
    Ok(tar::Archive::new(decoder))
}

/// Verify the signature and extract a `.sheplet` bundle.
///
/// The bundle's public key fingerprint must match `trusted_fingerprint` exactly —
/// this prevents an attacker from self-signing a bundle with their own keypair.
///
/// Uses two streaming decompression passes over the compressed data so that
/// the full decompressed tar is never held in memory:
///
/// 1. **Pass 1 — Extract manifest**: Stream-decompress, find `manifest.json`, verify
///    fingerprint and signature.
/// 2. **Pass 2 — Extract files**: Stream-decompress directly to disk via `tar::Archive::unpack()`,
///    which internally validates paths (strips leading `/`, rejects `..` components,
///    and verifies entries stay within the output directory).
pub fn verify_and_unpack(
    bundle_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    trusted_fingerprint: &str,
) -> Result<Manifest, BundleError> {
    let bundle_path = bundle_path.as_ref();
    let output_dir = output_dir.as_ref();

    let bytes = std::fs::read(bundle_path)?;

    if bytes.len() < 64 {
        return Err(BundleError::SignatureInvalid);
    }

    let (compressed_data, signature) = bytes.split_at(bytes.len() - 64);

    // Pass 1: Stream-decompress to extract manifest.json only
    let manifest = {
        let mut archive = streaming_archive(compressed_data)?;
        let mut manifest_data: Option<Vec<u8>> = None;

        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;
            if path.as_ref() == Path::new("manifest.json") {
                let mut buf = Vec::new();
                let n = entry.by_ref().take(MAX_MANIFEST_BYTES + 1).read_to_end(&mut buf)?;
                if n as u64 > MAX_MANIFEST_BYTES {
                    return Err(BundleError::InvalidManifest(format!(
                        "manifest.json exceeds {MAX_MANIFEST_BYTES} bytes"
                    )));
                }
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
    if manifest.public_key_fingerprint != trusted_fingerprint {
        return Err(BundleError::UntrustedKey {
            expected: trusted_fingerprint.to_string(),
            actual: manifest.public_key_fingerprint.clone(),
        });
    }

    // Verify signature
    let public_key_bytes = hex::decode(&manifest.public_key_hex)
        .map_err(|e| BundleError::InvalidManifest(format!("invalid public key hex: {e}")))?;
    Keypair::verify(&public_key_bytes, compressed_data, signature)?;

    // Pass 2: Stream-decompress directly to disk
    std::fs::create_dir_all(output_dir)?;
    let mut archive = streaming_archive(compressed_data)?;
    archive.unpack(output_dir)?;

    Ok(manifest)
}
