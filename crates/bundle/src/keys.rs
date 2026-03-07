use std::path::Path;

use ed25519_dalek::{Signer, Verifier};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::BundleError;

/// Ed25519 keypair for signing and verifying bundles.
pub struct Keypair {
    signing_key: ed25519_dalek::SigningKey,
}

#[derive(Serialize, Deserialize)]
struct StoredKeypair {
    secret_hex: String,
    public_hex: String,
}

impl Keypair {
    /// Generate a new random keypair.
    pub fn generate() -> Self {
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        Self { signing_key }
    }

    /// Load a keypair from a JSON file, or create and save a new one if the file doesn't exist.
    pub fn load_or_create(path: impl AsRef<Path>) -> Result<Self, BundleError> {
        let path = path.as_ref();

        if path.exists() {
            let contents = std::fs::read_to_string(path)?;
            let stored: StoredKeypair = serde_json::from_str(&contents)?;
            let secret_bytes = hex::decode(&stored.secret_hex)
                .map_err(|e| BundleError::InvalidManifest(format!("invalid secret hex: {e}")))?;
            let secret_array: [u8; 32] = secret_bytes.try_into().map_err(|_| {
                BundleError::InvalidManifest("secret key must be 32 bytes".to_string())
            })?;
            let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret_array);
            Ok(Self { signing_key })
        } else {
            let keypair = Self::generate();

            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let stored = StoredKeypair {
                secret_hex: hex::encode(keypair.signing_key.to_bytes()),
                public_hex: hex::encode(keypair.signing_key.verifying_key().to_bytes()),
            };
            let json = serde_json::to_string_pretty(&stored)?;
            std::fs::write(path, json)?;

            Ok(keypair)
        }
    }

    /// Sign data, returning the 64-byte signature as a `Vec<u8>`.
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(data);
        signature.to_bytes().to_vec()
    }

    /// Verify a signature against the given public key bytes and data.
    pub fn verify(public_key_bytes: &[u8], data: &[u8], signature: &[u8]) -> Result<(), BundleError> {
        let pk_array: [u8; 32] = public_key_bytes
            .try_into()
            .map_err(|_| BundleError::SignatureInvalid)?;
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_array)
            .map_err(|_| BundleError::SignatureInvalid)?;

        let sig_array: [u8; 64] = signature
            .try_into()
            .map_err(|_| BundleError::SignatureInvalid)?;
        let sig = ed25519_dalek::Signature::from_bytes(&sig_array);

        verifying_key
            .verify(data, &sig)
            .map_err(|_| BundleError::SignatureInvalid)
    }

    /// Return the raw public key bytes.
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.signing_key.verifying_key().to_bytes().to_vec()
    }

    /// SHA-256 fingerprint of the public key (first 16 hex chars).
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_key.verifying_key().to_bytes());
        let hash = hasher.finalize();
        hex::encode(hash)[..16].to_string()
    }

    /// Default path for the instructor keypair: `~/.sheplet-instructor/keypair.json`.
    pub fn default_keypair_path() -> Option<std::path::PathBuf> {
        dirs::home_dir().map(|h| h.join(".sheplet-instructor").join("keypair.json"))
    }
}
