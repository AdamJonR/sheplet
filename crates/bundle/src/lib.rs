pub mod error;
pub mod keys;
pub mod manifest;
pub mod pack;
pub mod unpack;

pub use error::BundleError;
pub use keys::Keypair;
pub use manifest::Manifest;
pub use pack::pack;
pub use unpack::verify_and_unpack;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_test_manifest(keypair: &Keypair) -> Manifest {
        Manifest {
            version: "1.0.0".to_string(),
            course_name: "Test Course".to_string(),
            model_name: "phi-4-mini".to_string(),
            quantization: "Q4_K_M".to_string(),
            build_timestamp: "2026-03-07T12:00:00Z".to_string(),
            public_key_hex: hex::encode(keypair.public_key_bytes()),
            public_key_fingerprint: keypair.fingerprint(),
        }
    }

    fn setup_project_dir(dir: &std::path::Path, keypair: &Keypair) {
        let manifest = make_test_manifest(keypair);
        fs::write(
            dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        fs::write(dir.join("config.json"), r#"{"temperature": 0.7}"#).unwrap();

        // Create optional directories with content
        let model_dir = dir.join("model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("weights.bin"), b"fake model weights").unwrap();

        let db_dir = dir.join("database");
        fs::create_dir_all(&db_dir).unwrap();
        fs::write(db_dir.join("vectors.lance"), b"fake vector data").unwrap();
    }

    #[test]
    fn keypair_generate_save_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let key_path = tmp.path().join("subdir").join("keypair.json");

        let kp1 = Keypair::load_or_create(&key_path).unwrap();
        assert!(key_path.exists());

        let kp2 = Keypair::load_or_create(&key_path).unwrap();
        assert_eq!(kp1.public_key_bytes(), kp2.public_key_bytes());
    }

    #[test]
    fn sign_and_verify() {
        let kp = Keypair::generate();
        let data = b"hello world";
        let sig = kp.sign(data);

        // Verification should succeed
        Keypair::verify(&kp.public_key_bytes(), data, &sig).unwrap();

        // Tampered data should fail
        let mut tampered = data.to_vec();
        tampered[0] ^= 0xFF;
        let result = Keypair::verify(&kp.public_key_bytes(), &tampered, &sig);
        assert!(matches!(result, Err(BundleError::SignatureInvalid)));
    }

    #[test]
    fn fingerprint_is_16_hex_chars() {
        let kp = Keypair::generate();
        let fp = kp.fingerprint();
        assert_eq!(fp.len(), 16);
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn pack_and_unpack_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        fs::create_dir_all(&project_dir).unwrap();

        let kp = Keypair::generate();
        setup_project_dir(&project_dir, &kp);

        let bundle_path = tmp.path().join("test.sheplet");
        pack(&project_dir, &bundle_path, &kp).unwrap();
        assert!(bundle_path.exists());

        let extract_dir = tmp.path().join("extracted");
        let manifest = verify_and_unpack(&bundle_path, &extract_dir, &kp.fingerprint()).unwrap();

        assert_eq!(manifest.course_name, "Test Course");
        assert_eq!(manifest.model_name, "phi-4-mini");

        // Verify files were extracted
        assert!(extract_dir.join("manifest.json").exists());
        assert!(extract_dir.join("config.json").exists());
        assert_eq!(
            fs::read(extract_dir.join("model").join("weights.bin")).unwrap(),
            b"fake model weights"
        );
        assert_eq!(
            fs::read(extract_dir.join("database").join("vectors.lance")).unwrap(),
            b"fake vector data"
        );
    }

    #[test]
    fn signature_failure_on_corrupted_bundle() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        fs::create_dir_all(&project_dir).unwrap();

        let kp = Keypair::generate();
        setup_project_dir(&project_dir, &kp);

        let bundle_path = tmp.path().join("test.sheplet");
        pack(&project_dir, &bundle_path, &kp).unwrap();

        // Corrupt one byte in the compressed data (not the signature)
        let mut bytes = fs::read(&bundle_path).unwrap();
        if bytes.len() > 65 {
            bytes[0] ^= 0xFF;
        }
        fs::write(&bundle_path, &bytes).unwrap();

        let extract_dir = tmp.path().join("extracted");
        let result = verify_and_unpack(&bundle_path, &extract_dir, &kp.fingerprint());
        assert!(result.is_err());
    }

    #[test]
    fn untrusted_key_error() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        fs::create_dir_all(&project_dir).unwrap();

        let kp_a = Keypair::generate();
        setup_project_dir(&project_dir, &kp_a);

        let bundle_path = tmp.path().join("test.sheplet");
        pack(&project_dir, &bundle_path, &kp_a).unwrap();

        let extract_dir = tmp.path().join("extracted");
        let kp_b = Keypair::generate();
        let result = verify_and_unpack(&bundle_path, &extract_dir, &kp_b.fingerprint());
        assert!(
            matches!(result, Err(BundleError::UntrustedKey { .. })),
            "expected UntrustedKey error, got: {:?}",
            result
        );
    }

    #[test]
    fn missing_manifest_error() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        fs::create_dir_all(&project_dir).unwrap();

        // Only create config.json, no manifest.json
        fs::write(project_dir.join("config.json"), "{}").unwrap();

        let kp = Keypair::generate();
        let bundle_path = tmp.path().join("test.sheplet");
        let result = pack(&project_dir, &bundle_path, &kp);
        assert!(matches!(result, Err(BundleError::MissingEntry(ref s)) if s == "manifest.json"));
    }

    #[test]
    fn manifest_serialization_roundtrip() {
        let kp = Keypair::generate();
        let manifest = make_test_manifest(&kp);
        let json = serde_json::to_string(&manifest).unwrap();
        let loaded: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.version, manifest.version);
        assert_eq!(loaded.course_name, manifest.course_name);
        assert_eq!(loaded.model_name, manifest.model_name);
        assert_eq!(loaded.quantization, manifest.quantization);
        assert_eq!(loaded.public_key_hex, manifest.public_key_hex);
        assert_eq!(loaded.public_key_fingerprint, manifest.public_key_fingerprint);
    }

    #[test]
    fn pack_with_empty_model_dir() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        fs::create_dir_all(&project_dir).unwrap();

        let kp = Keypair::generate();
        let manifest = make_test_manifest(&kp);
        fs::write(
            project_dir.join("manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        fs::write(project_dir.join("config.json"), "{}").unwrap();
        // Create model dir but leave it empty
        fs::create_dir_all(project_dir.join("model")).unwrap();

        let bundle_path = tmp.path().join("test.sheplet");
        // Should succeed — empty model dir is just an empty directory in the archive
        let result = pack(&project_dir, &bundle_path, &kp);
        assert!(result.is_ok(), "packing with empty model dir should work: {:?}", result.err());
    }

    #[test]
    fn missing_config_error() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        fs::create_dir_all(&project_dir).unwrap();

        // Only create manifest.json, no config.json
        let kp = Keypair::generate();
        let manifest = make_test_manifest(&kp);
        fs::write(
            project_dir.join("manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        let bundle_path = tmp.path().join("test.sheplet");
        let result = pack(&project_dir, &bundle_path, &kp);
        assert!(matches!(result, Err(BundleError::MissingEntry(ref s)) if s == "config.json"));
    }
}
