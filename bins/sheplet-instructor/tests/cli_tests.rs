use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

fn cmd() -> Command {
    Command::cargo_bin("sheplet-instructor").unwrap()
}

#[test]
fn test_help_flag() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage"));
}

#[test]
fn test_init_creates_project_structure() {
    let dir = tempfile::tempdir().unwrap();
    let project_dir = dir.path().join("my_course");

    cmd()
        .args(["init", "--course", "Biology 101", "--output"])
        .arg(&project_dir)
        .assert()
        .success();

    assert!(project_dir.join("manifest.json").exists());
    assert!(project_dir.join("config.json").exists());
}

#[test]
fn test_init_rejects_existing_project() {
    let dir = tempfile::tempdir().unwrap();
    let project_dir = dir.path().join("my_course");

    // First init succeeds
    cmd()
        .args(["init", "--course", "Biology 101", "--output"])
        .arg(&project_dir)
        .assert()
        .success();

    // Second init should fail (project already exists)
    cmd()
        .args(["init", "--course", "Biology 101", "--output"])
        .arg(&project_dir)
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exists"));
}

#[test]
fn test_ingest_requires_init() {
    let dir = tempfile::tempdir().unwrap();
    let uninit_dir = dir.path().join("uninit");
    fs::create_dir_all(&uninit_dir).unwrap();

    cmd()
        .args(["ingest", "--sources", ".", "--project"])
        .arg(&uninit_dir)
        .assert()
        .failure();
}

#[test]
fn test_config_view_after_init() {
    let dir = tempfile::tempdir().unwrap();
    let project_dir = dir.path().join("my_course");

    cmd()
        .args(["init", "--course", "Test Course", "--output"])
        .arg(&project_dir)
        .assert()
        .success();

    cmd()
        .args(["config", "--project"])
        .arg(&project_dir)
        .assert()
        .success();
}

#[test]
fn test_bundle_succeeds_after_init() {
    let dir = tempfile::tempdir().unwrap();
    let project_dir = dir.path().join("my_course");

    cmd()
        .args(["init", "--course", "Test Course", "--output"])
        .arg(&project_dir)
        .assert()
        .success();

    let bundle_path = dir.path().join("out.sheplet");
    cmd()
        .args(["bundle", "--project"])
        .arg(&project_dir)
        .args(["--output"])
        .arg(&bundle_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("Bundle created"));

    assert!(bundle_path.exists());
}
