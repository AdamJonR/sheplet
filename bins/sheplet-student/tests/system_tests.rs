mod test_data;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use conversations::ConversationStore;
use http_body_util::BodyExt;
use rag::TextGenerator;
use serde_json::Value;
use tokio::sync::RwLock;
use tower::ServiceExt;

use sheplet_student::app_state::AppState;
use sheplet_student::course::CourseManager;
use sheplet_student::server;

use test_data::*;

/// Full end-to-end pipeline: instructor CLI → student server → chat.
///
/// Exercises: init, ingest, model download, finetune (DPO), config, bundle,
/// then loads the bundle in the student app and runs a chat query.
#[tokio::test]
#[ignore]
async fn test_full_pipeline_test_model() {
    if !test_model_available() {
        eprintln!("SKIP: test_model model not found in downloaded-models/");
        return;
    }

    let pipeline = run_instructor_pipeline("System Test", "test-course.sheplet");

    // --- Student: load bundle via tower ---
    println!("=== Student: Load Bundle ===");
    let student_dir = pipeline._tmpdir.path().join("student-data");
    std::fs::create_dir_all(&student_dir).unwrap();

    let conversations_path = student_dir.join("conversations");
    let conversations = Arc::new(ConversationStore::open(&conversations_path).unwrap());

    let state = Arc::new(AppState {
        courses: RwLock::new(CourseManager::new()),
        conversations,
        base_dir: student_dir,
        no_adapter: false,
    });

    let app = server::build_router(state.clone());

    let load_body = serde_json::json!({
        "path": pipeline.bundle_path.to_str().unwrap(),
        "trusted_fingerprint": pipeline.fingerprint,
    });

    let start = std::time::Instant::now();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/bundles/load")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&load_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK, "Bundle load should succeed");
    println!("  Bundle loaded in {:?}", start.elapsed());

    // --- Chat ---
    println!("=== Student: Chat ===");
    let chat_body = serde_json::json!({
        "message": "What is a cell?",
        "max_tokens": 64,
    });

    let start = std::time::Instant::now();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/sync")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&chat_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK, "Chat should succeed");
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let chat_response: Value = serde_json::from_slice(&body_bytes).unwrap();
    println!("  Chat completed in {:?}", start.elapsed());
    println!(
        "  Response: {}",
        chat_response["response"].as_str().unwrap_or("<none>")
    );

    let response_text = chat_response["response"].as_str().unwrap_or("");
    assert!(!response_text.is_empty(), "Response should not be empty");
    assert!(
        !chat_response["blocked"].as_bool().unwrap_or(true),
        "Response should not be blocked"
    );

    // --- Verify active course ---
    println!("=== Student: Verify Active Course ===");
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/courses/active")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let active_course: Value = serde_json::from_slice(&body_bytes).unwrap();
    let course_name = active_course["course"]["course_name"]
        .as_str()
        .unwrap_or("");
    println!("  Active course: {course_name}");
    assert_eq!(course_name, "System Test");

    println!("\n=== Full pipeline test PASSED ===");
}

/// Loads the Gemma 270M model WITHOUT LoRA adapter and verifies it generates
/// non-empty, non-degenerate output. This isolates QK-norm correctness from
/// LoRA merge correctness.
#[tokio::test]
#[ignore]
async fn test_test_model_base_inference() {
    if !test_model_available() {
        eprintln!("SKIP: test_model model not found in downloaded-models/");
        return;
    }

    let model_dir = workspace_root()
        .join("downloaded-models")
        .join(MODEL_DIR_NAME);

    let device = compute::device_for(compute::Workload::Inference);
    println!("Device: {:?}", device);

    let mut generator =
        rag::PhiGenerator::load(&model_dir, None, &device).expect("Failed to load model");

    let prompt = "<start_of_turn>user\nWhat is biology?<end_of_turn>\n<start_of_turn>model\n";
    let output = generator
        .generate(prompt, 32)
        .expect("Generation failed");

    println!("Output: {output}");

    assert!(!output.is_empty(), "Output should not be empty");

    // Verify tokens are not all the same (degenerate output)
    let words: Vec<&str> = output.split_whitespace().collect();
    if words.len() >= 3 {
        let all_same = words.windows(2).all(|w| w[0] == w[1]);
        assert!(!all_same, "Output should not be all the same token repeated");
    }

    println!("\n=== Base inference test PASSED ===");
}

/// Isolates generation performance from the RAG pipeline.
/// Loads the model directly and times token generation.
#[tokio::test]
#[ignore]
async fn test_test_model_generation_performance() {
    if !test_model_available() {
        eprintln!("SKIP: test_model model not found in downloaded-models/");
        return;
    }

    let model_dir = workspace_root()
        .join("downloaded-models")
        .join(MODEL_DIR_NAME);

    let device = compute::device_for(compute::Workload::Inference);
    println!("Device: {:?}", device);

    // Time model loading
    let start = std::time::Instant::now();
    let mut generator =
        rag::PhiGenerator::load(&model_dir, None, &device).expect("Failed to load model");
    let load_time = start.elapsed();
    println!("Model load time: {:?}", load_time);

    // Time generation
    let prompt = "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n";
    let max_tokens: usize = 32;

    let start = std::time::Instant::now();
    let output = generator
        .generate(prompt, max_tokens)
        .expect("Generation failed");
    let gen_time = start.elapsed();

    let approx_tokens = output.split_whitespace().count();
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        approx_tokens as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    println!("Generation time: {:?}", gen_time);
    println!("Output ({approx_tokens} approx tokens): {output}");
    println!("Approx tokens/sec: {tokens_per_sec:.1}");

    assert!(
        gen_time.as_secs() < 60,
        "Generation took too long: {:?}",
        gen_time
    );
    assert!(!output.is_empty(), "Output should not be empty");

    println!("\n=== Generation performance test PASSED ===");
}

/// Tests just the instructor CLI pipeline (no student model loading).
/// Verifies that init → ingest → model → finetune → config → bundle works.
#[tokio::test]
#[ignore]
async fn test_instructor_pipeline_only() {
    if !test_model_available() {
        eprintln!("SKIP: test_model model not found in downloaded-models/");
        return;
    }

    let pipeline = run_instructor_pipeline("Pipeline Test", "pipeline-test.sheplet");
    println!("  Bundle: {}", pipeline.bundle_path.display());
    println!("\n=== Instructor pipeline test PASSED ===");
}
