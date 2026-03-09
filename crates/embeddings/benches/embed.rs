use criterion::{Criterion, criterion_group, criterion_main};
use std::path::PathBuf;

fn embed_benchmarks(c: &mut Criterion) {
    let models_dir = match std::env::var("SHEPLET_BENCH_MODELS_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => {
            eprintln!(
                "SHEPLET_BENCH_MODELS_DIR not set — skipping embedding benchmarks. \
                 Set it to a directory containing embeddings/ with the all-MiniLM-L6-v2 model."
            );
            return;
        }
    };

    let model_path = models_dir.join("embeddings");
    let device = candle_core::Device::Cpu;
    let model = match embeddings::EmbeddingModel::from_local(&model_path, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "Failed to load embedding model from {:?}: {e} — skipping benchmarks.",
                model_path
            );
            return;
        }
    };

    let short_text = "What is photosynthesis?";
    let long_text = "Photosynthesis is the biological process by which green plants and certain \
        other organisms convert light energy into chemical energy. During this process, plants \
        absorb carbon dioxide from the atmosphere and water from the soil, using sunlight as the \
        energy source to produce glucose and oxygen as byproducts.";

    let mut group = c.benchmark_group("embed");
    group.sample_size(50);

    group.bench_function("embed_short", |b| {
        b.iter(|| model.embed(short_text).unwrap());
    });

    group.bench_function("embed_long", |b| {
        b.iter(|| model.embed(long_text).unwrap());
    });

    let batch: Vec<&str> = vec![
        short_text,
        "Explain the concept of natural selection.",
        "What are the main causes of the French Revolution?",
        "How does cellular respiration produce ATP?",
        "Describe the structure of DNA.",
        "What is the difference between mitosis and meiosis?",
        "Define the term electronegativity.",
        "How do tectonic plates cause earthquakes?",
        "What role does RNA play in protein synthesis?",
        "Explain the water cycle and its importance.",
    ];
    group.bench_function("embed_batch_10", |b| {
        b.iter(|| model.embed_batch(&batch).unwrap());
    });

    group.finish();
}

criterion_group!(benches, embed_benchmarks);
criterion_main!(benches);
