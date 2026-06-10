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

    let short_text = "When was Rome founded?";
    let long_text = "Rome was traditionally founded in 753 BC, a date calculated by the Roman \
        scholar Marcus Terentius Varro. According to legend, the city was established by Romulus, \
        who became its first king and gave the city its name. The city was built on seven hills \
        overlooking the Tiber River, which provided fresh water, a route for trade, and a natural \
        defensive barrier for the growing settlement.";

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
        "What was the Roman Senate?",
        "What are the main causes of the fall of the Republic?",
        "How did Roman aqueducts move water?",
        "Describe the structure of the cursus honorum.",
        "What is the difference between patricians and plebeians?",
        "Define the term pomerium.",
        "How does a Roman arch carry weight?",
        "What role did the tribunes of the plebs play?",
        "Explain the Twelve Tables and their importance.",
    ];
    group.bench_function("embed_batch_10", |b| {
        b.iter(|| model.embed_batch(&batch).unwrap());
    });

    group.finish();
}

criterion_group!(benches, embed_benchmarks);
criterion_main!(benches);
