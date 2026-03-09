use criterion::{Criterion, criterion_group, criterion_main};
use db::SearchResult;
use conversations::{Message, Role};
use rag::prompt::{assemble_prompt, assemble_prompt_gemma};

fn make_results(n: usize) -> Vec<SearchResult> {
    (0..n)
        .map(|i| SearchResult {
            text: format!(
                "This is chunk {i} containing approximately two hundred characters of text \
                 that simulates a real document chunk from course materials. It discusses \
                 various topics relevant to the course content and provides context."
            ),
            source_file: format!("chapter_{}.pdf", i + 1),
            chunk_index: i as u32,
            score: 0.95 - (i as f32 * 0.05),
        })
        .collect()
}

fn make_history(n: usize) -> Vec<Message> {
    (0..n)
        .map(|i| Message {
            role: if i % 2 == 0 { Role::User } else { Role::Assistant },
            content: format!(
                "This is message {i} in the conversation history with typical length."
            ),
            timestamp: format!("2026-01-01T00:{:02}:{:02}Z", i / 60, i % 60),
            citations: vec![],
        })
        .collect()
}

fn prompt_assembly(c: &mut Criterion) {
    let system = "You are a helpful course tutor. Answer questions using only the provided context.";
    let question = "What is the relationship between photosynthesis and cellular respiration?";

    let results_3 = make_results(3);
    let results_5 = make_results(5);
    let history_10 = make_history(10);

    let mut group = c.benchmark_group("prompt_assembly");

    group.bench_function("phi3_3results_0history", |b| {
        b.iter(|| assemble_prompt(system, &results_3, &[], question));
    });

    group.bench_function("phi3_5results_10history", |b| {
        b.iter(|| assemble_prompt(system, &results_5, &history_10, question));
    });

    group.bench_function("gemma_3results_0history", |b| {
        b.iter(|| assemble_prompt_gemma(system, &results_3, &[], question));
    });

    group.bench_function("gemma_5results_10history", |b| {
        b.iter(|| assemble_prompt_gemma(system, &results_5, &history_10, question));
    });

    group.finish();
}

criterion_group!(benches, prompt_assembly);
criterion_main!(benches);
