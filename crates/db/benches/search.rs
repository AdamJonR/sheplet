use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use db::{ChunkRecord, InMemoryStore, VectorStore};
use rand::Rng;

const DIM: usize = 384;
const K: usize = 5;
const SIZES: [usize; 3] = [100, 1000, 5000];

fn random_normalized_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.r#gen::<f32>() - 0.5).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn make_test_data(n: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let vectors = random_normalized_vectors(n, dim);
    let query = random_normalized_vectors(1, dim).into_iter().next().unwrap();
    (vectors, query)
}

fn make_inmemory_store(vectors: Vec<Vec<f32>>) -> InMemoryStore {
    let records: Vec<(Vec<f32>, String, String, u32)> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, format!("chunk_{i}"), "doc.pdf".to_string(), i as u32))
        .collect();
    InMemoryStore::from_records(&records)
}

fn make_chunk_records(vectors: Vec<Vec<f32>>) -> Vec<ChunkRecord> {
    vectors
        .into_iter()
        .enumerate()
        .map(|(i, v)| ChunkRecord {
            vector: v,
            text: format!("chunk_{i}"),
            source_file: "doc.pdf".to_string(),
            chunk_index: i as u32,
        })
        .collect()
}

fn inmemory_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("inmemory_topk");
    for n in SIZES {
        let (vectors, query) = make_test_data(n, DIM);
        let store = make_inmemory_store(vectors);
        group.bench_with_input(BenchmarkId::new("topk", n), &n, |b, _| {
            b.iter(|| store.search_top_k(&query, K));
        });
    }
    group.finish();
}

fn inmemory_mmr(c: &mut Criterion) {
    let mut group = c.benchmark_group("inmemory_mmr");
    for n in SIZES {
        let (vectors, query) = make_test_data(n, DIM);
        let store = make_inmemory_store(vectors);
        group.bench_with_input(BenchmarkId::new("mmr", n), &n, |b, _| {
            b.iter(|| store.search_mmr(&query, K, 0.7));
        });
    }
    group.finish();
}

fn inmemory_vs_lancedb(c: &mut Criterion) {
    let n = 1000;
    let mut group = c.benchmark_group("inmemory_vs_lancedb");

    // Both stores use the same data for a fair comparison.
    let (vectors, query) = make_test_data(n, DIM);
    let inmem_store = make_inmemory_store(vectors.clone());

    group.bench_function("inmemory_topk_1000", |b| {
        b.iter(|| inmem_store.search_top_k(&query, K));
    });

    // LanceDB setup — rt.block_on overhead is included in each iteration since
    // VectorStore::search_top_k is async. This measures end-to-end latency as
    // seen by a caller, not pure search time.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let dir = tempfile::TempDir::new().unwrap();
    let lance_store = rt.block_on(async {
        let store = VectorStore::open_or_create(dir.path(), "bench", DIM)
            .await
            .unwrap();
        store.insert(&make_chunk_records(vectors)).await.unwrap();
        store
    });

    group.bench_function("lancedb_topk_1000", |b| {
        b.iter(|| {
            rt.block_on(async { lance_store.search_top_k(&query, K).await.unwrap() })
        });
    });

    group.finish();
}

criterion_group!(benches, inmemory_topk, inmemory_mmr, inmemory_vs_lancedb);
criterion_main!(benches);
