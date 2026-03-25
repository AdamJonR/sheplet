# Sheplet

A fully local RAG + LoRA fine-tuning platform for education, built entirely in pure Rust. Professors ingest course documents, fine-tune a small language model, and package everything into a signed `.sheplet` bundle. Students download the bundle and a single executable to chat with a course-specific AI grounded in their materials — no API keys, no cloud services, no Python, no setup.

## Binaries

- **sheplet-instructor** — CLI for professors to ingest documents, fine-tune models, and package signed bundles
- **sheplet-instructor-web** — Web UI alternative to the CLI (axum + vanilla HTML/CSS/JS) with guided workflow and real-time progress tracking
- **sheplet-student** — Desktop client for students to chat with course-specific fine-tuned models

## Key Features

- **Fully local** — everything runs on the user's machine with no network calls at inference time
- **Pure Rust** — ML inference via Candle, no Python or LibTorch dependency
- **GPU-accelerated** — Metal (macOS) and CUDA (Windows/Linux) via feature flags, with automatic CPU fallback
- **Academic integrity** — queries below the relevance threshold are blocked rather than answered with hallucinated content
- **Signed bundles** — Ed25519 signing and verification prevents tampering with course materials

## Supported Models

| Model | Parameters | Architecture | License |
|---|---|---|---|
| Phi-3-mini-4k-instruct (default) | 3.8B | phi3 | MIT |
| Llama-3.2-1B-Instruct | 1B | llama | Llama 3.2 Community |
| Llama-3.2-3B-Instruct | 3B | llama | Llama 3.2 Community |
| Qwen2.5-0.5B-Instruct | 0.5B | qwen2 | Apache 2.0 |
| Qwen2.5-1.5B-Instruct | 1.5B | qwen2 | Apache 2.0 |
| Qwen2.5-3B-Instruct | 3B | qwen2 | Apache 2.0 |
| google/gemma-2b-it | 2B | gemma | Gemma |
| google/gemma-2-2b-it | 2B | gemma2 | Gemma |
| Mistral-7B-Instruct-v0.3 | 7B | mistral | Apache 2.0 |

All models use full-precision SafeTensors. Architecture is auto-detected from the model's `config.json`.

## Build

```bash
cargo build                    # Debug build (all workspace members)
cargo build --release          # Release build
cargo test                     # Run all tests
```

GPU acceleration:
```bash
cargo build --features metal   # macOS Metal
cargo build --features cuda    # NVIDIA CUDA
```

## Architecture

```
bins/
  sheplet-instructor      — CLI (clap)
  sheplet-instructor-web  — Web UI (axum)
  sheplet-student         — Desktop client (axum)

crates/
  parser       — Document parsing (PDF, Word, Excel, CSV, text) + semantic chunking
  embeddings   — Embedding model (all-MiniLM-L6-v2) via Candle
  db           — Vector database (LanceDB)
  rag          — Retrieval-augmented generation pipeline + inference
  finetune     — LoRA fine-tuning (DPO + SFT) via Candle
  bundle       — Bundle compression (zstd) and signing (Ed25519)
  conversations — Persistent conversation storage (sled)
  compute      — Device selection and GPU feature flags
  project      — Project manifest and config management
```

See [SPEC.md](SPEC.md) for the full technical specification.
