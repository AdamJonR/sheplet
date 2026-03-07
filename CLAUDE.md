# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sheplet is a fully local RAG + LoRA fine-tuning platform for education, built entirely in pure Rust. Three executables:
- **sheplet-instructor** (`bins/sheplet-instructor`) — CLI for professors to ingest documents, fine-tune models, and package signed bundles
- **sheplet-instructor-web** (`bins/sheplet-instructor-web`) — Web UI alternative to the CLI (axum + vanilla HTML/CSS/JS) with guided workflow and SSE progress tracking
- **sheplet-student** (`bins/sheplet-student`) — Desktop client (axum + HTML/CSS/JS) for students to chat with course-specific fine-tuned models

Everything runs locally with no API keys, cloud services, or Python. See `SPEC.md` for the full technical specification.

## Build & Development Commands

```bash
cargo build                    # Debug build (all workspace members)
cargo build --release          # Release build
cargo check                    # Quick type/syntax check
cargo test                     # Run all tests
cargo test -p <crate>          # Run tests for a specific crate (e.g., cargo test -p finetune)
cargo build -p sheplet-instructor      # Build only the instructor CLI
cargo build -p sheplet-instructor-web  # Build only the instructor web UI
cargo build -p sheplet-student         # Build only the student binary
```

## Workspace Architecture

Cargo workspace with 3 binaries and 7 library crates:

```
bins/
  sheplet-instructor      — CLI (clap): init, ingest, model, finetune, config, bundle
  sheplet-instructor-web  — Web UI (axum): same workflow as CLI with browser interface
  sheplet-student          — Desktop app: axum server + HTML/CSS/JS frontend

crates/
  parser                — Document parsing (PDF, Word, Excel, CSV, text) + semantic chunking
  embeddings            — Embedding model (all-MiniLM-L6-v2) via Candle
  db                    — Vector database abstraction (LanceDB)
  rag                   — Retrieval-augmented generation pipeline
  finetune              — LoRA fine-tuning (DPO + SFT) via Candle
  bundle                — Bundle compression (zstd) and signing (Ed25519)
  conversations         — Persistent conversation storage (sled)
```

## Key Technical Decisions

- **Pure Rust, no Python**: ML inference uses `candle-core`/`candle-transformers`/`candle-nn` (not PyTorch/LibTorch)
- **Default model**: Phi-4-mini-instruct (3.8B params), quantized to Q4_K_M
- **Embedding model**: all-MiniLM-L6-v2, bundled inside `.sheplet` files (~90MB)
- **Vector DB**: LanceDB (embedded, no server process)
- **Bundle format**: `.sheplet` — zstd-compressed archive containing quantized model, LoRA adapter, embeddings, LanceDB, and config, signed with Ed25519
- **Rust edition 2024** across the workspace
- **Dependencies managed at workspace level** in root `Cargo.toml` — individual crates use `dependency.workspace = true`

## Data Flow

1. **Instructor**: Documents → `parser` (chunk) → `embeddings` (embed) → `db` (store in LanceDB) → `finetune` (LoRA on Phi-4-mini) → `bundle` (compress + sign → `.sheplet`)
2. **Student**: `.sheplet` → `bundle` (verify + extract) → query → `embeddings` (embed query) → `db` (retrieve chunks) → `rag` (assemble prompt) → Candle inference → response

## Important Patterns

- Academic integrity: queries below the relevance threshold are blocked rather than answered with hallucinated content
- Student settings are partially locked by instructor config — model weights, LoRA adapter, and system prompt cannot be changed by students
- Each course gets its own extracted bundle directory and conversation store
