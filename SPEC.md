# Sheplet — Project Specification v1.8
> A fully local, socioeconomically accessible RAG + fine-tuning platform for students and instructors, built entirely in Rust.

---

## 1. Project Overview

Sheplet consists of three executables:

- **sheplet-instructor** — a CLI tool used by professors to ingest source documents, build a LanceDB vector database, select and quantize a model, perform LoRA fine-tuning, and package everything into a single signed distributable bundle.
- **sheplet-instructor-web** — a web-based alternative to the CLI that provides the same capabilities through a guided browser interface with real-time progress visualization. Ideal for instructors who prefer a graphical workflow.
- **sheplet-student** — a zero-setup desktop client used by students. They download the executable and a `.sheplet` bundle from their professor, and everything works immediately.

Everything runs locally. No API keys, no cloud services, no Python environment, no configuration required of students.

---

## 2. Core Design Principles

- **Zero cost to run** — no external API dependencies at runtime.
- **Zero student setup** — students download two files and start chatting.
- **Cross-platform** — both executables target Windows, macOS, and Linux.
- **Accessible hardware** — designed for modest consumer hardware (e.g., 16GB RAM Intel laptops).
- **Pure Rust throughout** — every component is Rust-native, including the student frontend via axum + HTML/CSS/JS.
- **Instructor empowerment** — professors build ready-to-use course models without writing code.
- **Academic integrity** — low-relevance queries are blocked rather than hallucinated.

---

## 3. Technology Stack

| Component | Crate / Technology | Notes |
|---|---|---|
| Language | Rust | Two executables, cross-platform |
| Vector Database | `lancedb` | Pure Rust core, embedded, no server |
| ML Framework | `candle-core`, `candle-transformers` | Pure Rust, no Python or LibTorch |
| LoRA Fine-tuning | Manual LoRA via `candle-core`/`candle-nn` | Pure Rust, supports DPO + SFT |
| Text Chunking | `text-splitter` | Pure Rust, semantic + token-aware |
| PDF Parsing | `pdf-extract` | Pure Rust; limitations documented |
| Word Parsing | `docx-rs` | Pure Rust |
| Excel/CSV Parsing | `calamine` | Pure Rust, handles xlsx/xls/ods/csv |
| CSV | `csv` crate | Pure Rust, mature |
| Embeddings | Candle native (all-MiniLM-L6-v2) | Pure Rust via Candle, SafeTensors |
| Student Web Server | `axum` | Pure Rust, modern async ergonomics |
| Instructor Interface | CLI (`clap`) + Web UI (`axum`) | CLI or browser-based workflow |
| Bundle Compression | `zstd` | Pure Rust, high compression ratio |
| Bundle Signing | `ed25519-dalek` | Pure Rust, asymmetric signing |
| Conversation Storage | `sled` | Pure Rust embedded key-value store |
| Model Downloading | `hf-hub` | Pure Rust, Hugging Face Hub |
| Async Runtime | `tokio` | Pure Rust standard |
| Serialization | `serde`, `serde_json` | Pure Rust standard |

---

## 4. GPU Acceleration

The `compute` crate provides workload-aware device selection with optional GPU support:

- **Workload routing**: Inference and training tasks prefer GPU when available; embedding and quantization always run on CPU
- **Feature flags**: `metal` (macOS GPU via Metal Performance Shaders), `cuda` (Linux/Windows GPU via CUDA)
- **Fallback chain**: Metal → CUDA → CPU — the first available accelerator is selected automatically
- **Default**: CPU-only in release builds; GPU support opt-in via Cargo feature flags
- **CI**: Metal acceleration tested on macOS runners

---

## 5. Default Model & Embedding Model

**Small Language Model**
- Default: **Phi-4-mini-instruct** (3.8B parameters, Microsoft, MIT License)
- The `-instruct` variant is specifically chosen because it has already been fine-tuned by Microsoft using SFT and DPO for instruction-following and chat. This makes it the correct foundation for educational use — it produces coherent conversational responses out of the box.
- LoRA fine-tuning layers on top of the instruct base, nudging behavior toward professor preferences while preserving underlying chat capabilities. The base instruct weights are frozen throughout training; only the small LoRA adapter matrices are trained.
- Format: SafeTensors, downloaded via `hf-hub` during instructor bundle build
- Default quantization: Q4_K_M (best balance of quality and memory for modest hardware)
- Instructor-selectable quantization: Q4_K_M, Q5_K_M, Q8_0, Q4_0
- Context window: 128K tokens
- Architecture auto-detected from `config.json` `model_type` field

**Embedding Model**
- Default: all-MiniLM-L6-v2 in SafeTensors format, run via Candle
- Always bundled inside the `.sheplet` file — students need no separate downloads (~90MB overhead per bundle, accepted for full self-containment)

---

## 6. Supported Models

| Model | Parameters | License | Quantization | Notes |
|---|---|---|---|---|
| Phi-4-mini-instruct (default) | 3.8B | MIT | Q4_K_M, Q5_K_M, Q8_0, Q4_0 | Open, instruct-tuned |
| Gemma-3-1B-IT | 1B | Gemma license | None (BF16 SafeTensors) | Gated (requires `HF_TOKEN`) |

- Architecture auto-detected from `config.json` `model_type` field (`phi3` or `gemma3`)
- Gemma uses full BF16 SafeTensors — no GGUF quantization is applied
- Gemma is a gated model on Hugging Face; instructors must set `HF_TOKEN` before downloading

---

## 7. The Executables

### sheplet-instructor (CLI)
A CLI tool for professors. Runs on the professor's machine. Used to:
- Ingest a directory of source documents
- Build and populate a LanceDB vector database
- Download Phi-4-mini-instruct from Hugging Face Hub
- Apply quantization
- Perform LoRA fine-tuning (DPO and/or SFT)
- Sign and package everything into a versioned `.sheplet` bundle file

### sheplet-instructor-web (Web UI)
A web-based alternative to the CLI for professors who prefer a graphical interface. Provides the same capabilities as `sheplet-instructor` through a browser-based dashboard:
- Guided workflow with project status checklist
- Visual progress tracking for long-running operations (document ingestion, model download, fine-tuning) via Server-Sent Events
- Form-based configuration with descriptions of each option
- Multi-project management with project selector
- Runs on `localhost:8421` — self-contained binary with embedded HTML/CSS/JS frontend

### sheplet-student
A self-contained desktop client for students. Runs on the student's machine. Used to:
- Verify, load, and extract `.sheplet` bundle files from professors
- Manage multiple course bundles with a course switcher
- Chat with the fine-tuned model grounded in course documents
- Adjust retrieval settings (Top-K / MMR) within instructor-permitted bounds
- Browse and resume past conversations
- Export conversations as plain text

---

## 8. The .sheplet Bundle

The `.sheplet` bundle is a single fully self-contained compressed archive, generated and cryptographically signed by `sheplet-instructor`. It contains everything the student needs — no internet required after download.

### Bundle Contents
```
class_model.sheplet  (zstd-compressed archive)
├── manifest.json          ← version, course name, model info, build timestamp, public key
├── signature.sig          ← Ed25519 signature over the bundle contents
├── model/                 ← quantized Phi-4-mini-instruct weights (SafeTensors, Q4_K_M)
├── adapter.lora           ← LoRA fine-tuning adapter
├── embeddings/            ← all-MiniLM-L6-v2 embedding model weights
├── database/              ← LanceDB vector database (pre-populated)
└── config.json            ← system prompt, retrieval settings, locked flags
```

### Approximate Bundle Size
- Quantized model (Q4_K_M): ~2.5GB
- Embedding model: ~90MB
- LanceDB database: varies by course materials, typically 10–200MB
- LoRA adapter: ~50MB
- **Total: approximately 2.7–3GB**

Professors should be aware of LMS upload limits and advise students accordingly. The project website will document recommended hosting options for large bundles.

### Versioning
- `manifest.json` contains a semantic version number and build timestamp
- When a professor updates the bundle and re-posts it, students re-download and reload
- `sheplet-student` displays bundle version and course name prominently in the UI
- On loading a new version of an existing course bundle, the old extracted contents are overwritten — one active version per course at a time
- No automatic update mechanism — re-download is intentional and manual

### Multi-Bundle Support
- Students can load bundles from multiple courses simultaneously
- A course switcher in the `sheplet-student` UI allows switching between active courses
- Each course bundle is stored and extracted independently
- Loading a new version of an existing course overwrites that course's extracted directory only — other courses are unaffected

### Student Settings Permissions
Defined in `config.json` by the professor:
- **Locked (student cannot change):** model weights, LoRA adapter, system prompt, embedding model
- **Unlocked (student can adjust):** retrieval strategy (Top-K vs MMR), K value, relevance threshold

---

## 9. Bundle Signing & Verification

Sheplet uses Ed25519 asymmetric cryptography (`ed25519-dalek`) to sign bundles at build time and verify them at load time.

### Instructor Side (sheplet-instructor)
- On first run, `sheplet-instructor` generates an Ed25519 keypair stored locally on the professor's machine
- The private key is used to sign each bundle at pack time
- The public key is embedded in `manifest.json` of every bundle
- `sheplet-instructor` also outputs a human-readable public key fingerprint that professors can optionally publish alongside their bundle download link

### Student Side (sheplet-student)
- Before extracting any `.sheplet` file, `sheplet-student` verifies the Ed25519 signature against the public key in `manifest.json`
- If verification fails, extraction is refused and the student is shown a clear error: *"This bundle could not be verified. Do not use it. Contact your professor."*
- If verification passes, extraction proceeds silently

---

## 10. sheplet-instructor CLI Workflow

### Step 1 — Initialize a course project
```bash
sheplet-instructor init --course "Introduction to Biology" --output ./bio101
```
Creates a scaffolded project directory. On first run ever, also generates the professor's Ed25519 keypair.

### Step 2 — Generate fine-tuning data templates
```bash
sheplet-instructor templates --project ./bio101
```
Writes example JSONL template files to `./bio101/finetune_data/`:
- `dpo_template.jsonl` — annotated DPO examples with comments explaining the format
- `sft_template.jsonl` — annotated SFT examples with comments explaining the format

### Step 3 — Ingest source documents
```bash
sheplet-instructor ingest --sources ./sources/ --project ./bio101
```
- Recursively processes all supported files in the `sources/` directory
- Supported formats: PDF, Word (.docx), Excel (.xlsx), CSV, plain text (.txt)
- Chunks documents using `text-splitter` (semantic, 200–500 tokens, 20% overlap)
- Embeds chunks using all-MiniLM-L6-v2 via Candle
- Populates LanceDB vector database at `./bio101/database/`
- Warns if PDF parsing quality appears degraded

### Step 4 — Download and quantize the model
```bash
sheplet-instructor model --name phi-4-mini-instruct --quantization q4_k_m --project ./bio101
# or, for Gemma:
sheplet-instructor model --name gemma-3-1b-it --project ./bio101
```
- Downloads model SafeTensors from Hugging Face Hub via `hf-hub`
- For Phi: applies selected quantization (Q4_K_M, Q5_K_M, Q8_0, Q4_0) via Candle
- For Gemma: skips quantization (uses BF16 SafeTensors directly); requires `HF_TOKEN` (gated model)
- Saves weights to `./bio101/model/`

### Step 5 — Fine-tune with LoRA
```bash
sheplet-instructor finetune --method dpo --data ./bio101/finetune_data/dpo.jsonl --project ./bio101
sheplet-instructor finetune --method sft --data ./bio101/finetune_data/sft.jsonl --project ./bio101
```
Pre-flight hardware warning displayed before training begins:
```
⚠  Hardware check:
   Available RAM : 14.2 GB
   Estimated time: ~45 minutes
   Minimum recommended RAM: 16 GB
Proceed? [y/N]
```

### Step 6 — Configure course settings
```bash
sheplet-instructor config --project ./bio101 \
  --system-prompt "You are a helpful biology tutor. Answer only from the provided course materials." \
  --retrieval top-k \
  --top-k 5 \
  --relevance-threshold 0.7
```

### Step 7 — Package and sign the bundle
```bash
sheplet-instructor bundle --project ./bio101 --output ./bio101_v1.sheplet
```
- Compresses with `zstd`, signs with Ed25519, outputs `.sheplet` file

### Re-bundling for updates
```bash
sheplet-instructor bundle --project ./bio101 --output ./bio101_v2.sheplet --bump-version
```

---

## 11. Document Ingestion Details

**Supported Formats:** PDF (.pdf), Word (.docx), Excel (.xlsx), CSV (.csv), Plain text (.txt)

**PDF Parsing Notes:** `pdf-extract` handles standard text-based PDFs well. Scanned PDFs and complex layouts may degrade. CLI warns professor during ingestion. Word or plain text recommended for best results.

**Chunking — text-splitter**
- Semantic splitting on paragraph then sentence boundaries
- Target chunk size: 200–500 tokens (token-aware)
- Overlap: ~20% between adjacent chunks
- Excel/CSV: each row becomes a chunk with column headers prepended
- Fallback to fixed-size chunking if semantic parsing fails

---

## 12. Fine-Tuning Data Templates

**dpo_template.jsonl**
```json
{"prompt": "What is mitosis?", "chosen": "Mitosis is a process of cell division whereby a single cell divides into two genetically identical daughter cells.", "rejected": "I'm not sure exactly, it might be something to do with cells splitting."}
```

**sft_template.jsonl**
```json
{"input": "What is mitosis?", "output": "Mitosis is a process of cell division whereby a single cell divides into two genetically identical daughter cells."}
```

---

## 13. sheplet-student Chat Interface & RAG Pipeline

### First Launch
1. Student launches `sheplet-student`
2. Browser opens to localhost UI
3. Student loads one or more `.sheplet` bundle files
4. Each bundle is signature-verified then extracted
5. Course switcher appears if multiple bundles are loaded
6. Chat interface available immediately

### Relevance Threshold Behavior
If all retrieved chunks fall below the relevance threshold, the query is blocked:
> *"No relevant material was found in your course documents for this question. Try rephrasing, or ask about a topic covered in your course materials."*

The model does not fall back to its own training knowledge. This protects academic integrity.

### RAG Loop (when threshold is met)
1. Student submits a question
2. Question embedded via Candle (all-MiniLM-L6-v2 from bundle)
3. On bundle load, vectors are loaded into an in-memory store (`InMemoryStore`) for fast brute-force search, bypassing LanceDB I/O on each query. Top-K or MMR chunks retrieved from this in-memory store.
4. Relevance scores checked — if below threshold, query blocked
5. Prompt assembled: system prompt + chunks + conversation history + question
6. Model generates response (Phi-4-mini-instruct or Gemma-3-1B-IT, depending on bundle)
7. Response shown with collapsible source citations

**Prompt Structure**
```
[SYSTEM PROMPT — from bundle config.json]
[RETRIEVED CONTEXT CHUNKS with source labels]
[CONVERSATION HISTORY — rolling window]
[STUDENT QUESTION]
```

**Prompt Formats** (auto-selected by model architecture):
- **Phi-3/4**: `<|system|>...<|end|>\n<|user|>...<|end|>\n<|assistant|>`
- **Gemma-3**: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model` (system prompt folded into first user turn)

### Conversation Persistence
- All conversations saved locally via `sled` embedded key-value store
- Persist across sessions — students resume where they left off
- Conversation browser in sidebar, grouped by course and date
- Students can delete individual conversations or clear all history for a course
- Stored in `conversations/` directory alongside the executable

### Conversation Export
- Export any conversation as plain text (.txt)
- Includes timestamps, source citations, bundle version, and course name

---

## 14. Fine-Tuning Details

**Why LoRA on an Instruct Model Works:** Phi-4-mini-instruct already understands conversation. LoRA adds small trainable rank decomposition matrices that shift behavior toward professor preferences. The instruct base is frozen. Professors need far less training data than starting from a base model.

**Methods**
- **DPO** — preferred vs. non-preferred pairs; stronger alignment; recommended
- **SFT** — example input-output pairs; simpler; good starting point
- Both available and combinable in sequence

**Supported Architectures:** LoRA supports both Phi-3/4 and Gemma-3. Phi-3 uses a fused `qkv_proj` weight matrix; Gemma-3 uses separate `q_proj`/`k_proj`/`v_proj`/`o_proj` projections, `GemmaRmsNorm` (1 + weight scaling), and a tied `lm_head`. The LoRA implementation handles these differences transparently.

**Hardware Pre-flight:** Detects available RAM, estimates training time, requires professor confirmation before proceeding. Recommended minimum: 16GB RAM.

---

## 15. Application Architecture

### sheplet-instructor (CLI)
```
sheplet-instructor
├── init        → scaffold project, generate Ed25519 keypair on first run
├── templates   → write annotated DPO and SFT JSONL template files
├── ingest      → parse docs → chunk → embed → populate LanceDB
├── model       → download phi-4-mini-instruct via hf-hub → quantize via Candle
├── finetune    → hardware pre-flight → DPO or SFT via manual LoRA → adapter.lora
├── config      → write system prompt + retrieval settings to config.json
└── bundle      → compress with zstd → sign with Ed25519 → output .sheplet
```

### sheplet-instructor-web (Web UI)
```
┌─────────────────────────────────────────────────┐
│          HTML/CSS/JS Frontend                   │
│      Served on localhost:8421 via axum           │
│  Dashboard | Ingest | Model | Finetune | Bundle │
└──────────────────┬──────────────────────────────┘
                   │ HTTP + SSE (localhost only)
┌──────────────────▼──────────────────────────────┐
│              axum + Task Manager                 │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │   Projects   │  │   Background Tasks       │ │
│  │  (multi)     │  │   (SSE progress)         │ │
│  └──────────────┘  └──────────────────────────┘ │
│  ┌──────────────────────────────────────────┐   │
│  │  Library Crates (same as CLI)            │   │
│  │  parser | embeddings | db | rag          │   │
│  │  finetune | bundle | compute | project   │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### sheplet-student (Desktop Client)
```
┌────────────────────────────────────────────────┐
│           HTML/CSS Frontend         │
│     Served on localhost via axum                 │
│  Course Switcher | Chat | Conversations | Export │
└───────────────────┬──────────────────────────────┘
                    │ HTTP (localhost only)
┌───────────────────▼──────────────────────────────┐
│               axum + Sheplet Core                 │
│  ┌─────────────────┐  ┌──────────────────────┐   │
│  │   RAG Pipeline  │  │   Bundle Manager     │   │
│  │   (Candle)      │  │   verify (ed25519)   │   │
│  └────────┬────────┘  │   extract (zstd)     │   │
│           │           └──────────────────────┘   │
│  ┌────────▼────────┐  ┌──────────────────────┐   │
│  │  InMemoryStore  │  │  Conversation Store  │   │
│  │  + LanceDB      │  │  (sled)              │   │
│  │  (per course)   │  │                      │   │
│  └─────────────────┘  └──────────────────────┘   │
│  ┌────────────────────────────────────────────┐  │
│  │  SLM (Phi-4 or Gemma-3, from bundle)      │  │
│  │  all-MiniLM-L6-v2 | adapter.lora           │  │
│  │  (all sourced from active bundle)          │  │
│  └────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

---

## 16. Project Directory Structure (Instructor Machine)
```
bio101/
  manifest.json
  config.json
  sources/                  ← original source documents (not bundled)
  database/                 ← LanceDB vector database
  model/                    ← quantized Phi-4-mini-instruct weights
  embeddings/               ← all-MiniLM-L6-v2 weights
  adapter.lora
  finetune_data/
    dpo_template.jsonl
    sft_template.jsonl

~/.sheplet-instructor/
  keypair.json              ← Ed25519 keypair (private key stays here)

bio101_v1.sheplet           ← signed distributable bundle
```

---

## 17. Student File Layout
```
sheplet-student/
  sheplet-student.exe
  courses/
    bio101/
      manifest.json
      model/
      embeddings/
      database/
      adapter.lora
      config.json
    chem201/
      ...
  conversations/
    bio101/                 ← sled conversation store
    chem201/
```

---

## 18. Distribution

### sheplet-instructor / sheplet-instructor-web
- Downloaded by professors from the Sheplet project website
- Single binary per platform: Windows, macOS, Linux
- Professors can choose either the CLI (`sheplet-instructor`) or the web UI (`sheplet-instructor-web`) — both produce identical outputs

### sheplet-student
- Downloaded by students from the Sheplet project website
- Single binary per platform: Windows, macOS, Linux
- Paired with `.sheplet` bundle files from professors

### Project Website
- Separate guides for professors and students
- Instructor CLI reference and fine-tuning tutorial
- Student quick-start (download two files, double-click, done)
- Bundle hosting recommendations for large files
- Signed download links with SHA256 checksums

---

## 19. Open Source Considerations

To be decided. Recommended: open source core (MIT or Apache 2.0) with official signed builds via the project website. Community can audit and contribute; malicious forks remain clearly distinguishable.

---

## 20. Known Limitations (v1.0)

- Bundle size ~2.7–3GB — professors should verify LMS upload limits
- PDF parsing may degrade on scanned or complex layout PDFs
- CPU inference by default; Metal (macOS) and CUDA (Linux/Windows) available via feature flags but not compiled in release builds
- No automatic bundle update mechanism
- No image or audio document support

---

## 21. Performance Testing

Criterion benchmarks cover the student hot path:

- **Vector search**: `cargo bench -p db --bench search` — measures brute-force and LanceDB query latency
- **Prompt assembly**: `cargo bench -p rag --bench prompt` — measures RAG prompt construction
- **Embedding**: `cargo bench -p embeddings --bench embed` — gated on `SHEPLET_BENCH_MODELS_DIR` (requires model weights)
- CI smoke-runs benchmarks on every push to detect regressions

---

## 22. Full Rust Crate Reference

| Crate | Used In | Purpose | Pure Rust |
|---|---|---|---|
| `candle-core` | Both | Tensor ops, model inference | ✅ |
| `candle-transformers` | Both | Phi-4 and Gemma-3 model architectures | ✅ |
| Manual LoRA (`candle-core`/`candle-nn`) | Instructor | LoRA fine-tuning (DPO + SFT) | ✅ |
| `lancedb` | Both | Vector database | ✅ |
| `text-splitter` | Instructor | Semantic + token-aware chunking | ✅ |
| `pdf-extract` | Instructor | PDF text extraction | ✅ |
| `docx-rs` | Instructor | Word document parsing | ✅ |
| `calamine` | Instructor | Excel/CSV/ODS parsing | ✅ |
| `csv` | Instructor | CSV parsing | ✅ |
| `clap` | Instructor CLI | CLI argument parsing | ✅ |
| `zstd` | Both | Bundle compression/decompression | ✅ |
| `ed25519-dalek` | Both | Bundle signing and verification | ✅ |
| `sled` | Student | Conversation persistence | ✅ |
| `axum` | Student + Instructor Web | Local web server | ✅ |
| `tokio-stream` | Student + Instructor Web | SSE streaming | ✅ |
| `tower-http` | Student + Instructor Web | CORS middleware | ✅ |
| `hf-hub` | Instructor | Hugging Face model downloading | ✅ |
| `tokio` | Both | Async runtime | ✅ |
| `serde` / `serde_json` | Both | Serialization | ✅ |
| `compute` (workspace crate) | Both | Device selection, GPU feature flags (Metal/CUDA) | ✅ |
| `project` (workspace crate) | Instructor | Project manifest and config management | ✅ |
| `criterion` | Dev only | Performance benchmarking (student hot path) | ✅ |

---
