# Sheplet — Project Specification v1.6
> A fully local, socioeconomically accessible RAG + fine-tuning platform for students and instructors, built entirely in Rust.

---

## 1. Project Overview

Sheplet consists of two separate executables:

- **sheplet-instructor** — a CLI tool used by professors to ingest source documents, build a LanceDB vector database, select and quantize a model, perform LoRA fine-tuning, and package everything into a single signed distributable bundle.
- **sheplet-student** — a zero-setup desktop client used by students. They download the executable and a `.sheplet` bundle from their professor, and everything works immediately.

Everything runs locally. No API keys, no cloud services, no Python environment, no configuration required of students.

---

## 2. Core Design Principles

- **Zero cost to run** — no external API dependencies at runtime.
- **Zero student setup** — students download two files and start chatting.
- **Cross-platform** — both executables target Windows, macOS, and Linux.
- **Accessible hardware** — designed for modest consumer hardware (e.g., 16GB RAM Intel laptops).
- **Pure Rust throughout** — every component is Rust-native, including the student frontend via Leptos.
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
| Instructor Interface | CLI (`clap`) | Pure Rust argument parsing |
| Bundle Compression | `zstd` | Pure Rust, high compression ratio |
| Bundle Signing | `ed25519-dalek` | Pure Rust, asymmetric signing |
| Conversation Storage | `sled` | Pure Rust embedded key-value store |
| Model Downloading | `hf-hub` | Pure Rust, Hugging Face Hub |
| Async Runtime | `tokio` | Pure Rust standard |
| Serialization | `serde`, `serde_json` | Pure Rust standard |

---

## 4. Default Model

**Small Language Model**
- Default: **Phi-4-mini-instruct** (3.8B parameters, Microsoft, MIT License)
- The `-instruct` variant is specifically chosen because it has already been fine-tuned by Microsoft using SFT and DPO for instruction-following and chat. This makes it the correct foundation for educational use — it produces coherent conversational responses out of the box.
- LoRA fine-tuning layers on top of the instruct base, nudging behavior toward professor preferences while preserving underlying chat capabilities. The base instruct weights are frozen throughout training; only the small LoRA adapter matrices are trained.
- Format: SafeTensors, downloaded via `hf-hub` during instructor bundle build
- Default quantization: Q4_K_M (best balance of quality and memory for modest hardware)
- Instructor-selectable quantization: Q4_K_M, Q8_0, Q4_0
- Context window: 128K tokens

**Embedding Model**
- Default: all-MiniLM-L6-v2 in SafeTensors format, run via Candle
- Always bundled inside the `.sheplet` file — students need no separate downloads (~90MB overhead per bundle, accepted for full self-containment)

---

## 5. The Two Executables

### sheplet-instructor
A CLI tool for professors. Runs on the professor's machine. Used to:
- Ingest a directory of source documents
- Build and populate a LanceDB vector database
- Download Phi-4-mini-instruct from Hugging Face Hub
- Apply quantization
- Perform LoRA fine-tuning (DPO and/or SFT)
- Sign and package everything into a versioned `.sheplet` bundle file

### sheplet-student
A self-contained desktop client for students. Runs on the student's machine. Used to:
- Verify, load, and extract `.sheplet` bundle files from professors
- Manage multiple course bundles with a course switcher
- Chat with the fine-tuned model grounded in course documents
- Adjust retrieval settings (Top-K / MMR) within instructor-permitted bounds
- Browse and resume past conversations
- Export conversations as plain text

---

## 6. The .sheplet Bundle

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

## 7. Bundle Signing & Verification

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

## 8. sheplet-instructor CLI Workflow

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
```
- Downloads Phi-4-mini-instruct SafeTensors from Hugging Face Hub via `hf-hub`
- Applies selected quantization via Candle
- Saves quantized weights to `./bio101/model/`

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

## 9. Document Ingestion Details

**Supported Formats:** PDF (.pdf), Word (.docx), Excel (.xlsx), CSV (.csv), Plain text (.txt)

**PDF Parsing Notes:** `pdf-extract` handles standard text-based PDFs well. Scanned PDFs and complex layouts may degrade. CLI warns professor during ingestion. Word or plain text recommended for best results.

**Chunking — text-splitter**
- Semantic splitting on paragraph then sentence boundaries
- Target chunk size: 200–500 tokens (token-aware)
- Overlap: ~20% between adjacent chunks
- Excel/CSV: each row becomes a chunk with column headers prepended
- Fallback to fixed-size chunking if semantic parsing fails

---

## 10. Fine-Tuning Data Templates

**dpo_template.jsonl**
```json
{"prompt": "What is mitosis?", "chosen": "Mitosis is a process of cell division whereby a single cell divides into two genetically identical daughter cells.", "rejected": "I'm not sure exactly, it might be something to do with cells splitting."}
```

**sft_template.jsonl**
```json
{"input": "What is mitosis?", "output": "Mitosis is a process of cell division whereby a single cell divides into two genetically identical daughter cells."}
```

---

## 11. sheplet-student Chat Interface & RAG Pipeline

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
3. Top-K or MMR chunks retrieved from bundled LanceDB
4. Relevance scores checked — if below threshold, query blocked
5. Prompt assembled: system prompt + chunks + conversation history + question
6. Phi-4-mini-instruct generates response
7. Response shown with collapsible source citations

**Prompt Structure**
```
[SYSTEM PROMPT — from bundle config.json]
[RETRIEVED CONTEXT CHUNKS with source labels]
[CONVERSATION HISTORY — rolling window]
[STUDENT QUESTION]
```

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

## 12. Fine-Tuning Details

**Why LoRA on an Instruct Model Works:** Phi-4-mini-instruct already understands conversation. LoRA adds small trainable rank decomposition matrices that shift behavior toward professor preferences. The instruct base is frozen. Professors need far less training data than starting from a base model.

**Methods**
- **DPO** — preferred vs. non-preferred pairs; stronger alignment; recommended
- **SFT** — example input-output pairs; simpler; good starting point
- Both available and combinable in sequence

**Hardware Pre-flight:** Detects available RAM, estimates training time, requires professor confirmation before proceeding. Recommended minimum: 16GB RAM.

---

## 13. Application Architecture

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
│  │    LanceDB      │  │  Conversation Store  │   │
│  │  (per course)   │  │  (sled)              │   │
│  └─────────────────┘  └──────────────────────┘   │
│  ┌────────────────────────────────────────────┐  │
│  │  Phi-4-mini-instruct (quantized)           │  │
│  │  all-MiniLM-L6-v2 | adapter.lora           │  │
│  │  (all sourced from active bundle)          │  │
│  └────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

---

## 14. Project Directory Structure (Instructor Machine)
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

## 15. Student File Layout
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

## 16. Distribution

### sheplet-instructor
- Downloaded by professors from the Sheplet project website
- Single binary per platform: Windows, macOS, Linux

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

## 17. Open Source Considerations

To be decided. Recommended: open source core (MIT or Apache 2.0) with official signed builds via the project website. Community can audit and contribute; malicious forks remain clearly distinguishable.

---

## 18. Known Limitations (v1.0)

- Bundle size ~2.7–3GB — professors should verify LMS upload limits
- PDF parsing may degrade on scanned or complex layout PDFs
- CPU-only inference in v1.0
- No automatic bundle update mechanism
- No image or audio document support

---

## 19. Full Rust Crate Reference

| Crate | Used In | Purpose | Pure Rust |
|---|---|---|---|
| `candle-core` | Both | Tensor ops, model inference | ✅ |
| `candle-transformers` | Both | Phi-4-mini-instruct architecture | ✅ |
| Manual LoRA (`candle-core`/`candle-nn`) | Instructor | LoRA fine-tuning (DPO + SFT) | ✅ |
| `lancedb` | Both | Vector database | ✅ |
| `text-splitter` | Instructor | Semantic + token-aware chunking | ✅ |
| `pdf-extract` | Instructor | PDF text extraction | ✅ |
| `docx-rs` | Instructor | Word document parsing | ✅ |
| `calamine` | Instructor | Excel/CSV/ODS parsing | ✅ |
| `csv` | Instructor | CSV parsing | ✅ |
| `clap` | Instructor | CLI argument parsing | ✅ |
| `zstd` | Both | Bundle compression/decompression | ✅ |
| `ed25519-dalek` | Both | Bundle signing and verification | ✅ |
| `sled` | Student | Conversation persistence | ✅ |
| `axum` | Student | Local web server | ✅ |
| `hf-hub` | Instructor | Hugging Face model downloading | ✅ |
| `tokio` | Both | Async runtime | ✅ |
| `serde` / `serde_json` | Both | Serialization | ✅ |

---
