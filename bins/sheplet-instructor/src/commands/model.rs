use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::progress;
use crate::project::{require_init, project_dirs, local_model_source, copy_local_model, is_gemma_model};

pub fn run(name: &str, quantization: &str, project: &Path) -> Result<()> {
    let mut manifest = require_init(project)?;
    let dirs = project_dirs(project);

    // Guard: GGUF quantization is not supported for Gemma models
    if is_gemma_model(name) && quantization != "none" {
        bail!(
            "GGUF quantization is not supported for Gemma models yet \
             (the quantizer's tensor name mapper is Phi-specific). \
             Use --quantization none for Gemma models."
        );
    }

    let model_dir = &dirs.model;

    // Check if this is a locally available model first
    if let Some(local_dir_name) = local_model_source(name) {
        let src_dir = std::env::current_dir()?.join("downloaded-models").join(local_dir_name);
        if !src_dir.exists() {
            bail!(
                "Local model directory not found: {}. \
                 Please download the model first.",
                src_dir.display()
            );
        }
        let pb = progress::spinner(&format!("Copying local model {}...", name));
        copy_local_model(&src_dir, model_dir)?;
        pb.finish_with_message(format!("Model {} copied from local files.", name));
    } else {
        let repo_id = match name {
            "phi-4-mini-instruct" => "microsoft/Phi-4-mini-instruct",
            "gemma-3-1b-it" => "google/gemma-3-1b-it",
            other => other,
        };

        // Download model files from HF Hub
        let pb = progress::spinner(&format!("Downloading model {}...", repo_id));
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo_id.to_string());

        std::fs::create_dir_all(model_dir)?;

        let files_to_download = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ];

        for filename in &files_to_download {
            match repo.get(filename) {
                Ok(src_path) => {
                    let dest = model_dir.join(filename);
                    if src_path != dest {
                        std::fs::copy(&src_path, &dest)
                            .with_context(|| format!("failed to copy {}", filename))?;
                    }
                }
                Err(e) => {
                    println!("  Warning: could not download {}: {}", filename, e);
                }
            }
        }

        // Download model weights (safetensors)
        // Try model.safetensors first, fall back to model.safetensors.index.json for sharded models
        match repo.get("model.safetensors") {
            Ok(src_path) => {
                let dest = model_dir.join("model.safetensors");
                if src_path != dest {
                    std::fs::copy(&src_path, &dest)?;
                }
            }
            Err(_) => {
                // Try sharded model
                match repo.get("model.safetensors.index.json") {
                    Ok(index_path) => {
                        let dest = model_dir.join("model.safetensors.index.json");
                        if index_path != dest {
                            std::fs::copy(&index_path, &dest)?;
                        }

                        // Parse index to find shard files
                        let index_content = std::fs::read_to_string(&index_path)?;
                        let index: serde_json::Value = serde_json::from_str(&index_content)?;
                        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                            let shard_files: std::collections::HashSet<&str> =
                                weight_map.values().filter_map(|v| v.as_str()).collect();
                            for shard in shard_files {
                                match repo.get(shard) {
                                    Ok(src_path) => {
                                        let dest = model_dir.join(shard);
                                        if src_path != dest {
                                            std::fs::copy(&src_path, &dest)?;
                                        }
                                    }
                                    Err(e) => {
                                        println!("  Warning: could not download shard {}: {}", shard, e);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("  Warning: could not find model weights: {}", e);
                    }
                }
            }
        }
        pb.finish_with_message(format!("Model {} downloaded.", repo_id));

        // Bail if no weight files were downloaded
        let has_weights = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"));

        if !has_weights {
            bail!(
                "No model weight files (.safetensors) were downloaded to {}. \
                 If this is a gated model (e.g. Gemma), set HF_TOKEN or run \
                 `huggingface-cli login` first.",
                model_dir.display()
            );
        }
    }

    // Also download embedding model
    let pb = progress::spinner("Downloading embedding model...");
    let device = compute::device_for(compute::Workload::Embedding);
    let _embedding_model = embeddings::EmbeddingModel::download_and_load(&dirs.embeddings, &device)
        .context("failed to download embedding model")?;
    pb.finish_with_message("Embedding model downloaded.");

    if quantization != "none" {
        // Quantize the model
        let gguf_path = model_dir.join("model.gguf");
        let pb = progress::spinner(&format!("Quantizing model to {}...", quantization));
        rag::quantize_safetensors_to_gguf(model_dir, &gguf_path, quantization, None)
            .context("failed to quantize model")?;
        pb.finish_with_message(format!("Model quantized to {}.", quantization));

        // Keep SafeTensors files alongside GGUF — they are needed for LoRA fine-tuning.
        // GGUF is used for quantized inference; SafeTensors are used for training.

        println!("Model setup complete.");
        println!("  Model: {}", name);
        println!("  Quantization: {}", quantization);
        println!("  GGUF: {}", gguf_path.display());
    } else {
        println!("Model setup complete (no quantization).");
        println!("  Model: {}", name);
        println!("  Format: SafeTensors (full precision)");
    }

    // Update manifest
    manifest.model_name = Some(name.to_string());
    manifest.quantization = Some(quantization.to_string());
    manifest.save(&dirs.root)?;

    Ok(())
}
