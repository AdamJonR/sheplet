use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::progress;
use crate::project::{require_init, project_dirs, local_model_source, copy_local_model};

pub fn run(name: &str, project: &Path) -> Result<()> {
    let mut manifest = require_init(project)?;
    let dirs = project_dirs(project);

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
            "phi-3-mini-4k-instruct" => "microsoft/Phi-3-mini-4k-instruct",
            "llama-3.2-1b" => "meta-llama/Llama-3.2-1B-Instruct",
            "llama-3.2-3b" => "meta-llama/Llama-3.2-3B-Instruct",
            "qwen2.5-0.5b" => "Qwen/Qwen2.5-0.5B-Instruct",
            "qwen2.5-1.5b" => "Qwen/Qwen2.5-1.5B-Instruct",
            "qwen2.5-3b" => "Qwen/Qwen2.5-3B-Instruct",
            "gemma-2b" => "google/gemma-2b-it",
            "gemma-2-2b" => "google/gemma-2-2b-it",
            "mistral-7b" => "mistralai/Mistral-7B-Instruct-v0.3",
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
                 If this is a gated model, set HF_TOKEN or run \
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

    println!("Model setup complete.");
    println!("  Model: {}", name);
    println!("  Format: SafeTensors");

    // Update manifest
    manifest.model_name = Some(name.to_string());
    manifest.save(&dirs.root)?;

    Ok(())
}
