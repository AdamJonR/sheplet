use anyhow::{bail, Context, Result};
use candle_nn::Linear;
use std::path::Path;

use crate::progress;
use crate::project::{require_model, project_dirs};

struct SimpleTokenizer;

impl finetune::sft::Tokenize for SimpleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| i as u32)
            .collect())
    }
}

pub fn run(
    method: &str,
    data: &Path,
    project: &Path,
    learning_rate: Option<f64>,
    epochs: Option<usize>,
) -> Result<()> {
    let _manifest = require_model(project)?;
    let dirs = project_dirs(project);
    let device = compute::device_for(compute::Workload::Training);

    // Hardware preflight
    let report = finetune::preflight::preflight_check(16.0);
    println!("Hardware check:");
    println!("  Available RAM: {:.1} GB", report.hardware.available_ram_gb);
    println!("  CPU cores: {}", report.hardware.cpu_count);
    if !report.is_sufficient {
        println!(
            "  Warning: Recommended {:.0} GB RAM, you have {:.1} GB",
            report.recommended_ram_gb, report.hardware.available_ram_gb
        );
    }

    let lora_config = finetune::lora::LoraConfig::default();
    let adapter_path = dirs.root.join("adapter.safetensors");

    // Check if we have a full model with SafeTensors for full LoRA training
    let has_safetensors = dirs.model.join("config.json").exists()
        && std::fs::read_dir(&dirs.model)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| {
                        e.path()
                            .extension()
                            .is_some_and(|ext| ext == "safetensors")
                    })
            })
            .unwrap_or(false);

    if has_safetensors {
        // Full model LoRA training path — detect architecture and load trainer
        let arch = rag::detect_model_arch(&dirs.model)
            .context("failed to detect model architecture")?;

        println!("Loading model for LoRA fine-tuning...");
        let arch_name = match arch {
            rag::ModelArch::Phi3 => "Phi-3",
            rag::ModelArch::Gemma3 => "Gemma 3",
        };
        let pb = progress::spinner(&format!("Loading {arch_name} model with LoRA layers..."));

        let mut trainer: Box<dyn finetune::LoraTrainable> = match arch {
            rag::ModelArch::Phi3 => Box::new(
                finetune::Phi3LoraTrainer::new(&dirs.model, &lora_config, &device)
                    .context("failed to load model for training")?,
            ),
            rag::ModelArch::Gemma3 => Box::new(
                finetune::Gemma3LoraTrainer::new(&dirs.model, &lora_config, &device)
                    .context("failed to load model for training")?,
            ),
        };
        pb.finish_with_message("Model loaded with LoRA layers.");

        match method {
            "sft" => {
                let pb = progress::spinner("Loading SFT training data...");
                let examples = finetune::data::load_sft_data(data)
                    .context("failed to load SFT data")?;
                pb.finish_with_message(format!("Loaded {} SFT examples.", examples.len()));

                let mut config = finetune::sft::SftConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let pb = progress::spinner("Training SFT with full model...");
                let final_loss = finetune::sft::train_sft_full(
                    trainer.as_mut(),
                    &examples,
                    &config,
                )?;
                pb.finish_with_message(format!(
                    "SFT training complete. Final loss: {:.6}",
                    final_loss
                ));

                trainer
                    .save_adapter(&adapter_path)
                    .context("failed to save adapter")?;
                println!("Adapter saved to {}", adapter_path.display());
            }
            "dpo" => {
                let pb = progress::spinner("Loading DPO training data...");
                let examples = finetune::data::load_dpo_data(data)
                    .context("failed to load DPO data")?;
                pb.finish_with_message(format!("Loaded {} DPO examples.", examples.len()));

                let mut config = finetune::dpo::DpoConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let pb = progress::spinner("Training DPO with full model...");
                let final_loss = finetune::dpo::train_dpo_full(
                    trainer.as_mut(),
                    &examples,
                    &config,
                )?;
                pb.finish_with_message(format!(
                    "DPO training complete. Final loss: {:.6}",
                    final_loss
                ));

                trainer
                    .save_adapter(&adapter_path)
                    .context("failed to save adapter")?;
                println!("Adapter saved to {}", adapter_path.display());
            }
            _ => bail!("Unknown training method: {}. Use 'sft' or 'dpo'.", method),
        }
    } else {
        // Standalone LoRA training (no full model weights available)
        println!("No SafeTensors model found; using standalone LoRA training.");

        let in_features = 128;
        let out_features = 128;

        let frozen_weight =
            candle_core::Tensor::randn(0f32, 1.0, &[out_features, in_features], &device)?;
        let frozen = Linear::new(frozen_weight, None);
        let mut lora = finetune::lora::LoraLinear::new(
            frozen,
            in_features,
            out_features,
            &lora_config,
            &device,
        )?;

        let tokenizer = SimpleTokenizer;

        match method {
            "sft" => {
                let pb = progress::spinner("Loading SFT training data...");
                let examples = finetune::data::load_sft_data(data)
                    .context("failed to load SFT data")?;
                pb.finish_with_message(format!("Loaded {} SFT examples.", examples.len()));

                let mut config = finetune::sft::SftConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let pb = progress::spinner("Training SFT...");
                let final_loss = finetune::sft::train_sft(
                    &mut lora,
                    &examples,
                    &config,
                    &tokenizer,
                    &device,
                )?;
                pb.finish_with_message(format!(
                    "SFT training complete. Final loss: {:.6}",
                    final_loss
                ));

                lora.save(&adapter_path)?;
                println!("Adapter saved to {}", adapter_path.display());
            }
            "dpo" => {
                let pb = progress::spinner("Loading DPO training data...");
                let examples = finetune::data::load_dpo_data(data)
                    .context("failed to load DPO data")?;
                pb.finish_with_message(format!("Loaded {} DPO examples.", examples.len()));

                let mut config = finetune::dpo::DpoConfig::default();
                if let Some(lr) = learning_rate {
                    config.learning_rate = lr;
                }
                if let Some(ep) = epochs {
                    config.epochs = ep;
                }

                let pb = progress::spinner("Training DPO...");
                let final_loss = finetune::dpo::train_dpo(
                    &mut lora,
                    &examples,
                    &config,
                    &tokenizer,
                    &device,
                )?;
                pb.finish_with_message(format!(
                    "DPO training complete. Final loss: {:.6}",
                    final_loss
                ));

                lora.save(&adapter_path)?;
                println!("Adapter saved to {}", adapter_path.display());
            }
            _ => bail!("Unknown training method: {}. Use 'sft' or 'dpo'.", method),
        }

        // Save checkpoint
        let checkpoint_dir = dirs.root.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;
        let meta = finetune::checkpoint::CheckpointMeta {
            epoch: epochs.unwrap_or(3),
            step: 0,
            loss: 0.0,
            lora_config: lora_config.clone(),
        };
        finetune::checkpoint::save_checkpoint(&lora, &meta, &checkpoint_dir)?;
        println!("Checkpoint saved to {}", checkpoint_dir.display());
    }

    Ok(())
}

