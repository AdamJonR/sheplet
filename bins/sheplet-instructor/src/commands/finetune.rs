use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::progress;
use crate::project::{require_model, project_dirs, CourseConfig};

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
            rag::ModelArch::Llama => "Llama 3.2",
            rag::ModelArch::Qwen2 => "Qwen2",
            rag::ModelArch::Gemma | rag::ModelArch::Gemma2 => "Gemma",
            rag::ModelArch::Mistral => "Mistral",
        };
        let pb = progress::spinner(&format!("Loading {arch_name} model with LoRA layers..."));

        let mut trainer: Box<dyn finetune::LoraTrainable> = match arch {
            rag::ModelArch::Phi3 => Box::new(
                finetune::Phi3LoraTrainer::new(&dirs.model, &lora_config, &device)
                    .context("failed to load model for training")?,
            ),
            rag::ModelArch::Llama => Box::new(
                finetune::LlamaLoraTrainer::new(&dirs.model, &lora_config, &device)
                    .context("failed to load model for training")?,
            ),
            rag::ModelArch::Qwen2 => Box::new(
                finetune::Qwen2LoraTrainer::new(&dirs.model, &lora_config, &device)
                    .context("failed to load model for training")?,
            ),
            rag::ModelArch::Gemma | rag::ModelArch::Gemma2 => Box::new(
                finetune::GemmaLoraTrainer::new(&dirs.model, &lora_config, &device)
                    .context("failed to load model for training")?,
            ),
            rag::ModelArch::Mistral => Box::new(
                finetune::MistralLoraTrainer::new(&dirs.model, &lora_config, &device)
                    .context("failed to load model for training")?,
            ),
        };
        pb.finish_with_message("Model loaded with LoRA layers.");

        // Train on the same chat template used at inference (system prompt,
        // user/assistant markers, turn-end token) so the adapter sees the
        // token distribution it will actually be used with.
        let system_prompt = CourseConfig::load(project)
            .unwrap_or_default()
            .system_prompt;

        match method {
            "sft" => {
                let pb = progress::spinner("Loading SFT training data...");
                let examples = finetune::data::load_sft_data(data)
                    .context("failed to load SFT data")?;
                pb.finish_with_message(format!("Loaded {} SFT examples.", examples.len()));

                let examples: Vec<finetune::SftExample> = examples
                    .into_iter()
                    .map(|ex| {
                        let (input, output) = rag::format_training_example(
                            arch,
                            &system_prompt,
                            &ex.input,
                            &ex.output,
                        );
                        finetune::SftExample { input, output }
                    })
                    .collect();

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

                let turn_end = rag::turn_end_token(arch);
                let examples: Vec<finetune::DpoExample> = examples
                    .into_iter()
                    .map(|ex| finetune::DpoExample {
                        prompt: rag::prompt::assemble_prompt_for_arch(
                            arch,
                            &system_prompt,
                            &[],
                            &[],
                            &ex.prompt,
                        ),
                        chosen: format!("{}{}", ex.chosen, turn_end),
                        rejected: format!("{}{}", ex.rejected, turn_end),
                    })
                    .collect();

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
        bail!(
            "No SafeTensors model found in {}. Fine-tuning requires full model \
             weights — download a model first with `sheplet-instructor model download`.",
            dirs.model.display()
        );
    }

    Ok(())
}

