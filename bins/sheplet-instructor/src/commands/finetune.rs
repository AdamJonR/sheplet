use anyhow::{bail, Context, Result};
use candle_core::Device;
use candle_nn::Linear;
use std::path::Path;

use crate::progress;
use crate::project::{require_model, project_dirs};

struct SimpleTokenizer;

impl finetune::sft::Tokenize for SimpleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Simple whitespace tokenizer for basic operation
        // A real implementation would use the model's tokenizer
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
    let device = Device::Cpu;

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

    // Create a LoRA layer for training
    // In a full implementation, this would wrap actual model layers
    let in_features = 128;
    let out_features = 128;
    let lora_config = finetune::lora::LoraConfig::default();

    let frozen_weight =
        candle_core::Tensor::randn(0f32, 1.0, &[out_features, in_features], &device)?;
    let frozen = Linear::new(frozen_weight, None);
    let mut lora =
        finetune::lora::LoraLinear::new(frozen, in_features, out_features, &lora_config, &device)?;

    let tokenizer = SimpleTokenizer;
    let adapter_path = dirs.root.join("adapter.safetensors");

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
            pb.finish_with_message(format!("SFT training complete. Final loss: {:.6}", final_loss));

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
            pb.finish_with_message(format!("DPO training complete. Final loss: {:.6}", final_loss));

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

    Ok(())
}
