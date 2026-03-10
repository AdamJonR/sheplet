use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "sheplet-instructor")]
#[command(about = "Sheplet instructor CLI — build course bundles for students")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize a new course project directory
    Init {
        /// Course name
        #[arg(long)]
        course: String,
        /// Output directory for the project
        #[arg(long)]
        output: PathBuf,
    },

    /// Generate fine-tuning data template files
    Templates {
        /// Path to the project directory
        #[arg(long)]
        project: PathBuf,
    },

    /// Ingest source documents into the vector database
    Ingest {
        /// Path to the source documents directory
        #[arg(long)]
        sources: PathBuf,
        /// Path to the project directory
        #[arg(long)]
        project: PathBuf,
    },

    /// Download and quantize a model
    Model {
        /// Model name (phi-4-mini-instruct, gemma270m, gemma1b, or HF repo ID)
        #[arg(long, default_value = "phi-4-mini-instruct")]
        name: String,
        /// Quantization level
        #[arg(long, default_value = "q4-k-m")]
        quantization: String,
        /// Path to the project directory
        #[arg(long)]
        project: PathBuf,
    },

    /// Fine-tune the model with LoRA
    Finetune {
        /// Training method: sft or dpo
        #[arg(long)]
        method: String,
        /// Path to the training data JSONL file
        #[arg(long)]
        data: PathBuf,
        /// Path to the project directory
        #[arg(long)]
        project: PathBuf,
        /// Learning rate
        #[arg(long)]
        learning_rate: Option<f64>,
        /// Number of epochs
        #[arg(long)]
        epochs: Option<usize>,
    },

    /// View or update course configuration
    Config {
        /// Path to the project directory
        #[arg(long)]
        project: PathBuf,
        /// Set the system prompt
        #[arg(long)]
        system_prompt: Option<String>,
        /// Set retrieval strategy (top-k or mmr)
        #[arg(long)]
        retrieval: Option<String>,
        /// Set top-k value
        #[arg(long)]
        top_k: Option<usize>,
        /// Set relevance threshold (0.0 - 1.0)
        #[arg(long)]
        relevance_threshold: Option<f64>,
        /// Set MMR lambda (0.0 - 1.0)
        #[arg(long)]
        mmr_lambda: Option<f32>,
    },

    /// Package and sign the project into a .sheplet bundle
    Bundle {
        /// Path to the project directory
        #[arg(long)]
        project: PathBuf,
        /// Output path for the .sheplet bundle file
        #[arg(long)]
        output: PathBuf,
        /// Bump the version number
        #[arg(long)]
        bump_version: bool,
    },
}
