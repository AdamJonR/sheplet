mod cli;
mod commands;
mod progress;
mod project;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { course, output } => {
            commands::init::run(&course, &output)?;
        }
        Commands::Templates { project } => {
            commands::templates::run(&project)?;
        }
        Commands::Ingest { sources, project } => {
            commands::ingest::run(&sources, &project).await?;
        }
        Commands::Model {
            name,
            quantization,
            project,
        } => {
            commands::model::run(&name, &quantization, &project)?;
        }
        Commands::Finetune {
            method,
            data,
            project,
            learning_rate,
            epochs,
        } => {
            commands::finetune::run(&method, &data, &project, learning_rate, epochs)?;
        }
        Commands::Config {
            project,
            system_prompt,
            retrieval,
            top_k,
            relevance_threshold,
            mmr_lambda,
        } => {
            commands::config::run(
                &project,
                system_prompt.as_deref(),
                retrieval.as_deref(),
                top_k,
                relevance_threshold,
                mmr_lambda,
            )?;
        }
        Commands::Bundle {
            project,
            output,
            bump_version,
        } => {
            commands::bundle::run(&project, &output, bump_version)?;
        }
    }

    Ok(())
}
