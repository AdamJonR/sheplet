use anyhow::{bail, Result};
use std::path::Path;

use crate::project::{CourseConfig, require_init};

pub fn run(
    project: &Path,
    system_prompt: Option<&str>,
    retrieval: Option<&str>,
    top_k: Option<usize>,
    relevance_threshold: Option<f64>,
    mmr_lambda: Option<f32>,
) -> Result<()> {
    let _manifest = require_init(project)?;
    let mut config = CourseConfig::load(project).unwrap_or_default();
    let mut changed = false;

    if let Some(prompt) = system_prompt {
        config.system_prompt = prompt.to_string();
        changed = true;
    }
    if let Some(strategy) = retrieval {
        match strategy {
            "top-k" | "mmr" => config.retrieval_strategy = strategy.to_string(),
            _ => bail!("Invalid retrieval strategy: {}. Use 'top-k' or 'mmr'.", strategy),
        }
        changed = true;
    }
    if let Some(k) = top_k {
        config.top_k = k;
        changed = true;
    }
    if let Some(threshold) = relevance_threshold {
        if !(0.0..=1.0).contains(&threshold) {
            bail!("Relevance threshold must be between 0.0 and 1.0");
        }
        config.relevance_threshold = threshold;
        changed = true;
    }
    if let Some(lambda) = mmr_lambda {
        if !(0.0..=1.0).contains(&lambda) {
            bail!("MMR lambda must be between 0.0 and 1.0");
        }
        config.mmr_lambda = lambda;
        changed = true;
    }

    if changed {
        config.save(project)?;
        println!("Configuration updated.");
    }

    println!("Current configuration:");
    println!("{}", serde_json::to_string_pretty(&config)?);
    Ok(())
}
