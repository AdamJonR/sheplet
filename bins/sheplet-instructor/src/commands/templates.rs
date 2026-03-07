use anyhow::Result;
use std::path::Path;

use crate::project::{require_init, project_dirs};

const DPO_TEMPLATE: &str = r#"{"prompt": "What is photosynthesis?", "chosen": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.", "rejected": "I think it has something to do with plants and light, maybe they eat the sun or something."}
{"prompt": "Explain the water cycle.", "chosen": "The water cycle describes the continuous movement of water within the Earth and atmosphere. It involves evaporation from surface water, transpiration from plants, condensation into clouds, and precipitation back to the surface as rain or snow.", "rejected": "Water goes up and comes back down."}
"#;

const SFT_TEMPLATE: &str = r#"{"input": "What is photosynthesis?", "output": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct."}
{"input": "Explain the water cycle.", "output": "The water cycle describes the continuous movement of water within the Earth and atmosphere. It involves evaporation from surface water, transpiration from plants, condensation into clouds, and precipitation back to the surface as rain or snow."}
"#;

pub fn run(project: &Path) -> Result<()> {
    let _manifest = require_init(project)?;
    let dirs = project_dirs(project);

    std::fs::create_dir_all(&dirs.finetune_data)?;

    let dpo_path = dirs.finetune_data.join("dpo_template.jsonl");
    let sft_path = dirs.finetune_data.join("sft_template.jsonl");

    std::fs::write(&dpo_path, DPO_TEMPLATE)?;
    std::fs::write(&sft_path, SFT_TEMPLATE)?;

    println!("Generated template files:");
    println!("  DPO: {}", dpo_path.display());
    println!("  SFT: {}", sft_path.display());
    println!("\nEdit these files with your course-specific examples, then run `sheplet-instructor finetune`.");
    Ok(())
}
