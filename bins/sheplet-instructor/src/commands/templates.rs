use anyhow::Result;
use std::path::Path;

use crate::project::{require_init, project_dirs};

const DPO_TEMPLATE: &str = r#"{"prompt": "When was Rome traditionally founded?", "chosen": "Rome was traditionally founded in 753 BC, a date calculated by the Roman scholar Varro. According to legend, the city was established by Romulus, who became its first king, though archaeology shows settlements existed on the site from at least the 8th century BC.", "rejected": "I think Rome was founded sometime in the Middle Ages, maybe by some merchants or something."}
{"prompt": "What does SPQR stand for?", "chosen": "SPQR stands for Senatus Populusque Romanus, meaning the Senate and the People of Rome. It appeared on official documents, monuments, and standards, symbolizing the shared authority of the governing council and the citizens.", "rejected": "It's some kind of Roman abbreviation."}
"#;

const SFT_TEMPLATE: &str = r#"{"input": "When was Rome traditionally founded?", "output": "Rome was traditionally founded in 753 BC, a date calculated by the Roman scholar Varro. According to legend, the city was established by Romulus, who became its first king, though archaeology shows settlements existed on the site from at least the 8th century BC."}
{"input": "What does SPQR stand for?", "output": "SPQR stands for Senatus Populusque Romanus, meaning the Senate and the People of Rome. It appeared on official documents, monuments, and standards, symbolizing the shared authority of the governing council and the citizens."}
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
