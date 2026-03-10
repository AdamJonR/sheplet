use std::path::PathBuf;
use std::process::Output;

pub const MODEL_CLI_NAME: &str = "gemma270m";
pub const MODEL_DIR_NAME: &str = "gemma-3-transformers-gemma-3-270m-it-v1";

pub const CELL_BIOLOGY_TEXT: &str = r#"Introduction to Cell Biology

The cell is the basic structural and functional unit of all living organisms. Cell theory,
one of the fundamental principles of biology, states three key ideas: all living organisms
are composed of one or more cells, the cell is the basic unit of life, and all cells arise
from pre-existing cells. This theory was developed through the contributions of scientists
such as Matthias Schleiden, Theodor Schwann, and Rudolf Virchow in the 19th century.

Cells are broadly classified into two types: prokaryotic and eukaryotic. Prokaryotic cells,
found in bacteria and archaea, lack a membrane-bound nucleus and other membrane-bound
organelles. Eukaryotic cells, found in animals, plants, fungi, and protists, contain a
well-defined nucleus enclosed by a nuclear envelope, along with various specialized organelles.

Key organelles in eukaryotic cells include:

The nucleus serves as the control center of the cell, housing the cell's DNA organized into
chromosomes. The nuclear envelope is a double membrane with nuclear pores that regulate the
transport of molecules between the nucleus and the cytoplasm.

Mitochondria are the powerhouses of the cell, responsible for cellular respiration and ATP
production. They have a double membrane structure with an inner membrane folded into cristae,
which increases the surface area for the electron transport chain. Mitochondria contain their
own DNA, supporting the endosymbiotic theory.

The endoplasmic reticulum (ER) comes in two forms: rough ER, studded with ribosomes and
involved in protein synthesis and modification, and smooth ER, which lacks ribosomes and is
involved in lipid synthesis, detoxification, and calcium storage.

The Golgi apparatus processes, packages, and ships proteins and lipids received from the ER.
It consists of stacked membrane-bound sacs called cisternae. Proteins move through the cis,
medial, and trans compartments before being sorted and shipped to their final destinations.

The cell membrane (plasma membrane) is a phospholipid bilayer embedded with proteins. It
follows the fluid mosaic model, where the membrane is fluid and proteins float within it.
The membrane is selectively permeable, controlling the passage of substances into and out
of the cell.

Osmosis is the movement of water molecules across a selectively permeable membrane from an
area of lower solute concentration to an area of higher solute concentration. In a hypotonic
solution, water moves into the cell, potentially causing it to swell and burst (lysis). In a
hypertonic solution, water moves out of the cell, causing it to shrink (crenation). In an
isotonic solution, there is no net movement of water.

Lysosomes are membrane-bound organelles containing digestive enzymes that break down waste
materials, cellular debris, and foreign invaders. They maintain an acidic internal pH of
about 4.5-5.0, which is optimal for their hydrolytic enzymes."#;

pub const GENETICS_TEXT: &str = r#"Fundamentals of Genetics

Genetics is the branch of biology that studies genes, heredity, and genetic variation in
living organisms. The field began with the work of Gregor Mendel, an Augustinian friar who
conducted breeding experiments with pea plants in the mid-1800s.

DNA (deoxyribonucleic acid) is the hereditary material in nearly all organisms. It has a
double helix structure, discovered by James Watson and Francis Crick in 1953, with
contributions from Rosalind Franklin's X-ray crystallography data. DNA consists of two
complementary strands made of nucleotides, each containing a sugar (deoxyribose), a phosphate
group, and one of four nitrogenous bases: adenine (A), thymine (T), guanine (G), and cytosine
(C). Base pairing rules dictate that A pairs with T and G pairs with C.

Chromosomes are structures made of DNA tightly coiled around histone proteins. Humans have
46 chromosomes arranged in 23 pairs — 22 pairs of autosomes and one pair of sex chromosomes
(XX in females, XY in males). Each chromosome contains many genes, which are specific
segments of DNA that code for proteins.

Mendel's Laws of Inheritance:

The Law of Segregation states that during gamete formation, the two alleles for each gene
separate (segregate) so that each gamete carries only one allele for each trait. This occurs
during meiosis I when homologous chromosomes are separated.

The Law of Independent Assortment states that genes located on different chromosomes are
inherited independently of each other. During meiosis, the orientation of each pair of
homologous chromosomes is random, leading to various combinations of maternal and paternal
chromosomes in the gametes.

The Law of Dominance states that in a heterozygous organism, only the dominant allele is
expressed in the phenotype, while the recessive allele is masked. A dominant allele is
typically represented by an uppercase letter (e.g., B) and a recessive allele by a lowercase
letter (e.g., b).

An allele is one of the variant forms of a gene at a particular locus on a chromosome.
Organisms that have two identical alleles for a gene are homozygous (e.g., BB or bb), while
organisms with two different alleles are heterozygous (e.g., Bb). The genotype refers to
the genetic makeup of an organism, while the phenotype is the observable physical
characteristic resulting from the genotype.

RNA (ribonucleic acid) differs from DNA in several ways: it is typically single-stranded,
uses ribose sugar instead of deoxyribose, and contains uracil (U) instead of thymine (T).
The three main types of RNA are messenger RNA (mRNA), which carries the genetic code from DNA
to ribosomes; transfer RNA (tRNA), which brings amino acids to the ribosome during translation;
and ribosomal RNA (rRNA), which makes up part of the ribosome structure.

Protein synthesis occurs in two main stages: transcription (DNA to mRNA in the nucleus) and
translation (mRNA to protein at the ribosome). During transcription, RNA polymerase reads the
template strand of DNA and synthesizes a complementary mRNA strand. During translation,
ribosomes read the mRNA codons (three-nucleotide sequences) and tRNA molecules deliver the
corresponding amino acids to build a polypeptide chain.

Mutations are changes in the DNA sequence that can occur spontaneously or be induced by
mutagens. Point mutations affect a single nucleotide, while chromosomal mutations affect
larger segments of DNA. Mutations can be silent (no effect), missense (different amino acid),
or nonsense (premature stop codon)."#;

pub const DPO_DATA: &str = r#"{"prompt":"What is the basic unit of life?","chosen":"The cell is the basic structural and functional unit of all living organisms. According to cell theory, all living organisms are composed of one or more cells, the cell is the basic unit of life, and all cells arise from pre-existing cells.","rejected":"Atoms are the basic unit of life because everything is made of atoms."}
{"prompt":"What are the two main types of cells?","chosen":"Cells are broadly classified into prokaryotic and eukaryotic. Prokaryotic cells, found in bacteria and archaea, lack a membrane-bound nucleus. Eukaryotic cells, found in animals, plants, fungi, and protists, contain a well-defined nucleus and specialized organelles.","rejected":"There are plant cells and animal cells, and those are the only two types."}
{"prompt":"What is the function of mitochondria?","chosen":"Mitochondria are the powerhouses of the cell, responsible for cellular respiration and ATP production. They have a double membrane structure with an inner membrane folded into cristae, which increases surface area for the electron transport chain.","rejected":"Mitochondria store DNA and control cell division."}
{"prompt":"Describe the structure of DNA.","chosen":"DNA has a double helix structure consisting of two complementary strands of nucleotides. Each nucleotide contains a deoxyribose sugar, a phosphate group, and one of four nitrogenous bases: adenine (A), thymine (T), guanine (G), and cytosine (C). A pairs with T and G pairs with C.","rejected":"DNA is a single strand of proteins that carries genetic information in a linear chain."}
{"prompt":"What is Mendel's Law of Segregation?","chosen":"The Law of Segregation states that during gamete formation, the two alleles for each gene separate so that each gamete carries only one allele for each trait. This occurs during meiosis I when homologous chromosomes are separated.","rejected":"The Law of Segregation says that dominant traits always separate from recessive traits during cell growth."}
{"prompt":"What is osmosis?","chosen":"Osmosis is the movement of water molecules across a selectively permeable membrane from an area of lower solute concentration to an area of higher solute concentration. In a hypotonic solution, water moves into the cell; in a hypertonic solution, water moves out.","rejected":"Osmosis is the movement of all molecules across any membrane in both directions equally."}
{"prompt":"What is the role of the Golgi apparatus?","chosen":"The Golgi apparatus processes, packages, and ships proteins and lipids received from the endoplasmic reticulum. It consists of stacked membrane-bound sacs called cisternae, and proteins move through cis, medial, and trans compartments before being sorted.","rejected":"The Golgi apparatus is where DNA replication occurs inside the cell."}
{"prompt":"What are alleles?","chosen":"An allele is one of the variant forms of a gene at a particular locus on a chromosome. Organisms with two identical alleles are homozygous (e.g., BB or bb), while organisms with two different alleles are heterozygous (e.g., Bb).","rejected":"Alleles are different types of chromosomes found only in reproductive cells."}
{"prompt":"How does the cell membrane regulate transport?","chosen":"The cell membrane is a phospholipid bilayer embedded with proteins, following the fluid mosaic model. It is selectively permeable, controlling the passage of substances into and out of the cell through various transport mechanisms.","rejected":"The cell membrane lets everything pass through freely since it has large holes."}
{"prompt":"What is the difference between DNA and RNA?","chosen":"RNA differs from DNA in several ways: it is typically single-stranded, uses ribose sugar instead of deoxyribose, and contains uracil (U) instead of thymine (T). The three main types are mRNA, tRNA, and rRNA.","rejected":"DNA and RNA are identical molecules, the names are just used interchangeably."}
{"prompt":"Explain the Law of Independent Assortment.","chosen":"The Law of Independent Assortment states that genes on different chromosomes are inherited independently. During meiosis, the orientation of each pair of homologous chromosomes is random, leading to various combinations of maternal and paternal chromosomes in gametes.","rejected":"Independent assortment means all genes are always inherited together as a complete set from one parent."}
{"prompt":"What are lysosomes and what do they do?","chosen":"Lysosomes are membrane-bound organelles containing digestive enzymes that break down waste materials, cellular debris, and foreign invaders. They maintain an acidic internal pH of about 4.5-5.0, optimal for their hydrolytic enzymes.","rejected":"Lysosomes are found outside cells and help with photosynthesis in plants."}
{"prompt":"What is the endoplasmic reticulum?","chosen":"The endoplasmic reticulum comes in two forms: rough ER, studded with ribosomes and involved in protein synthesis and modification, and smooth ER, which lacks ribosomes and is involved in lipid synthesis, detoxification, and calcium storage.","rejected":"The endoplasmic reticulum is a type of organelle found only in prokaryotic cells that produces energy."}
{"prompt":"How many chromosomes do humans have?","chosen":"Humans have 46 chromosomes arranged in 23 pairs — 22 pairs of autosomes and one pair of sex chromosomes (XX in females, XY in males). Each chromosome contains many genes that code for proteins.","rejected":"Humans have 24 chromosomes total, with 12 from each parent."}
{"prompt":"What happens in transcription?","chosen":"During transcription, RNA polymerase reads the template strand of DNA and synthesizes a complementary mRNA strand. This occurs in the nucleus and is the first stage of protein synthesis, converting the genetic code from DNA to mRNA.","rejected":"Transcription is when proteins are directly assembled from DNA without any intermediate steps."}
{"prompt":"What is the Law of Dominance?","chosen":"The Law of Dominance states that in a heterozygous organism, only the dominant allele is expressed in the phenotype, while the recessive allele is masked. Dominant alleles are represented by uppercase letters (e.g., B) and recessive by lowercase (e.g., b).","rejected":"The Law of Dominance says that all traits blend together equally in offspring."}
{"prompt":"What is the difference between genotype and phenotype?","chosen":"The genotype refers to the genetic makeup of an organism (e.g., BB, Bb, or bb), while the phenotype is the observable physical characteristic resulting from the genotype. A dominant allele can mask a recessive allele in the phenotype.","rejected":"Genotype and phenotype mean the same thing — they both refer to how an organism looks."}
{"prompt":"What are the types of mutations?","chosen":"Mutations are changes in the DNA sequence that can be spontaneous or induced by mutagens. Point mutations affect a single nucleotide, while chromosomal mutations affect larger segments. They can be silent, missense (different amino acid), or nonsense (premature stop codon).","rejected":"Mutations only happen when organisms are exposed to radiation and always cause cancer."}
{"prompt":"What happens to a cell in a hypertonic solution?","chosen":"In a hypertonic solution, water moves out of the cell by osmosis, causing the cell to shrink — a process called crenation. This occurs because the solute concentration is higher outside the cell, drawing water out across the selectively permeable membrane.","rejected":"In a hypertonic solution, the cell absorbs extra water and grows larger until it divides."}
{"prompt":"What is the endosymbiotic theory?","chosen":"The endosymbiotic theory proposes that mitochondria (and chloroplasts) were once free-living prokaryotes that were engulfed by ancestral eukaryotic cells. Evidence includes the fact that mitochondria contain their own DNA and have a double membrane structure.","rejected":"The endosymbiotic theory states that all cells evolved from viruses that merged together."}"#;

/// Returns the workspace root directory (parent of bins/ and crates/).
pub fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // bins/sheplet-student -> workspace root
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("Could not find workspace root from CARGO_MANIFEST_DIR")
        .to_path_buf()
}

/// Check if the gemma270m model is available in downloaded-models/.
pub fn gemma270m_available() -> bool {
    let model_dir = workspace_root()
        .join("downloaded-models")
        .join(MODEL_DIR_NAME);
    model_dir.exists() && model_dir.is_dir()
}

/// Create an `assert_cmd::Command` for the sheplet-instructor binary.
#[allow(deprecated)]
pub fn instructor_cmd() -> assert_cmd::Command {
    assert_cmd::Command::cargo_bin("sheplet-instructor").expect("sheplet-instructor binary not found")
}

/// Result of running the instructor CLI pipeline (init through bundle).
#[allow(dead_code)]
pub struct PipelineResult {
    pub bundle_path: PathBuf,
    pub fingerprint: String,
    pub project_dir: PathBuf,
    // Hold tmpdir to keep it alive; dropped when PipelineResult is dropped.
    pub _tmpdir: tempfile::TempDir,
}

/// Run the full instructor pipeline: init → ingest → model → finetune → config → bundle.
/// Returns the bundle path, fingerprint, and project directory.
pub fn run_instructor_pipeline(course_name: &str, bundle_name: &str) -> PipelineResult {
    let ws_root = workspace_root();
    let tmpdir = tempfile::tempdir().expect("Failed to create temp dir");
    let project_dir = tmpdir.path().to_path_buf();

    // Init
    println!("=== Init ===");
    let start = std::time::Instant::now();
    instructor_cmd()
        .args(["init", "--course", course_name, "--output"])
        .arg(&project_dir)
        .assert()
        .success();
    println!("  Init: {:?}", start.elapsed());

    // Create documents
    let sources_dir = project_dir.join("sources");
    std::fs::create_dir_all(&sources_dir).unwrap();
    std::fs::write(sources_dir.join("cell_biology.txt"), CELL_BIOLOGY_TEXT).unwrap();
    std::fs::write(sources_dir.join("genetics_basics.txt"), GENETICS_TEXT).unwrap();

    // Ingest
    println!("=== Ingest ===");
    let start = std::time::Instant::now();
    instructor_cmd()
        .args(["ingest", "--sources"])
        .arg(&sources_dir)
        .args(["--project"])
        .arg(&project_dir)
        .assert()
        .success();
    println!("  Ingest: {:?}", start.elapsed());

    // Model
    println!("=== Model ===");
    let start = std::time::Instant::now();
    instructor_cmd()
        .current_dir(&ws_root)
        .args(["model", "--name", MODEL_CLI_NAME, "--quantization", "none", "--project"])
        .arg(&project_dir)
        .assert()
        .success();
    println!("  Model: {:?}", start.elapsed());

    // DPO data + finetune
    let dpo_path = project_dir.join("dpo_data.jsonl");
    std::fs::write(&dpo_path, DPO_DATA).unwrap();
    println!("=== Finetune (DPO) ===");
    let start = std::time::Instant::now();
    instructor_cmd()
        .current_dir(&ws_root)
        .args(["finetune", "--method", "dpo", "--data"])
        .arg(&dpo_path)
        .args(["--project"])
        .arg(&project_dir)
        .args(["--epochs", "1"])
        .assert()
        .success();
    println!("  Finetune: {:?}", start.elapsed());

    // Config
    println!("=== Config ===");
    instructor_cmd()
        .args(["config", "--project"])
        .arg(&project_dir)
        .args([
            "--system-prompt",
            "You are a helpful biology tutor. Answer questions accurately using course materials.",
        ])
        .assert()
        .success();

    // Bundle
    println!("=== Bundle ===");
    let bundle_path = tmpdir.path().join(bundle_name);
    let start = std::time::Instant::now();
    let bundle_output: Output = instructor_cmd()
        .args(["bundle", "--project"])
        .arg(&project_dir)
        .args(["--output"])
        .arg(&bundle_path)
        .output()
        .expect("Failed to run bundle command");
    assert!(
        bundle_output.status.success(),
        "Bundle failed: {}",
        String::from_utf8_lossy(&bundle_output.stderr)
    );
    println!("  Bundle: {:?}", start.elapsed());

    let bundle_stdout = String::from_utf8_lossy(&bundle_output.stdout);
    let fingerprint = extract_fingerprint(&bundle_stdout)
        .expect("Could not find fingerprint in bundle output");

    assert!(bundle_path.exists(), "Bundle file should exist");
    let bundle_size = std::fs::metadata(&bundle_path).unwrap().len();
    assert!(bundle_size > 1024, "Bundle should be >1KB, got {bundle_size} bytes");
    println!("  Bundle size: {bundle_size} bytes, fingerprint: {fingerprint}");

    PipelineResult {
        bundle_path,
        fingerprint,
        project_dir,
        _tmpdir: tmpdir,
    }
}

/// Extract fingerprint from bundle command output.
/// Looks for a pattern like "Fingerprint: abcdef0123456789" (16 hex chars).
pub fn extract_fingerprint(output: &str) -> Option<String> {
    const FINGERPRINT_HEX_LEN: usize = 16;
    for line in output.lines() {
        if let Some(pos) = line.to_lowercase().find("fingerprint") {
            let after = &line[pos..];
            for word in after.split_whitespace().skip(1) {
                let cleaned = word.trim_matches(|c: char| !c.is_ascii_hexdigit());
                if cleaned.len() == FINGERPRINT_HEX_LEN
                    && cleaned.chars().all(|c| c.is_ascii_hexdigit())
                {
                    return Some(cleaned.to_string());
                }
            }
        }
    }
    None
}
