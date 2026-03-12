#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Sheplet E2E Manual Test Script
# Exercises the full instructor pipeline: init, ingest, model, finetune, config, bundle
# =============================================================================

# --- Step 0: Environment -----------------------------------------------------

export PATH="$HOME/.cargo/bin:$HOME/miniconda3/envs/ml-env/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_DIR="$PROJECT_ROOT/test-project"
INSTRUCTOR="$PROJECT_ROOT/target/release/sheplet-instructor"

usage() {
    echo "Usage: $0 [--model llama1b|llama3b] [--quantization none|q4-k-m|q5-k-m|q8-0|q4-0] [--lora yes|no]"
    echo ""
    echo "Defaults: --model llama1b --quantization none --lora yes"
    exit 1
}

# Defaults
MODEL="llama1b"
QUANTIZATION="none"
LORA="yes"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --lora)       LORA="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

case "$MODEL" in
  llama1b)  MODEL_NAME="llama-3.2-1b" ;;
  llama3b)  MODEL_NAME="llama-3.2-3b" ;;
  *)        echo "Unknown model: $MODEL"; usage ;;
esac

case "$QUANTIZATION" in
  none|q4-k-m|q5-k-m|q8-0|q4-0) ;;
  *) echo "Unknown quantization: $QUANTIZATION"; usage ;;
esac

case "$LORA" in
  yes|no) ;;
  *) echo "Invalid --lora value: $LORA (must be yes or no)"; usage ;;
esac

echo "Model: $MODEL_NAME (quantization: $QUANTIZATION, lora: $LORA)"

TOTAL_START=$SECONDS

# --- Detect GPU features ----------------------------------------------------
source "$SCRIPT_DIR/detect_gpu.sh"

# --- Step 1: Cleanup ---------------------------------------------------------

if [ -d "$TEST_DIR" ]; then
    rm -rf "$TEST_DIR"
    echo "Cleaned previous test run"
fi

# --- Step 2: Build ------------------------------------------------------------

echo ""
echo "=== Build ==="
STEP_START=$SECONDS
cargo build --release -p sheplet-instructor $CARGO_FEATURES --manifest-path "$PROJECT_ROOT/Cargo.toml"
TIME_BUILD=$(( SECONDS - STEP_START ))
echo "--- Build completed in ${TIME_BUILD}s ---"

# --- Step 3: Init -------------------------------------------------------------

echo ""
echo "=== Init ==="
STEP_START=$SECONDS
"$INSTRUCTOR" init --course "Basic Biology" --output "$TEST_DIR"
TIME_INIT=$(( SECONDS - STEP_START ))
echo "--- Init completed in ${TIME_INIT}s ---"

# --- Step 4: Create test documents --------------------------------------------

mkdir -p "$TEST_DIR/sources"

cat > "$TEST_DIR/sources/cell_biology.txt" << 'CELL_EOF'
Introduction to Cell Biology

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
about 4.5-5.0, which is optimal for their hydrolytic enzymes.

The cytoskeleton is a network of protein filaments that provides structural support, aids in
cell movement, and plays a role in intracellular transport. It consists of three main types
of filaments: microfilaments (actin filaments), intermediate filaments, and microtubules.
Microfilaments are the thinnest, made of actin, and are involved in muscle contraction, cell
division, and cell shape. Microtubules are the thickest, made of tubulin, and form the
mitotic spindle during cell division, as well as the structural core of cilia and flagella.

Cell division occurs through two main processes: mitosis and meiosis. Mitosis produces two
genetically identical daughter cells and is used for growth and repair. It consists of
prophase, metaphase, anaphase, and telophase, followed by cytokinesis. During prophase,
chromatin condenses into visible chromosomes and the mitotic spindle begins to form. In
metaphase, chromosomes align at the cell's equator. During anaphase, sister chromatids
separate and move to opposite poles. In telophase, nuclear envelopes reform around each set
of chromosomes.

Meiosis is a specialized form of cell division that produces four genetically unique haploid
cells (gametes). It involves two rounds of division: meiosis I (reductional division) and
meiosis II (equational division). Crossing over occurs during prophase I, where homologous
chromosomes exchange genetic material, increasing genetic diversity. This is a key source
of genetic variation in sexually reproducing organisms.

Active transport requires energy (ATP) to move substances against their concentration
gradient. The sodium-potassium pump (Na+/K+ ATPase) is a well-known example, pumping 3
sodium ions out of the cell and 2 potassium ions into the cell per ATP molecule hydrolyzed.
This pump is essential for maintaining the cell's resting membrane potential and is critical
for nerve impulse transmission.
CELL_EOF

cat > "$TEST_DIR/sources/genetics_basics.txt" << 'GENETICS_EOF'
Fundamentals of Genetics

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
or nonsense (premature stop codon).

Epigenetics refers to heritable changes in gene expression that do not involve changes to the
DNA sequence itself. Key epigenetic mechanisms include DNA methylation, where methyl groups are
added to cytosine bases (often at CpG dinucleotides), typically silencing gene expression.
Histone modification is another mechanism, where chemical groups (acetyl, methyl, phosphate)
are added to histone tails, altering chromatin structure and gene accessibility. Acetylation
generally promotes gene expression by loosening chromatin, while methylation can either
activate or repress genes depending on the specific residue modified.

Genetic linkage occurs when genes are located close together on the same chromosome and tend
to be inherited together. The frequency of recombination between linked genes depends on
their physical distance on the chromosome, measured in centimorgans (cM). Thomas Hunt Morgan
demonstrated linkage using Drosophila (fruit flies), showing that some traits did not assort
independently as predicted by Mendel's second law.

Polymerase chain reaction (PCR) is a laboratory technique used to amplify specific DNA
sequences. It involves repeated cycles of denaturation (separating DNA strands at ~95°C),
annealing (primers binding at ~55-65°C), and extension (DNA polymerase synthesizing new
strands at ~72°C). PCR can produce millions of copies of a target sequence from a tiny
amount of starting material and is widely used in diagnostics, forensics, and research.
GENETICS_EOF

cat > "$TEST_DIR/sources/ecology_basics.txt" << 'ECOLOGY_EOF'
Fundamentals of Ecology

Ecology is the scientific study of the interactions between organisms and their environment.
It encompasses the study of individuals, populations, communities, ecosystems, and the
biosphere. The term was coined by Ernst Haeckel in 1866, derived from the Greek words
"oikos" (house) and "logos" (study).

A population is a group of individuals of the same species living in a particular area at a
given time. Population ecology studies factors that affect population size and growth,
including birth rates, death rates, immigration, and emigration. Population growth can follow
two main models: exponential growth (J-shaped curve) occurs when resources are unlimited,
described by dN/dt = rN, where r is the intrinsic rate of increase. Logistic growth (S-shaped
curve) occurs when resources are limited, described by dN/dt = rN(1 - N/K), where K is the
carrying capacity of the environment.

A community consists of all the populations of different species living and interacting in a
particular area. Species interactions include competition (both interspecific and
intraspecific), predation, mutualism, commensalism, and parasitism. The competitive exclusion
principle (Gause's principle) states that two species competing for the same limited resource
cannot coexist indefinitely — one will outcompete the other.

An ecosystem includes all the living organisms (biotic factors) and non-living components
(abiotic factors) in a given area, along with their interactions. Energy flows through
ecosystems in one direction: from the sun to producers (autotrophs) to consumers
(heterotrophs) to decomposers. Only about 10% of energy is transferred from one trophic
level to the next (the 10% rule), with the rest lost as heat through cellular respiration.

Biogeochemical cycles describe the movement of chemical elements through the biotic and
abiotic components of ecosystems. The carbon cycle involves photosynthesis (removing CO2 from
the atmosphere), cellular respiration (releasing CO2), decomposition, and combustion of fossil
fuels. The nitrogen cycle includes nitrogen fixation (converting N2 to NH3 by bacteria),
nitrification (NH3 to NO3-), assimilation by plants, and denitrification (NO3- back to N2).

Ecological succession is the process of change in the species structure of an ecological
community over time. Primary succession occurs on bare surfaces where no soil exists (e.g.,
after a volcanic eruption), beginning with pioneer species like lichens. Secondary succession
occurs in areas where a community has been disturbed but soil remains (e.g., after a fire).
Both types progress toward a climax community, a relatively stable end stage.

Biodiversity refers to the variety of life at all levels — genetic diversity within species,
species diversity within communities, and ecosystem diversity across landscapes. Biodiversity
is critical for ecosystem stability and resilience. Threats to biodiversity include habitat
destruction, invasive species, pollution, overexploitation, and climate change (collectively
known by the acronym HIPPO).
ECOLOGY_EOF

cat > "$TEST_DIR/sources/evolution_basics.txt" << 'EVOLUTION_EOF'
Fundamentals of Evolution

Evolution is the change in the inherited characteristics of biological populations over
successive generations. Charles Darwin and Alfred Russel Wallace independently developed the
theory of evolution by natural selection, which Darwin published in "On the Origin of Species"
in 1859.

Natural selection is the primary mechanism of evolution. It requires four conditions:
variation (individuals in a population differ in their traits), heritability (some of that
variation is genetic and can be passed to offspring), differential reproduction (individuals
with certain traits produce more offspring), and adaptation (over generations, favorable
traits become more common in the population). Natural selection acts on phenotypes, but
evolution occurs through changes in allele frequencies in the gene pool.

The Hardy-Weinberg principle describes the conditions under which allele frequencies in a
population remain constant across generations. The equation p² + 2pq + q² = 1 predicts
genotype frequencies, where p and q are the frequencies of two alleles. The five conditions
for Hardy-Weinberg equilibrium are: no mutation, random mating, no natural selection, large
population size (no genetic drift), and no gene flow. Any deviation from these conditions
causes evolution.

Genetic drift is the random change in allele frequencies due to chance events, especially
significant in small populations. Two special cases are the bottleneck effect (a drastic
reduction in population size due to a catastrophic event) and the founder effect (a small
group establishes a new population with different allele frequencies than the original).

Speciation is the formation of new species. Allopatric speciation occurs when populations
are geographically separated by a physical barrier (e.g., a mountain range or river),
preventing gene flow. Over time, the isolated populations diverge genetically and may become
reproductively isolated. Sympatric speciation occurs without geographic separation, often
through polyploidy (common in plants) or ecological specialization.

Evidence for evolution comes from multiple sources: the fossil record shows transitional
forms and patterns of change over time; comparative anatomy reveals homologous structures
(similar structures in different species due to common ancestry, e.g., the pentadactyl limb)
and analogous structures (similar function but different origin, e.g., bird and insect wings);
molecular biology shows that all life shares the same genetic code (DNA/RNA) and that more
closely related species have more similar DNA sequences; biogeography shows that species
distribution patterns reflect evolutionary history.

Phylogenetics is the study of evolutionary relationships among organisms. Phylogenetic trees
(cladograms) are diagrams that show the inferred evolutionary relationships based on shared
derived characteristics (synapomorphies). Molecular phylogenetics uses DNA or protein sequence
comparisons to reconstruct evolutionary history, often using the molecular clock hypothesis,
which assumes that genetic mutations accumulate at a roughly constant rate over time.

Coevolution occurs when two or more species reciprocally affect each other's evolution.
Examples include predator-prey arms races (e.g., rough-skinned newts producing increasingly
potent tetrodotoxin and garter snakes evolving resistance), plant-pollinator relationships
(e.g., flowers evolving shapes and colors that attract specific pollinators), and host-parasite
interactions. Coevolution can drive rapid diversification and specialization.
EVOLUTION_EOF

echo "Created test documents in $TEST_DIR/sources/"

# --- Step 5: Ingest -----------------------------------------------------------

echo ""
echo "=== Ingest ==="
STEP_START=$SECONDS
"$INSTRUCTOR" ingest --sources "$TEST_DIR/sources" --project "$TEST_DIR"
TIME_INGEST=$(( SECONDS - STEP_START ))
echo "--- Ingest completed in ${TIME_INGEST}s ---"

# --- Step 6: Model download + quantize ----------------------------------------

echo ""
echo "=== Model ==="
STEP_START=$SECONDS
"$INSTRUCTOR" model --name "$MODEL_NAME" --quantization "$QUANTIZATION" --project "$TEST_DIR"
TIME_MODEL=$(( SECONDS - STEP_START ))
echo "--- Model completed in ${TIME_MODEL}s ---"

# --- Step 7–8: DPO training data + fine-tuning (if LoRA enabled) -------------

TIME_DPO=0
if [ "$LORA" = "yes" ]; then

cat > "$TEST_DIR/dpo_data.jsonl" << 'DPO_EOF'
{"prompt":"What is the basic unit of life?","chosen":"The cell is the basic structural and functional unit of all living organisms. According to cell theory, all living organisms are composed of one or more cells, the cell is the basic unit of life, and all cells arise from pre-existing cells.","rejected":"Atoms are the basic unit of life because everything is made of atoms."}
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
{"prompt":"What is the endosymbiotic theory?","chosen":"The endosymbiotic theory proposes that mitochondria (and chloroplasts) were once free-living prokaryotes that were engulfed by ancestral eukaryotic cells. Evidence includes the fact that mitochondria contain their own DNA and have a double membrane structure.","rejected":"The endosymbiotic theory states that all cells evolved from viruses that merged together."}
{"prompt":"What is the cytoskeleton?","chosen":"The cytoskeleton is a network of protein filaments that provides structural support, aids in cell movement, and plays a role in intracellular transport. It consists of microfilaments (actin), intermediate filaments, and microtubules (tubulin).","rejected":"The cytoskeleton is the cell wall found in all cells that keeps them rigid and prevents any movement."}
{"prompt":"Describe the stages of mitosis.","chosen":"Mitosis consists of prophase (chromatin condenses into chromosomes, spindle forms), metaphase (chromosomes align at the equator), anaphase (sister chromatids separate to opposite poles), and telophase (nuclear envelopes reform), followed by cytokinesis.","rejected":"Mitosis has only two stages: the cell splits in half and then each half grows back to full size."}
{"prompt":"How does meiosis differ from mitosis?","chosen":"Meiosis produces four genetically unique haploid cells through two rounds of division (meiosis I and II), while mitosis produces two genetically identical diploid cells. Crossing over during prophase I of meiosis increases genetic diversity.","rejected":"Meiosis and mitosis are the same process with different names used for different organisms."}
{"prompt":"What is active transport?","chosen":"Active transport requires energy (ATP) to move substances against their concentration gradient. A key example is the sodium-potassium pump, which pumps 3 Na+ ions out and 2 K+ ions in per ATP molecule, maintaining the cell's resting membrane potential.","rejected":"Active transport is when molecules move freely through the membrane without any energy required."}
{"prompt":"What is epigenetics?","chosen":"Epigenetics refers to heritable changes in gene expression that do not involve changes to the DNA sequence. Key mechanisms include DNA methylation (adding methyl groups to cytosine, typically silencing genes) and histone modification (altering chromatin structure and gene accessibility).","rejected":"Epigenetics is the study of how genes mutate and change their DNA sequence over a person's lifetime."}
{"prompt":"What is genetic linkage?","chosen":"Genetic linkage occurs when genes are located close together on the same chromosome and tend to be inherited together. The recombination frequency between linked genes depends on their physical distance, measured in centimorgans. Thomas Hunt Morgan demonstrated this using Drosophila.","rejected":"Genetic linkage means all genes on the same chromosome are always inherited as an identical block with no variation."}
{"prompt":"How does PCR work?","chosen":"PCR amplifies specific DNA sequences through repeated cycles of denaturation (separating strands at ~95°C), annealing (primers binding at ~55-65°C), and extension (DNA polymerase synthesizing new strands at ~72°C). It can produce millions of copies from a tiny starting amount.","rejected":"PCR is a chemical reaction that creates entirely new DNA sequences that don't exist in the original sample."}
{"prompt":"What is a population in ecology?","chosen":"A population is a group of individuals of the same species living in a particular area at a given time. Population growth can follow exponential growth (unlimited resources, dN/dt = rN) or logistic growth (limited resources, dN/dt = rN(1-N/K), where K is carrying capacity).","rejected":"A population is any random collection of different species found anywhere on Earth at any time."}
{"prompt":"What is the competitive exclusion principle?","chosen":"The competitive exclusion principle (Gause's principle) states that two species competing for the same limited resource cannot coexist indefinitely — one will outcompete and exclude the other. This drives niche differentiation among coexisting species.","rejected":"The competitive exclusion principle states that all species in an ecosystem must cooperate equally to survive."}
{"prompt":"How does energy flow through ecosystems?","chosen":"Energy flows through ecosystems in one direction: from the sun to producers (autotrophs) to consumers (heterotrophs) to decomposers. Only about 10% of energy is transferred from one trophic level to the next, with the rest lost as heat through cellular respiration.","rejected":"Energy flows in a circle through ecosystems, with each organism passing 100% of its energy to the next level."}
{"prompt":"What is the nitrogen cycle?","chosen":"The nitrogen cycle includes nitrogen fixation (N2 to NH3 by bacteria), nitrification (NH3 to NO3-), assimilation by plants, and denitrification (NO3- back to N2). These processes move nitrogen through biotic and abiotic components of ecosystems.","rejected":"The nitrogen cycle is when plants absorb nitrogen gas directly from the atmosphere through their leaves."}
{"prompt":"What is ecological succession?","chosen":"Ecological succession is the process of change in community species structure over time. Primary succession occurs on bare surfaces (e.g., after volcanic eruption) starting with pioneer species like lichens. Secondary succession occurs where soil remains (e.g., after fire). Both progress toward a climax community.","rejected":"Ecological succession is when all species in an area go extinct simultaneously and are replaced by completely new species."}
{"prompt":"What is natural selection?","chosen":"Natural selection requires variation, heritability, differential reproduction, and adaptation. Individuals with favorable traits produce more offspring, and over generations those traits become more common. Natural selection acts on phenotypes, but evolution occurs through changes in allele frequencies.","rejected":"Natural selection is when organisms consciously choose to develop better traits to survive in their environment."}
{"prompt":"What is the Hardy-Weinberg principle?","chosen":"The Hardy-Weinberg principle describes conditions under which allele frequencies remain constant: no mutation, random mating, no selection, large population, and no gene flow. The equation p² + 2pq + q² = 1 predicts genotype frequencies. Any deviation causes evolution.","rejected":"The Hardy-Weinberg principle states that all populations always maintain the same allele frequencies regardless of conditions."}
{"prompt":"What is genetic drift?","chosen":"Genetic drift is the random change in allele frequencies due to chance events, especially significant in small populations. The bottleneck effect occurs when population size drastically decreases, and the founder effect occurs when a small group starts a new population.","rejected":"Genetic drift is when organisms intentionally migrate to new locations to find better mates with different genes."}
{"prompt":"How does allopatric speciation occur?","chosen":"Allopatric speciation occurs when populations are geographically separated by a physical barrier (e.g., mountain range or river), preventing gene flow. Over time, the isolated populations diverge genetically through mutation, drift, and selection, becoming reproductively isolated.","rejected":"Allopatric speciation happens when all individuals in a species simultaneously decide to split into two groups."}
{"prompt":"What evidence supports evolution?","chosen":"Evidence for evolution includes the fossil record (transitional forms), comparative anatomy (homologous structures like the pentadactyl limb), molecular biology (shared genetic code, DNA sequence similarity), and biogeography (species distribution reflecting evolutionary history).","rejected":"There is no scientific evidence for evolution; it is purely a philosophical concept with no observable support."}
{"prompt":"What is coevolution?","chosen":"Coevolution occurs when two or more species reciprocally affect each other's evolution. Examples include predator-prey arms races (newt toxin vs. snake resistance), plant-pollinator relationships, and host-parasite interactions. It drives rapid diversification and specialization.","rejected":"Coevolution means all species in an ecosystem evolve at exactly the same rate in the same direction."}
{"prompt":"What threats affect biodiversity?","chosen":"Major threats to biodiversity include habitat destruction, invasive species, pollution, overexploitation, and climate change (known by the acronym HIPPO). Biodiversity at genetic, species, and ecosystem levels is critical for ecosystem stability and resilience.","rejected":"Biodiversity is never threatened because nature always maintains a perfect balance regardless of human activity."}
{"prompt":"What is phylogenetics?","chosen":"Phylogenetics is the study of evolutionary relationships among organisms. Phylogenetic trees (cladograms) show inferred relationships based on shared derived characteristics (synapomorphies). Molecular phylogenetics uses DNA or protein sequence comparisons, often applying the molecular clock hypothesis.","rejected":"Phylogenetics is the classification of organisms by size and color, and has nothing to do with evolutionary history."}
DPO_EOF

echo "Created DPO training data (40 examples)"

echo ""
echo "=== DPO Train ==="
STEP_START=$SECONDS
"$INSTRUCTOR" finetune --method dpo --data "$TEST_DIR/dpo_data.jsonl" --project "$TEST_DIR"
TIME_DPO=$(( SECONDS - STEP_START ))
echo "--- DPO Train completed in ${TIME_DPO}s ---"

else
    echo ""
    echo "=== Skipping LoRA fine-tuning (--lora no) ==="
fi

# --- Step 9: Config -----------------------------------------------------------

echo ""
echo "=== Config ==="
STEP_START=$SECONDS
"$INSTRUCTOR" config --project "$TEST_DIR" --system-prompt "You are a helpful biology tutor. Answer questions accurately using course materials."
TIME_CONFIG=$(( SECONDS - STEP_START ))
echo "--- Config completed in ${TIME_CONFIG}s ---"

# --- Step 10: Bundle ----------------------------------------------------------

echo ""
echo "=== Bundle ==="
STEP_START=$SECONDS
"$INSTRUCTOR" bundle --project "$TEST_DIR" --output "$PROJECT_ROOT/test-project.sheplet"
TIME_BUNDLE=$(( SECONDS - STEP_START ))
echo "--- Bundle completed in ${TIME_BUNDLE}s ---"

# --- Step 11: Summary ---------------------------------------------------------

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))

echo ""
echo "=== Output Sizes ($MODEL_NAME) ==="
if [ -f "$TEST_DIR/model/model.gguf" ]; then
    echo "  model.gguf:          $(du -h "$TEST_DIR/model/model.gguf" | cut -f1)"
fi
# Show SafeTensors files (present for both quantized and full-precision models)
for f in "$TEST_DIR/model/"*.safetensors; do
    [ -f "$f" ] && echo "  $(basename "$f"): $(du -h "$f" | cut -f1)"
done
if [ -f "$TEST_DIR/model/adapter.safetensors" ]; then
    echo "  adapter.safetensors: $(du -h "$TEST_DIR/model/adapter.safetensors" | cut -f1)"
fi
if [ -d "$TEST_DIR/database" ]; then
    echo "  database/:           $(du -sh "$TEST_DIR/database" | cut -f1)"
fi
if [ -f "$PROJECT_ROOT/test-project.sheplet" ]; then
    echo "  test-project.sheplet: $(du -h "$PROJECT_ROOT/test-project.sheplet" | cut -f1)"
fi

echo ""
echo "=== Timing Summary ==="
printf "  %-14s %ss\n" "Build:" "$TIME_BUILD"
printf "  %-14s %ss\n" "Init:" "$TIME_INIT"
printf "  %-14s %ss\n" "Ingest:" "$TIME_INGEST"
printf "  %-14s %ss\n" "Model:" "$TIME_MODEL"
if [ "$LORA" = "yes" ]; then
printf "  %-14s %ss\n" "DPO Train:" "$TIME_DPO"
fi
printf "  %-14s %ss\n" "Config:" "$TIME_CONFIG"
printf "  %-14s %ss\n" "Bundle:" "$TIME_BUNDLE"
printf "  %-14s %ss\n" "Total:" "$TOTAL_ELAPSED"

echo ""
echo "Done!"
