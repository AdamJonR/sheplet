#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Sheplet QA Test Script
# Exercises the full instructor → student pipeline across all supported models,
# each with 2 configurations:
#   1. SafeTensors + LoRA
#   2. SafeTensors + No LoRA
# Use --model to test a single model instead.
# =============================================================================

# --- Section 1: Environment & Constants --------------------------------------

export PATH="$HOME/.cargo/bin:$HOME/miniconda3/envs/ml-env/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QA_DIR="$PROJECT_ROOT/test-qa"
INSTRUCTOR="$PROJECT_ROOT/target/release/sheplet-instructor"
STUDENT="$PROJECT_ROOT/target/release/sheplet-student"

usage() {
    echo "Usage: $0 [--model MODEL]"
    echo ""
    echo "Models: llama1b, llama3b, qwen0.5b, qwen1.5b, qwen3b,"
    echo "        gemma2b, gemma2-2b, mistral7b, phi3"
    echo ""
    echo "Default: test all models (with and without LoRA)"
    exit 1
}

# Defaults
MODEL="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# Map shortcut to full model name
resolve_model_name() {
    local shortcut="$1"
    case "$shortcut" in
        llama1b)    echo "llama-3.2-1b" ;;
        llama3b)    echo "llama-3.2-3b" ;;
        qwen0.5b)   echo "qwen2.5-0.5b" ;;
        qwen1.5b)   echo "qwen2.5-1.5b" ;;
        qwen3b)     echo "qwen2.5-3b" ;;
        gemma2b)    echo "gemma-2b" ;;
        gemma2-2b)  echo "gemma-2-2b" ;;
        mistral7b)  echo "mistral-7b" ;;
        phi3)       echo "phi-3-mini-4k-instruct" ;;
        *)          echo ""; return 1 ;;
    esac
}

# Map shortcut to downloaded-models directory name
resolve_model_dir() {
    local shortcut="$1"
    case "$shortcut" in
        llama1b)    echo "meta-llama--Llama-3.2-1B-Instruct" ;;
        llama3b)    echo "meta-llama--Llama-3.2-3B-Instruct" ;;
        qwen0.5b)   echo "Qwen--Qwen2.5-0.5B-Instruct" ;;
        qwen1.5b)   echo "Qwen--Qwen2.5-1.5B-Instruct" ;;
        qwen3b)     echo "Qwen--Qwen2.5-3B-Instruct" ;;
        gemma2b)    echo "google--gemma-2b-it" ;;
        gemma2-2b)  echo "google--gemma-2-2b-it" ;;
        mistral7b)  echo "mistralai--Mistral-7B-Instruct-v0.3" ;;
        phi3)       echo "microsoft--Phi-3-mini-4k-instruct" ;;
        *)          echo "" ;;
    esac
}

# Build the list of models to test
ALL_SHORTCUTS=(llama1b llama3b qwen0.5b qwen1.5b qwen3b gemma2b gemma2-2b mistral7b phi3)

if [ "$MODEL" != "all" ]; then
    # Validate the model shortcut
    if ! resolve_model_name "$MODEL" >/dev/null 2>&1; then
        echo "Unknown model: $MODEL"
        usage
    fi
    ALL_SHORTCUTS=("$MODEL")
fi

QUESTION="How many chromosomes does a human have?"
MAX_TOKENS=128
PORT=8421
CHAT_TIMEOUT=120
SERVER_WAIT=30

# Test configurations: "label|do_lora"
CONFIGS=(
    "SafeTensors + LoRA|yes"
    "SafeTensors + No LoRA|no"
)

TOTAL_START=$SECONDS

# --- Detect GPU features ----------------------------------------------------
source "$SCRIPT_DIR/detect_gpu.sh"

# --- Section 2: Build both binaries once ------------------------------------

echo ""
echo "=== Building instructor + student ==="
STEP_START=$SECONDS
cargo build --release -p sheplet-instructor -p sheplet-student $CARGO_FEATURES \
    --manifest-path "$PROJECT_ROOT/Cargo.toml"
TIME_BUILD=$(( SECONDS - STEP_START ))
echo "--- Build completed in ${TIME_BUILD}s ---"

# --- Section 3: Helper functions ---------------------------------------------

create_test_documents() {
    local project_dir="$1"
    mkdir -p "$project_dir/sources"

    cat > "$project_dir/sources/cell_biology.txt" << 'CELL_EOF'
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
CELL_EOF

    cat > "$project_dir/sources/genetics_basics.txt" << 'GENETICS_EOF'
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
GENETICS_EOF
}

create_dpo_data() {
    local project_dir="$1"

    cat > "$project_dir/dpo_data.jsonl" << 'DPO_EOF'
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
DPO_EOF
}

run_instructor_pipeline() {
    local project_dir="$1"
    local do_lora="$2"
    local model_name="$3"
    local bundle_path="$project_dir/bundle.sheplet"

    # Init
    echo "  [init]"
    local step_start=$SECONDS
    "$INSTRUCTOR" init --course "Biology QA Test" --output "$project_dir"
    local time_init=$(( SECONDS - step_start ))
    echo "    init: ${time_init}s"

    # Create documents and DPO data
    create_test_documents "$project_dir"
    if [ "$do_lora" = "yes" ]; then
        create_dpo_data "$project_dir"
    fi

    # Ingest
    echo "  [ingest]"
    step_start=$SECONDS
    "$INSTRUCTOR" ingest --sources "$project_dir/sources" --project "$project_dir"
    local time_ingest=$(( SECONDS - step_start ))
    echo "    ingest: ${time_ingest}s"

    # Model
    echo "  [model]"
    step_start=$SECONDS
    "$INSTRUCTOR" model --name "$model_name" --project "$project_dir"
    local time_model=$(( SECONDS - step_start ))
    echo "    model: ${time_model}s"

    # Finetune (conditional)
    local time_finetune=0
    if [ "$do_lora" = "yes" ]; then
        echo "  [finetune] DPO, 1 epoch"
        step_start=$SECONDS
        "$INSTRUCTOR" finetune --method dpo --data "$project_dir/dpo_data.jsonl" --project "$project_dir" --epochs 1
        time_finetune=$(( SECONDS - step_start ))
        echo "    finetune: ${time_finetune}s"
    else
        echo "  [finetune] skipped (no LoRA)"
    fi

    # Config
    echo "  [config]"
    step_start=$SECONDS
    "$INSTRUCTOR" config --project "$project_dir" \
        --system-prompt "You are a helpful biology tutor. Answer questions accurately using course materials."
    local time_config=$(( SECONDS - step_start ))
    echo "    config: ${time_config}s"

    # Bundle
    echo "  [bundle]"
    step_start=$SECONDS
    local bundle_output
    bundle_output=$("$INSTRUCTOR" bundle --project "$project_dir" --output "$bundle_path" 2>&1)
    echo "$bundle_output"
    local time_bundle=$(( SECONDS - step_start ))
    echo "    bundle: ${time_bundle}s"

    # Extract fingerprint
    RESULT_FINGERPRINT=$(echo "$bundle_output" | grep 'Fingerprint:' | head -1 | awk '{print $NF}')
    RESULT_BUNDLE_PATH="$bundle_path"
    RESULT_PIPELINE_TIME=$(( time_init + time_ingest + time_model + time_finetune + time_config + time_bundle ))
}

run_student_query() {
    local bundle_path="$1"
    local fingerprint="$2"
    local no_adapter_flag="$3"
    local student_dir
    student_dir=$(mktemp -d)
    local stderr_log="$student_dir/stderr.log"

    # Build student command
    local student_cmd=("$STUDENT" --dir "$student_dir" --port "$PORT")
    if [ "$no_adapter_flag" = "yes" ]; then
        student_cmd+=(--no-adapter)
    fi

    # Start student server in background
    "${student_cmd[@]}" 2>"$stderr_log" &
    local student_pid=$!

    # Ensure we clean up on exit from this function
    cleanup_student() {
        if kill -0 "$student_pid" 2>/dev/null; then
            kill "$student_pid" 2>/dev/null || true
            wait "$student_pid" 2>/dev/null || true
        fi
    }
    trap cleanup_student RETURN

    # Wait for server readiness
    echo "  [student] waiting for server (pid=$student_pid)..."
    local waited=0
    while ! curl -s "http://127.0.0.1:$PORT/api/courses" >/dev/null 2>&1; do
        if ! kill -0 "$student_pid" 2>/dev/null; then
            echo "  [student] ERROR: server exited prematurely"
            echo "  stderr:"
            cat "$stderr_log" 2>/dev/null || true
            RESULT_RESPONSE=""
            RESULT_BLOCKED="true"
            RESULT_LOAD_TIME=0
            RESULT_CHAT_TIME=0
            RESULT_STDERR=""
            rm -rf "$student_dir"
            return 1
        fi
        sleep 1
        waited=$(( waited + 1 ))
        if [ "$waited" -ge "$SERVER_WAIT" ]; then
            echo "  [student] ERROR: server did not start within ${SERVER_WAIT}s"
            RESULT_RESPONSE=""
            RESULT_BLOCKED="true"
            RESULT_LOAD_TIME=0
            RESULT_CHAT_TIME=0
            RESULT_STDERR=""
            rm -rf "$student_dir"
            return 1
        fi
    done
    echo "  [student] server ready (${waited}s)"

    # Load bundle
    echo "  [student] loading bundle..."
    local step_start=$SECONDS
    local load_response
    load_response=$(curl -s -X POST "http://127.0.0.1:$PORT/api/bundles/load" \
        -H "Content-Type: application/json" \
        -d "{\"path\": \"$bundle_path\", \"trusted_fingerprint\": \"$fingerprint\"}")
    RESULT_LOAD_TIME=$(( SECONDS - step_start ))
    echo "  [student] bundle loaded in ${RESULT_LOAD_TIME}s"
    echo "    load response: $load_response"

    # Check if load failed
    if echo "$load_response" | grep -qi "error"; then
        echo "  [student] ERROR: bundle load failed"
        RESULT_RESPONSE="LOAD_FAILED: $load_response"
        RESULT_BLOCKED="true"
        RESULT_CHAT_TIME=0
        RESULT_STDERR=$(cat "$stderr_log" 2>/dev/null || true)
        rm -rf "$student_dir"
        return 1
    fi

    # Send question
    echo "  [student] asking: \"$QUESTION\""
    step_start=$SECONDS
    local chat_response
    chat_response=$(curl -s -X POST "http://127.0.0.1:$PORT/api/chat/sync" \
        -H "Content-Type: application/json" \
        --max-time "$CHAT_TIMEOUT" \
        -d "{\"message\": \"$QUESTION\", \"max_tokens\": $MAX_TOKENS}")
    RESULT_CHAT_TIME=$(( SECONDS - step_start ))
    echo "  [student] response received in ${RESULT_CHAT_TIME}s"

    # Parse response
    RESULT_RESPONSE=$(echo "$chat_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('response', ''))
except:
    print('PARSE_ERROR')
" 2>/dev/null || echo "PARSE_ERROR")

    RESULT_BLOCKED=$(echo "$chat_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(str(data.get('blocked', False)).lower())
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

    # Collect stderr
    RESULT_STDERR=$(cat "$stderr_log" 2>/dev/null || true)

    # Cleanup student dir
    rm -rf "$student_dir"
}

# --- Section 4: Main loop ---------------------------------------------------

# Cleanup previous run
if [ -d "$QA_DIR" ]; then
    rm -rf "$QA_DIR"
    echo "Cleaned previous QA test run"
fi
mkdir -p "$QA_DIR"

# Check model availability and build run list
declare -a RUN_MODELS
declare -a SKIP_MODELS

for shortcut in "${ALL_SHORTCUTS[@]}"; do
    model_dir_name=$(resolve_model_dir "$shortcut")
    model_path="$PROJECT_ROOT/downloaded-models/$model_dir_name"
    if [ -d "$model_path" ]; then
        RUN_MODELS+=("$shortcut")
    else
        SKIP_MODELS+=("$shortcut")
        echo "SKIP: $shortcut — not found at downloaded-models/$model_dir_name"
    fi
done

if [ ${#RUN_MODELS[@]} -eq 0 ]; then
    echo ""
    echo "No models available to test. Download models to downloaded-models/ first."
    exit 1
fi

echo ""
echo "Models to test: ${RUN_MODELS[*]}"
if [ ${#SKIP_MODELS[@]} -gt 0 ]; then
    echo "Models skipped (not downloaded): ${SKIP_MODELS[*]}"
fi

# Arrays to store results
declare -a RESULT_LABELS
declare -a RESULT_STATUSES
declare -a RESULT_RESPONSES
declare -a RESULT_LOAD_TIMES
declare -a RESULT_CHAT_TIMES
declare -a RESULT_PIPELINE_TIMES
declare -a RESULT_STDERRS

# Add skip entries for unavailable models
for shortcut in "${SKIP_MODELS[@]}"; do
    for config in "${CONFIGS[@]}"; do
        IFS='|' read -r label do_lora <<< "$config"
        RESULT_LABELS+=("$shortcut | $label")
        RESULT_STATUSES+=("SKIP")
        RESULT_RESPONSES+=("Model not downloaded")
        RESULT_LOAD_TIMES+=(0)
        RESULT_CHAT_TIMES+=(0)
        RESULT_PIPELINE_TIMES+=(0)
        RESULT_STDERRS+=("")
    done
done

# Run tests for each available model
for shortcut in "${RUN_MODELS[@]}"; do
    MODEL_NAME=$(resolve_model_name "$shortcut")

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "  Model: $shortcut ($MODEL_NAME)"
    echo "╚══════════════════════════════════════════════════════════════════╝"

    config_num=0
    for config in "${CONFIGS[@]}"; do
        config_num=$(( config_num + 1 ))

        IFS='|' read -r label do_lora <<< "$config"
        full_label="$shortcut | $label"

        echo ""
        echo "================================================================"
        echo "  Config $config_num/${#CONFIGS[@]}: $full_label"
        echo "  lora=$do_lora"
        echo "================================================================"

        project_dir="$QA_DIR/$shortcut/config-$config_num"

        # Run instructor pipeline
        echo ""
        echo "--- Instructor Pipeline ---"
        if run_instructor_pipeline "$project_dir" "$do_lora" "$MODEL_NAME"; then
            echo "  Pipeline completed in ${RESULT_PIPELINE_TIME}s"
        else
            echo "  Pipeline FAILED"
            RESULT_LABELS+=("$full_label")
            RESULT_STATUSES+=("FAIL")
            RESULT_RESPONSES+=("Pipeline failed")
            RESULT_LOAD_TIMES+=(0)
            RESULT_CHAT_TIMES+=(0)
            RESULT_PIPELINE_TIMES+=(0)
            RESULT_STDERRS+=("")
            continue
        fi

        # Determine no-adapter flag
        no_adapter="no"
        if [ "$do_lora" = "no" ]; then
            no_adapter="yes"
        fi

        # Run student query
        echo ""
        echo "--- Student Query ---"
        if run_student_query "$RESULT_BUNDLE_PATH" "$RESULT_FINGERPRINT" "$no_adapter"; then
            query_status="ok"
        else
            query_status="error"
        fi

        # Determine pass/fail
        local_status="FAIL"
        if [ "$query_status" = "ok" ] && [ -n "$RESULT_RESPONSE" ] && [ "$RESULT_RESPONSE" != "PARSE_ERROR" ] && [ "$RESULT_BLOCKED" != "true" ]; then
            local_status="PASS"
        fi

        RESULT_LABELS+=("$full_label")
        RESULT_STATUSES+=("$local_status")
        RESULT_RESPONSES+=("$RESULT_RESPONSE")
        RESULT_LOAD_TIMES+=("$RESULT_LOAD_TIME")
        RESULT_CHAT_TIMES+=("$RESULT_CHAT_TIME")
        RESULT_PIPELINE_TIMES+=("$RESULT_PIPELINE_TIME")
        RESULT_STDERRS+=("$RESULT_STDERR")
    done
done

# --- Section 5: Summary table -----------------------------------------------

TOTAL_ELAPSED=$(( SECONDS - TOTAL_START ))

echo ""
echo ""
echo "========================================================================"
echo "  QA Test Results"
echo "  Question: \"$QUESTION\""
echo "  Build time: ${TIME_BUILD}s"
echo "========================================================================"

for i in "${!RESULT_LABELS[@]}"; do
    echo ""
    echo "--- ${RESULT_LABELS[$i]} [${RESULT_STATUSES[$i]}] ---"

    if [ "${RESULT_STATUSES[$i]}" = "SKIP" ]; then
        echo "  (model not downloaded)"
        continue
    fi

    echo "  Pipeline: ${RESULT_PIPELINE_TIMES[$i]}s | Bundle Load: ${RESULT_LOAD_TIMES[$i]}s | Chat: ${RESULT_CHAT_TIMES[$i]}s"
    echo "  Response:"

    # Word-wrap the response at ~76 chars with indentation
    if [ -n "${RESULT_RESPONSES[$i]}" ]; then
        echo "${RESULT_RESPONSES[$i]}" | fold -s -w 72 | sed 's/^/    /'
    else
        echo "    (empty)"
    fi

    # Show relevant stderr excerpts (filter out noise, keep diagnostics)
    if [ -n "${RESULT_STDERRS[$i]}" ]; then
        # Extract lines with useful diagnostics
        local_stderr=$(echo "${RESULT_STDERRS[$i]}" | grep -iE "(eos|error|warn|model|lora|adapter|loaded|config|logit|token|top-5|WARNING|debug|dtype|device)" || true)
        if [ -n "$local_stderr" ]; then
            echo "  Diagnostics:"
            echo "$local_stderr" | head -20 | sed 's/^/    /'
        fi
    fi
done

# Summary line
pass_count=0
fail_count=0
skip_count=0
for status in "${RESULT_STATUSES[@]}"; do
    case "$status" in
        PASS) pass_count=$(( pass_count + 1 )) ;;
        FAIL) fail_count=$(( fail_count + 1 )) ;;
        SKIP) skip_count=$(( skip_count + 1 )) ;;
    esac
done

echo ""
echo "========================================================================"
echo "  Total: $pass_count PASS / $fail_count FAIL / $skip_count SKIP | Elapsed: ${TOTAL_ELAPSED}s"
echo "========================================================================"
echo ""

if [ "$fail_count" -gt 0 ]; then
    echo "Some configs failed. Check test-qa/ for artifacts."
    exit 1
fi

if [ "$pass_count" -gt 0 ]; then
    echo "All tested configs passed!"
else
    echo "No configs were tested (all models skipped)."
fi
