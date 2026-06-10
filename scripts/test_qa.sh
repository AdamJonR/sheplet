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

QUESTION="On how many hills was the city of Rome built?"
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

    cat > "$project_dir/sources/roman_founding.txt" << 'FOUNDING_EOF'
The Founding and Geography of Rome

Rome was traditionally founded in 753 BC, a date calculated by the Roman scholar Marcus
Terentius Varro. According to legend, the city was established by Romulus, who became its
first king and gave the city its name. While the founding legend is mythological, archaeology
confirms that permanent settlements existed on the site from at least the 8th century BC.

The city of Rome was built on seven hills overlooking the Tiber River in west-central Italy.
The seven hills are the Palatine, Capitoline, Aventine, Caelian, Esquiline, Viminal, and
Quirinal. The Palatine Hill, where the earliest settlement is thought to have begun, later
became the location of imperial residences. The Capitoline Hill held the most important
temples, including the Temple of Jupiter Optimus Maximus, and served as the religious and
political heart of the city.

The Tiber River was central to Rome's growth. It provided fresh water, a route for trade,
and a natural defensive barrier. Rome's location about 25 kilometers inland gave it access
to the sea through the river while keeping it protected from coastal raids. A river island,
the Tiber Island, provided the easiest crossing point and became an early focus of settlement.

Roman history is conventionally divided into three periods. The Roman Kingdom (753–509 BC)
was ruled by a series of kings, traditionally numbered as seven. The Roman Republic (509–27
BC) replaced the monarchy with elected officials and a Senate. The Roman Empire (27 BC
onward) began when Augustus became the first emperor, concentrating authority in a single
ruler while preserving many republican institutions in name.

The boundary of the original city was marked by a sacred line called the pomerium. Within
the pomerium, certain activities were restricted by religious custom. As Rome grew, the
pomerium was extended several times, reflecting the expansion of the city beyond its
original walls.

The Italian peninsula provided Rome with fertile farmland, especially the plains of Latium
where Rome sat and the rich region of Campania to the south. The Apennine Mountains form the
peninsula's spine, and the surrounding Mediterranean Sea connected Italy to trade networks
across three continents. These geographic advantages supported a growing population and a
stable agricultural base.

Latium was home to the Latins, the people from whom Rome emerged, and their language, Latin,
became the language of Rome. Neighboring peoples included the Etruscans to the north, whose
culture strongly influenced early Rome, and various Italic peoples throughout the peninsula.
The Etruscans contributed engineering knowledge, religious practices, and even some of Rome's
early symbols of authority.

The Romans measured their calendar from the supposed founding of the city, using the phrase
"ab urbe condita," meaning "from the founding of the city." This dating system placed year
one at 753 BC, so events were sometimes recorded as occurring a given number of years after
the city's founding.
FOUNDING_EOF

    cat > "$project_dir/sources/roman_government.txt" << 'GOVERNMENT_EOF'
Government of the Roman Republic

The Roman Republic was established in 509 BC, when the Romans replaced their monarchy with a
system of elected officials and representative bodies. The word "republic" comes from the
Latin "res publica," meaning "public affair" or "the public thing," reflecting the idea that
the state belonged to its citizens rather than to a king.

The Senate was the most prestigious institution of the Republic. It was a council of
experienced statesmen, originally drawn from leading families, who advised the magistrates
and guided policy on finance, foreign relations, and religion. Although the Senate's formal
power was advisory, its authority and continuity made it the dominant force in Roman politics.
Its decrees were called senatus consulta.

Executive authority was held by two consuls, elected annually. Having two consuls who could
each veto the other prevented any single person from holding too much power, and the one-year
term ensured regular turnover. The consuls led the government, proposed laws, and presided
over the most important public business. After their year in office, former consuls often
continued to serve the state in other roles.

Roman officials advanced through a sequence of offices known as the cursus honorum, or
"course of honors." The typical sequence began with the quaestor, who managed financial
affairs, followed by the aedile, responsible for public buildings, games, and the grain
supply. Next came the praetor, who administered justice, and finally the consul, the highest
regular office. Each step required a minimum age and prior experience, ensuring that leaders
were tested before reaching the top.

The tribunes of the plebs were officials created to protect the interests of the common
citizens, known as plebeians. Tribunes could veto actions of magistrates and the Senate that
they judged harmful to the people, and their persons were legally protected. This office gave
ordinary citizens a formal voice in government and served as a check on the power of the
aristocracy.

Roman society was historically divided between patricians, the established noble families,
and plebeians, the broader body of common citizens. Over the early Republic, plebeians
gradually gained political rights through a long process of reform, eventually winning access
to the highest offices and a share in lawmaking.

The Twelve Tables, written around 451 BC, were Rome's first written code of law. Before this,
legal customs were known mainly to the priests and aristocrats, which left ordinary citizens
at a disadvantage. By inscribing the laws on twelve bronze tablets displayed in public, Rome
made the rules known to everyone. The Twelve Tables became the foundation of Roman law, a
tradition that influenced legal systems for centuries afterward.

The Romans expressed the partnership between the people and their governing council with the
abbreviation SPQR, standing for "Senatus Populusque Romanus," meaning "the Senate and the
People of Rome." This phrase appeared on official documents, public monuments, and the
standards carried by Roman institutions, symbolizing the shared authority of the state.

The Republic's system of balanced offices, term limits, and overlapping powers was admired by
later thinkers as an early example of separated and checked authority. The Greek historian
Polybius, who lived in Rome during the 2nd century BC, described the Roman constitution as a
mixture of monarchy, aristocracy, and democracy, each element restraining the others.
GOVERNMENT_EOF
}

create_dpo_data() {
    local project_dir="$1"

    cat > "$project_dir/dpo_data.jsonl" << 'DPO_EOF'
{"prompt":"When was Rome traditionally founded?","chosen":"Rome was traditionally founded in 753 BC, a date calculated by the scholar Varro. According to legend the city was established by Romulus, who became its first king, though archaeology shows settlements existed on the site from at least the 8th century BC.","rejected":"Rome was founded in 1200 AD during the Middle Ages by a group of traveling merchants."}
{"prompt":"On how many hills was the city of Rome built?","chosen":"Rome was built on seven hills overlooking the Tiber River: the Palatine, Capitoline, Aventine, Caelian, Esquiline, Viminal, and Quirinal. The Palatine is where the earliest settlement is thought to have begun.","rejected":"Rome was built on a single large mountain with no surrounding hills."}
{"prompt":"What are the three periods of Roman history?","chosen":"Roman history is divided into three periods: the Roman Kingdom (753–509 BC), ruled by kings; the Roman Republic (509–27 BC), governed by elected officials and a Senate; and the Roman Empire (27 BC onward), which began under Augustus.","rejected":"Roman history had only one period, the Empire, which lasted from the beginning to the end without any changes in government."}
{"prompt":"Why was the Tiber River important to Rome?","chosen":"The Tiber River provided fresh water, a route for trade, and a natural defensive barrier. Rome's location inland gave it access to the sea through the river while keeping it protected from coastal raids, and the Tiber Island offered the easiest crossing point.","rejected":"The Tiber River was unimportant to Rome because the city relied entirely on rainfall and had no use for the river."}
{"prompt":"When was the Roman Republic established?","chosen":"The Roman Republic was established in 509 BC, when the Romans replaced their monarchy with elected officials and representative bodies. The word republic comes from the Latin res publica, meaning the public affair.","rejected":"The Roman Republic was established in 753 BC at the same moment the city was founded, with no period of monarchy beforehand."}
{"prompt":"What was the Roman Senate?","chosen":"The Senate was the most prestigious institution of the Republic, a council of experienced statesmen who advised the magistrates and guided policy on finance, foreign relations, and religion. Its authority and continuity made it the dominant force in Roman politics.","rejected":"The Senate was a single elected king who ruled Rome with absolute power and answered to no one."}
{"prompt":"How many consuls led the Roman Republic, and why?","chosen":"The Republic was led by two consuls, elected annually. Having two consuls who could each veto the other prevented any single person from holding too much power, and the one-year term ensured regular turnover.","rejected":"The Republic was led by one consul who served for life and could not be removed or overruled by anyone."}
{"prompt":"What was the cursus honorum?","chosen":"The cursus honorum, or course of honors, was the sequence of offices a Roman official advanced through: quaestor (finances), aedile (public buildings and games), praetor (justice), and finally consul, the highest regular office. Each step required a minimum age and prior experience.","rejected":"The cursus honorum was a chariot race held each year to decide who would become the next king of Rome."}
{"prompt":"What were the Twelve Tables?","chosen":"The Twelve Tables, written around 451 BC, were Rome's first written code of law. By inscribing the laws on twelve bronze tablets displayed in public, Rome made the rules known to everyone, and they became the foundation of Roman law.","rejected":"The Twelve Tables were twelve dining tables in the Senate house where senators ate their meals during meetings."}
{"prompt":"What does SPQR stand for?","chosen":"SPQR stands for Senatus Populusque Romanus, meaning the Senate and the People of Rome. It appeared on official documents, monuments, and standards, symbolizing the shared authority of the governing council and the citizens.","rejected":"SPQR stands for the Strong Powerful Quick Romans, a motto used only by the Roman army."}
{"prompt":"What were the tribunes of the plebs?","chosen":"The tribunes of the plebs were officials created to protect the common citizens, the plebeians. They could veto actions of magistrates and the Senate judged harmful to the people, and their persons were legally protected, giving ordinary citizens a formal voice.","rejected":"The tribunes of the plebs were wealthy nobles who collected taxes from the poor and had no role in protecting citizens."}
{"prompt":"What was the first major Roman road?","chosen":"The first great Roman road was the Appian Way (Via Appia), begun in 312 BC, connecting Rome to the south of Italy. Roman roads were built in layers of stone, gravel, and fitted paving, and were cambered so rainwater drained to the sides.","rejected":"The first Roman road was the Silk Road, which the Romans built to connect directly to China."}
{"prompt":"How did Roman aqueducts move water?","chosen":"Roman aqueducts worked entirely by gravity. The channel sloped gently downhill for its whole length, often many kilometers, so water flowed steadily without any pump. Where valleys interrupted the route, engineers carried the channel across on rows of stone arches.","rejected":"Roman aqueducts used large mechanical pumps powered by steam engines to push water uphill into the cities."}
{"prompt":"What was Roman concrete made of?","chosen":"Roman concrete, called opus caementicium, was made by mixing lime, water, and a volcanic ash called pozzolana with pieces of stone or brick. It set into a hard, durable mass and could even harden underwater, making it ideal for harbors and large vaulted structures.","rejected":"Roman concrete was made from a mixture of mud and straw that dried in the sun and dissolved when it rained."}
{"prompt":"How does a Roman arch carry weight?","chosen":"An arch carries weight by directing the load outward and downward along a curve of wedge-shaped stones called voussoirs. The keystone at the top locks the others in place. Because the arch is strong in compression, it let the Romans bridge wide openings and build to great heights.","rejected":"A Roman arch carries weight because the stones are glued together with a strong adhesive, and it would collapse without the glue."}
{"prompt":"What is the Pantheon known for?","chosen":"The Pantheon in Rome, rebuilt under Hadrian around 126 AD, is famous for its enormous concrete dome with a circular opening at the center called the oculus, open to the sky. Its interior height and diameter are nearly equal, forming the shape of a sphere within the building.","rejected":"The Pantheon is known for being a tall narrow tower with a pointed spire and no roof of any kind."}
{"prompt":"What was the Colosseum?","chosen":"The Colosseum, formally the Flavian Amphitheatre, was completed around 80 AD. It was an oval arena used for public spectacles, built of stone and concrete with a system of arches and vaults supporting tiered seating for tens of thousands of spectators.","rejected":"The Colosseum was a small private house where a single Roman family lived on the outskirts of the city."}
{"prompt":"What was the Roman Forum?","chosen":"The Roman Forum was the civic center of the city, an open public square surrounded by government buildings, temples, and monuments. Citizens gathered there for elections, public speeches, legal proceedings, and commerce, and it remained the symbolic heart of public life.","rejected":"The Roman Forum was a farm outside the city where Romans grew grain and kept their livestock."}
{"prompt":"What are the three classical orders of columns?","chosen":"The three Greek classical orders are the Doric, plain and sturdy with a simple capital; the Ionic, more slender with scroll-shaped volutes; and the Corinthian, the most ornate, decorated with carved acanthus leaves. The Romans also added the Tuscan and Composite orders.","rejected":"The three classical orders are small, medium, and large, referring only to the height of the columns."}
{"prompt":"How were Roman roads measured?","chosen":"Distances along Roman roads were marked by milestones set at regular intervals. A Roman mile was one thousand paces, and the Latin mille passuum, meaning a thousand paces, is the origin of the English word mile. The Golden Milestone in Rome marked the point distances were measured from.","rejected":"Roman roads were never measured; travelers simply guessed how far they had gone based on how tired they felt."}
{"prompt":"Who was the first Roman emperor?","chosen":"Augustus was the first Roman emperor, beginning his rule in 27 BC. His reign marked the transition from the Roman Republic to the Roman Empire, concentrating authority in a single ruler while preserving many republican institutions in name.","rejected":"The first Roman emperor was Romulus, who ruled the empire immediately after founding the city in 753 BC."}
{"prompt":"What was the pomerium?","chosen":"The pomerium was a sacred boundary line marking the original limit of the city of Rome. Within the pomerium, certain activities were restricted by religious custom. As Rome grew, the pomerium was extended several times to reflect the city's expansion.","rejected":"The pomerium was a type of fruit grown in Roman gardens and used to make wine for festivals."}
{"prompt":"What does ab urbe condita mean?","chosen":"Ab urbe condita means from the founding of the city. The Romans used this phrase to measure their calendar from the supposed founding of Rome in 753 BC, sometimes recording events as occurring a given number of years after the city's founding.","rejected":"Ab urbe condita means the end of the world, a phrase the Romans used to predict the fall of their empire."}
{"prompt":"Who were the Etruscans?","chosen":"The Etruscans were a people who lived to the north of Rome and strongly influenced early Roman culture. They contributed engineering knowledge, religious practices, and some of Rome's early symbols of authority during the period of the Roman Kingdom.","rejected":"The Etruscans were a people from northern Europe who never had any contact with Rome or its culture."}
{"prompt":"What was the Cloaca Maxima?","chosen":"The Cloaca Maxima, or Great Sewer, was one of the earliest large sewer systems, originally built to drain marshy ground near the Forum and later used to carry waste water to the Tiber. Clean water from aqueducts helped flush the drains.","rejected":"The Cloaca Maxima was the largest temple in Rome, dedicated to the worship of the river gods."}
{"prompt":"What is a basilica in Roman architecture?","chosen":"A basilica was a large rectangular hall used for public business and law courts. It typically had a high central nave flanked by lower side aisles separated by rows of columns, with light entering through upper windows. The form was later adapted for places of worship.","rejected":"A basilica was a small underground room where Romans stored grain and olive oil for the winter."}
{"prompt":"What was the groma used for?","chosen":"The groma was a surveying instrument used by Roman engineers to sight straight lines and right angles. It was a cross of horizontal arms with hanging weights, and with it surveyors achieved the precise alignments their roads and aqueducts required.","rejected":"The groma was a musical instrument played at Roman festivals and had no practical engineering use."}
{"prompt":"How were Roman cities typically laid out?","chosen":"Many Roman cities followed a planned grid layout with two main streets crossing at right angles: the cardo running north to south and the decumanus running east to west. Their intersection often marked the location of the forum and main public buildings.","rejected":"Roman cities were laid out as a series of concentric circles with no straight streets anywhere in the plan."}
{"prompt":"What was a Roman mile based on?","chosen":"A Roman mile was based on one thousand paces. The Latin phrase mille passuum, meaning a thousand paces, is the origin of the English word mile. Milestones marked these distances at regular intervals along Roman roads.","rejected":"A Roman mile was based on the distance a horse could run in one hour without stopping to rest."}
{"prompt":"What materials did Roman builders use?","chosen":"Roman builders used travertine, a local limestone valued for its strength; marble, prized for fine decorative surfaces and imported from across the Mediterranean; and brick and concrete for the structural core, often faced with stone so a sturdy structure could present an elegant exterior.","rejected":"Roman builders used only wood and dried mud, which is why none of their buildings have survived to the present day."}
{"prompt":"What was the Palatine Hill known for?","chosen":"The Palatine Hill is where the earliest settlement of Rome is thought to have begun, and it later became the location of imperial residences. It is one of the seven hills on which the city of Rome was built.","rejected":"The Palatine Hill was located far outside Italy and had no connection to the city of Rome."}
{"prompt":"What did the Capitoline Hill hold?","chosen":"The Capitoline Hill held the most important temples of Rome, including the Temple of Jupiter Optimus Maximus, and served as the religious and political heart of the city. It is one of Rome's seven hills.","rejected":"The Capitoline Hill held the city's marketplace for selling fish and had no religious significance."}
{"prompt":"What is the oculus of the Pantheon?","chosen":"The oculus is the circular opening at the center of the Pantheon's dome, open to the sky, which lets light into the building. It is one of the most distinctive features of the Pantheon's famous concrete dome.","rejected":"The oculus is a statue of an eye placed at the entrance of the Pantheon to ward off bad luck."}
{"prompt":"Why did the Romans use two consuls instead of one?","chosen":"The Romans used two consuls so that each could veto the other, preventing any single person from gaining too much power. Combined with the annual term, this design ensured shared authority and regular turnover in the leadership of the Republic.","rejected":"The Romans used two consuls only because there was too much paperwork for one person to complete in a year."}
{"prompt":"What was the Via Appia?","chosen":"The Via Appia, or Appian Way, was the first great Roman road, begun in 312 BC to connect Rome with the south of Italy. It was built in durable layers and became a model for the extensive Roman road network.","rejected":"The Via Appia was a river in northern Italy that the Romans used for shipping wine and grain."}
{"prompt":"What distinguished the Corinthian order?","chosen":"The Corinthian order was the most ornate of the three Greek classical orders, distinguished by capitals decorated with carved acanthus leaves. The Romans adopted it widely and favored its decorative richness for grand public buildings.","rejected":"The Corinthian order was the plainest column style, with no capital and no decoration of any kind."}
{"prompt":"What role did the aedile play in Roman government?","chosen":"The aedile was a Roman magistrate in the cursus honorum responsible for public buildings, games, and the grain supply. The office came after the quaestor and before the praetor in the sequence of offices an official advanced through.","rejected":"The aedile was the supreme commander of Rome who held power for life and outranked the consuls."}
{"prompt":"How did Polybius describe the Roman constitution?","chosen":"The Greek historian Polybius, who lived in Rome in the 2nd century BC, described the Roman constitution as a mixture of monarchy, aristocracy, and democracy, with each element restraining the others, an early example of balanced and checked authority.","rejected":"Polybius described the Roman constitution as a pure dictatorship with no checks on the ruler's power."}
{"prompt":"What was the Milliarium Aureum?","chosen":"The Milliarium Aureum, or Golden Milestone, was a gilded monument in Rome regarded as the point from which all distances in the empire were measured. It reflected the idea that the road network radiated outward from the capital.","rejected":"The Milliarium Aureum was a golden crown worn by Roman consuls during religious festivals."}
{"prompt":"What is the difference between the cardo and the decumanus?","chosen":"In a planned Roman city, the cardo was the main street running north to south, and the decumanus was the main street running east to west. The two crossed at right angles, and their intersection often marked the forum and main public buildings.","rejected":"The cardo and decumanus were two rival Roman armies that fought each other for control of the city streets."}
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
    "$INSTRUCTOR" init --course "Ancient Rome QA Test" --output "$project_dir"
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
        --system-prompt "You are a helpful ancient history tutor. Answer questions accurately using course materials."
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
declare -a RUN_MODELS=()
declare -a SKIP_MODELS=()

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
declare -a RESULT_LABELS=()
declare -a RESULT_STATUSES=()
declare -a RESULT_RESPONSES=()
declare -a RESULT_LOAD_TIMES=()
declare -a RESULT_CHAT_TIMES=()
declare -a RESULT_PIPELINE_TIMES=()
declare -a RESULT_STDERRS=()

# Add skip entries for unavailable models
for shortcut in ${SKIP_MODELS[@]+"${SKIP_MODELS[@]}"}; do
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
for shortcut in ${RUN_MODELS[@]+"${RUN_MODELS[@]}"}; do
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
for status in ${RESULT_STATUSES[@]+"${RESULT_STATUSES[@]}"}; do
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
