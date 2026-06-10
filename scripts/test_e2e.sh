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
    echo "Usage: $0 [--model MODEL] [--lora yes|no]"
    echo ""
    echo "Models: llama1b, llama3b (default), qwen0.5b, qwen1.5b, qwen3b,"
    echo "        gemma2b, gemma2-2b, mistral7b, phi3"
    echo ""
    echo "Defaults: --model llama3b --lora no"
    exit 1
}

# Defaults
MODEL="llama3b"
LORA="no"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --lora)       LORA="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

case "$MODEL" in
  llama1b)    MODEL_NAME="llama-3.2-1b" ;;
  llama3b)    MODEL_NAME="llama-3.2-3b" ;;
  qwen0.5b)   MODEL_NAME="qwen2.5-0.5b" ;;
  qwen1.5b)   MODEL_NAME="qwen2.5-1.5b" ;;
  qwen3b)     MODEL_NAME="qwen2.5-3b" ;;
  gemma2b)    MODEL_NAME="gemma-2b" ;;
  gemma2-2b)  MODEL_NAME="gemma-2-2b" ;;
  mistral7b)  MODEL_NAME="mistral-7b" ;;
  phi3)       MODEL_NAME="phi-3-mini-4k-instruct" ;;
  *)          echo "Unknown model: $MODEL"; usage ;;
esac

case "$LORA" in
  yes|no) ;;
  *) echo "Invalid --lora value: $LORA (must be yes or no)"; usage ;;
esac

echo "Model: $MODEL_NAME (lora: $LORA)"

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
"$INSTRUCTOR" init --course "Ancient Rome" --output "$TEST_DIR"
TIME_INIT=$(( SECONDS - STEP_START ))
echo "--- Init completed in ${TIME_INIT}s ---"

# --- Step 4: Create test documents --------------------------------------------

mkdir -p "$TEST_DIR/sources"

cat > "$TEST_DIR/sources/roman_founding.txt" << 'FOUNDING_EOF'
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

cat > "$TEST_DIR/sources/roman_government.txt" << 'GOVERNMENT_EOF'
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

cat > "$TEST_DIR/sources/roman_engineering.txt" << 'ENGINEERING_EOF'
Roman Engineering and Infrastructure

The Romans were among the most accomplished engineers of the ancient world. Their roads,
aqueducts, bridges, and public works were built to last, and many survive today. Roman
engineering combined practical experience with standardized techniques that could be applied
consistently across a vast territory.

Roman roads were famous for their durability and straightness. The first great road, the
Appian Way (Via Appia), was begun in 312 BC and connected Rome to the south of Italy. Roads
were built in layers: a foundation of large stones, followed by gravel and sand, topped with
fitted paving stones that shed rainwater. Roads were slightly raised in the center, or
cambered, so water drained to ditches on either side. The saying "all roads lead to Rome"
reflects how the network radiated outward from the capital.

Distances along Roman roads were marked by milestones, stone columns set at regular
intervals. A Roman mile was one thousand paces, and the Latin "mille passuum," meaning "a
thousand paces," is the origin of the English word "mile." A gilded monument in Rome called
the Milliarium Aureum, or Golden Milestone, was regarded as the point from which all
distances in the empire were measured.

Aqueducts carried fresh water into cities from distant springs and rivers. They worked
entirely by gravity: the channel sloped gently downhill for the whole of its length, often
many kilometers, so the water flowed steadily without any pump. Engineers maintained a
precise, gradual gradient, and where valleys interrupted the route, they carried the channel
across on rows of arches. The Pont du Gard in southern France is a surviving aqueduct bridge
that crosses a river valley on three tiers of stone arches.

Roman concrete, called opus caementicium, was one of Rome's most important innovations. It
was made by mixing lime, water, and a volcanic ash called pozzolana with pieces of stone or
brick. This mixture set into a hard, durable mass and could even harden underwater, which
made it ideal for harbors, foundations, and large vaulted structures. Roman concrete allowed
builders to create spaces and spans that earlier stone construction could not achieve.

The arch was central to Roman building. An arch carries weight by directing the load outward
and downward along a curve of wedge-shaped stones called voussoirs. The central stone at the
top, the keystone, locks the others in place. Because the arch is strong in compression, it
let the Romans bridge wide openings and stack tiers of arches to great heights. Repeating an
arch in a line produces a vault, and rotating it produces a dome.

Roman cities were served by sewers and drainage systems. The Cloaca Maxima, or Great Sewer,
was one of the earliest large sewer systems, originally built to drain marshy ground near the
Forum and later used to carry waste water to the Tiber. Clean water from aqueducts supplied
public fountains, baths, and latrines, and the used water helped flush the drains.

To lay out their works accurately, Roman surveyors used an instrument called the groma, a
cross of horizontal arms with hanging weights that allowed them to sight straight lines and
right angles. With simple tools like the groma and careful measurement, Roman engineers
achieved the precise alignments their roads and aqueducts required.
ENGINEERING_EOF

cat > "$TEST_DIR/sources/roman_architecture.txt" << 'ARCHITECTURE_EOF'
Roman Architecture and Public Buildings

Roman architecture combined ideas borrowed from earlier cultures, especially the Greeks and
Etruscans, with Rome's own innovations in concrete and the arch. The result was a style
capable of enclosing vast interior spaces and serving large urban populations. Roman public
buildings were designed to impress as well as to function.

The Pantheon in Rome is one of the best-preserved buildings of the ancient world. Rebuilt
under the emperor Hadrian around 126 AD, it is famous for its enormous concrete dome. At the
center of the dome is a circular opening called the oculus, which is open to the sky and lets
in light. The Pantheon's dome was the largest of its kind for many centuries, and its
interior height and diameter are nearly equal, forming the shape of a perfect sphere resting
within the building.

The Colosseum, formally the Flavian Amphitheatre, was completed around 80 AD. It was an oval
arena used for public spectacles and games that the city provided as free entertainment. Built
of stone and concrete, it used a system of arches and vaults to support tiered seating for
tens of thousands of spectators. A network of passages and stairways, called vomitoria,
allowed the large crowds to enter and leave quickly. The exterior arcade displayed the
classical column styles stacked in tiers.

The Roman Forum was the civic center of the city. It was an open public square surrounded by
government buildings, temples, and monuments, where citizens gathered for elections, public
speeches, legal proceedings, and commerce. As Rome grew, later emperors added additional
forums nearby, but the original Forum remained the symbolic heart of public life.

The basilica was a large rectangular hall used for public business and law courts. It
typically had a high central aisle, called a nave, flanked by lower side aisles separated by
rows of columns, with light entering through upper windows. The basilica form was practical
for sheltering large gatherings, and its design was later adapted for places of worship.

Classical architecture used standardized column styles called orders. The Greeks developed
three: the Doric order, plain and sturdy with a simple capital; the Ionic order, more slender
and marked by scroll-shaped volutes at the top; and the Corinthian order, the most ornate,
decorated with carved acanthus leaves. The Romans adopted all three and added two of their
own: the Tuscan order, a simplified version of the Doric, and the Composite order, which
combined Ionic volutes with Corinthian leaves.

Roman builders worked in a range of materials. Travertine, a local limestone, was widely used
for its strength and pale color. Marble, prized for fine surfaces and decoration, was quarried
in Italy and imported from across the Mediterranean. Brick and concrete formed the structural
core of many buildings, often faced with stone or marble so that a sturdy concrete structure
could present an elegant exterior.

Many Roman cities followed a planned grid layout adapted from earlier traditions. Two main
streets crossed at right angles: the cardo running north to south and the decumanus running
east to west. Their intersection often marked the location of the forum and the main public
buildings, giving Roman towns an orderly and recognizable plan wherever they were founded.
ARCHITECTURE_EOF

echo "Created test documents in $TEST_DIR/sources/"

# --- Step 5: Ingest -----------------------------------------------------------

echo ""
echo "=== Ingest ==="
STEP_START=$SECONDS
"$INSTRUCTOR" ingest --sources "$TEST_DIR/sources" --project "$TEST_DIR"
TIME_INGEST=$(( SECONDS - STEP_START ))
echo "--- Ingest completed in ${TIME_INGEST}s ---"

# --- Step 6: Model download ---------------------------------------------------

echo ""
echo "=== Model ==="
STEP_START=$SECONDS
"$INSTRUCTOR" model --name "$MODEL_NAME" --project "$TEST_DIR"
TIME_MODEL=$(( SECONDS - STEP_START ))
echo "--- Model completed in ${TIME_MODEL}s ---"

# --- Step 7–8: SFT + DPO training (if LoRA enabled) -------------------------

TIME_SFT=0
TIME_DPO=0
if [ "$LORA" = "yes" ]; then

# --- SFT training data (input/output pairs from source documents) ---
cat > "$TEST_DIR/sft_data.jsonl" << 'SFT_EOF'
{"input":"When was Rome traditionally founded?","output":"Rome was traditionally founded in 753 BC, a date calculated by the Roman scholar Varro. According to legend the city was established by Romulus, who became its first king, though archaeology shows permanent settlements existed on the site from at least the 8th century BC."}
{"input":"On how many hills was the city of Rome built?","output":"Rome was built on seven hills overlooking the Tiber River: the Palatine, Capitoline, Aventine, Caelian, Esquiline, Viminal, and Quirinal. The Palatine is where the earliest settlement is thought to have begun, and the Capitoline held the city's most important temples."}
{"input":"What are the three periods of Roman history?","output":"Roman history is divided into three periods: the Roman Kingdom (753–509 BC), ruled by kings; the Roman Republic (509–27 BC), governed by elected officials and a Senate; and the Roman Empire (27 BC onward), which began when Augustus became the first emperor."}
{"input":"Why was the Tiber River important to Rome?","output":"The Tiber River provided fresh water, a route for trade, and a natural defensive barrier. Rome's location about 25 kilometers inland gave it access to the sea through the river while keeping it protected from coastal raids, and the Tiber Island offered the easiest crossing point."}
{"input":"When was the Roman Republic established?","output":"The Roman Republic was established in 509 BC, when the Romans replaced their monarchy with a system of elected officials and representative bodies. The word republic comes from the Latin res publica, meaning the public affair, reflecting the idea that the state belonged to its citizens."}
{"input":"What was the Roman Senate?","output":"The Senate was the most prestigious institution of the Republic, a council of experienced statesmen who advised the magistrates and guided policy on finance, foreign relations, and religion. Although its formal power was advisory, its authority and continuity made it the dominant force in Roman politics."}
{"input":"How many consuls led the Roman Republic, and why?","output":"The Republic was led by two consuls, elected annually. Having two consuls who could each veto the other prevented any single person from holding too much power, and the one-year term ensured regular turnover in leadership."}
{"input":"What was the cursus honorum?","output":"The cursus honorum, or course of honors, was the sequence of offices a Roman official advanced through. It began with the quaestor, who managed finances, then the aedile, responsible for public buildings and games, then the praetor, who administered justice, and finally the consul, the highest regular office."}
{"input":"What were the Twelve Tables?","output":"The Twelve Tables, written around 451 BC, were Rome's first written code of law. By inscribing the laws on twelve bronze tablets displayed in public, Rome made the rules known to everyone rather than only to priests and aristocrats. They became the foundation of Roman law."}
{"input":"What does SPQR stand for?","output":"SPQR stands for Senatus Populusque Romanus, meaning the Senate and the People of Rome. It appeared on official documents, public monuments, and standards, symbolizing the shared authority of the governing council and the citizens."}
{"input":"What were the tribunes of the plebs?","output":"The tribunes of the plebs were officials created to protect the interests of the common citizens, the plebeians. They could veto actions of magistrates and the Senate that they judged harmful to the people, and their persons were legally protected, giving ordinary citizens a formal voice in government."}
{"input":"What was the first major Roman road?","output":"The first great Roman road was the Appian Way (Via Appia), begun in 312 BC, which connected Rome to the south of Italy. Roman roads were built in layers of stone, gravel, and fitted paving, and were cambered in the center so rainwater drained to ditches on either side."}
{"input":"How did Roman aqueducts move water?","output":"Roman aqueducts worked entirely by gravity. The channel sloped gently downhill for its whole length, often many kilometers, so water flowed steadily without any pump. Where valleys interrupted the route, engineers carried the channel across on rows of stone arches, as at the Pont du Gard."}
{"input":"What was Roman concrete made of?","output":"Roman concrete, called opus caementicium, was made by mixing lime, water, and a volcanic ash called pozzolana with pieces of stone or brick. It set into a hard, durable mass and could even harden underwater, making it ideal for harbors, foundations, and large vaulted structures."}
{"input":"How does a Roman arch carry weight?","output":"An arch carries weight by directing the load outward and downward along a curve of wedge-shaped stones called voussoirs. The central stone at the top, the keystone, locks the others in place. Because the arch is strong in compression, it let the Romans bridge wide openings and build to great heights."}
{"input":"What is the Pantheon known for?","output":"The Pantheon in Rome, rebuilt under Hadrian around 126 AD, is famous for its enormous concrete dome. At the center of the dome is a circular opening called the oculus, open to the sky. Its interior height and diameter are nearly equal, forming the shape of a sphere resting within the building."}
{"input":"What was the Colosseum?","output":"The Colosseum, formally the Flavian Amphitheatre, was completed around 80 AD. It was an oval arena used for public spectacles, built of stone and concrete with a system of arches and vaults supporting tiered seating for tens of thousands of spectators."}
{"input":"What was the Roman Forum?","output":"The Roman Forum was the civic center of the city, an open public square surrounded by government buildings, temples, and monuments. Citizens gathered there for elections, public speeches, legal proceedings, and commerce, and it remained the symbolic heart of public life."}
{"input":"What are the three classical orders of columns?","output":"The three Greek classical orders are the Doric, plain and sturdy with a simple capital; the Ionic, more slender with scroll-shaped volutes; and the Corinthian, the most ornate, decorated with carved acanthus leaves. The Romans adopted all three and added the Tuscan and Composite orders."}
{"input":"How were Roman roads measured?","output":"Distances along Roman roads were marked by milestones set at regular intervals. A Roman mile was one thousand paces, and the Latin mille passuum, meaning a thousand paces, is the origin of the English word mile. The Golden Milestone in Rome was regarded as the point from which distances were measured."}
SFT_EOF

echo "Created SFT training data (20 examples)"

echo ""
echo "=== SFT Train ==="
STEP_START=$SECONDS
"$INSTRUCTOR" finetune --method sft --data "$TEST_DIR/sft_data.jsonl" --project "$TEST_DIR"
TIME_SFT=$(( SECONDS - STEP_START ))
echo "--- SFT Train completed in ${TIME_SFT}s ---"

# --- DPO training data ---
cat > "$TEST_DIR/dpo_data.jsonl" << 'DPO_EOF'
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
"$INSTRUCTOR" config --project "$TEST_DIR" --system-prompt "You are a helpful ancient history tutor. Answer questions accurately using course materials."
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
printf "  %-14s %ss\n" "SFT Train:" "$TIME_SFT"
printf "  %-14s %ss\n" "DPO Train:" "$TIME_DPO"
fi
printf "  %-14s %ss\n" "Config:" "$TIME_CONFIG"
printf "  %-14s %ss\n" "Bundle:" "$TIME_BUNDLE"
printf "  %-14s %ss\n" "Total:" "$TOTAL_ELAPSED"

echo ""
echo "Done!"
