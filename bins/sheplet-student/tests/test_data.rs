use std::path::PathBuf;
use std::process::Output;

pub const MODEL_CLI_NAME: &str = "llama-3.2-1b";
pub const MODEL_DIR_NAME: &str = "meta-llama--Llama-3.2-1B-Instruct";

pub const ROMAN_FOUNDING_TEXT: &str = r#"The Founding and Geography of Rome

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
the city's founding."#;

pub const ROMAN_GOVERNMENT_TEXT: &str = r#"Government of the Roman Republic

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
mixture of monarchy, aristocracy, and democracy, each element restraining the others."#;

pub const ROME_DPO_DATA: &str = r#"{"prompt":"When was Rome traditionally founded?","chosen":"Rome was traditionally founded in 753 BC, a date calculated by the scholar Varro. According to legend the city was established by Romulus, who became its first king, though archaeology shows settlements existed on the site from at least the 8th century BC.","rejected":"Rome was founded in 1200 AD during the Middle Ages by a group of traveling merchants."}
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
{"prompt":"What is the difference between the cardo and the decumanus?","chosen":"In a planned Roman city, the cardo was the main street running north to south, and the decumanus was the main street running east to west. The two crossed at right angles, and their intersection often marked the forum and main public buildings.","rejected":"The cardo and decumanus were two rival Roman armies that fought each other for control of the city streets."}"#;

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

/// Check if the test model is available in downloaded-models/.
pub fn test_model_available() -> bool {
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
    std::fs::write(sources_dir.join("roman_founding.txt"), ROMAN_FOUNDING_TEXT).unwrap();
    std::fs::write(sources_dir.join("roman_government.txt"), ROMAN_GOVERNMENT_TEXT).unwrap();

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
        .args(["model", "--name", MODEL_CLI_NAME, "--project"])
        .arg(&project_dir)
        .assert()
        .success();
    println!("  Model: {:?}", start.elapsed());

    // DPO data + finetune
    let dpo_path = project_dir.join("dpo_data.jsonl");
    std::fs::write(&dpo_path, ROME_DPO_DATA).unwrap();
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
            "You are a helpful ancient history tutor. Answer questions accurately using course materials.",
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
