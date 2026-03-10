use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use conversations::ConversationStore;
use tokio::sync::RwLock;

use sheplet_student::app_state::AppState;
use sheplet_student::course::CourseManager;
use sheplet_student::server;

const DEFAULT_PORT: u16 = 8420;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut port = DEFAULT_PORT;
    let mut base_dir = default_base_dir();
    let mut no_adapter = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_PORT);
            }
            "--dir" => {
                i += 1;
                if let Some(dir) = args.get(i) {
                    base_dir = PathBuf::from(dir);
                }
            }
            "--no-adapter" => {
                no_adapter = true;
            }
            "--help" | "-h" => {
                println!("sheplet-student - Course assistant desktop client");
                println!();
                println!("Usage: sheplet-student [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --port <PORT>    Port to listen on (default: {DEFAULT_PORT})");
                println!("  --dir <DIR>      Base directory for data (default: ~/sheplet-student)");
                println!("  --no-adapter     Skip LoRA adapter, use base model only (debug)");
                println!("  --help, -h       Show this help message");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    std::fs::create_dir_all(&base_dir)?;

    let conversations_path = base_dir.join("conversations");
    let conversations = Arc::new(ConversationStore::open(&conversations_path)?);

    let state = Arc::new(AppState {
        courses: RwLock::new(CourseManager::new()),
        conversations,
        base_dir: base_dir.clone(),
        no_adapter,
    });

    if no_adapter {
        println!("WARNING: --no-adapter flag set — LoRA adapters will NOT be loaded");
    }

    let app = server::build_router(state);
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    println!("Sheplet Student running at http://{addr}");
    println!("Data directory: {}", base_dir.display());
    println!("Press Ctrl+C to stop.");

    // Try to open browser
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg(format!("http://{addr}"))
            .spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open")
            .arg(format!("http://{addr}"))
            .spawn();
    }

    axum::serve(listener, app).await?;
    Ok(())
}

fn default_base_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("sheplet-student")
}
