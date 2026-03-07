mod app_state;
mod handlers;
mod project;
mod server;
mod task_manager;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::RwLock;

use app_state::AppState;
use task_manager::TaskManager;

const DEFAULT_PORT: u16 = 8421;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut port = DEFAULT_PORT;
    let mut base_dir = default_base_dir();

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
            "--help" | "-h" => {
                println!("sheplet-instructor-web - Instructor web interface");
                println!();
                println!("Usage: sheplet-instructor-web [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --port <PORT>  Port to listen on (default: {DEFAULT_PORT})");
                println!("  --dir <DIR>    Base directory for projects (default: ~/sheplet-instructor)");
                println!("  --help, -h     Show this help message");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    std::fs::create_dir_all(&base_dir)?;

    let state = Arc::new(AppState {
        base_dir: base_dir.clone(),
        active_project: RwLock::new(None),
        tasks: Arc::new(TaskManager::new()),
    });

    let app = server::build_router(state);
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    println!("Sheplet Instructor running at http://{addr}");
    println!("Projects directory: {}", base_dir.display());
    println!("Press Ctrl+C to stop.");

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
        .join("sheplet-instructor")
}
