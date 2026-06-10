use std::sync::Arc;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use axum::Router;

use crate::app_state::AppState;
use crate::handlers;

/// Reject requests whose Host header is not a local address. The server only
/// binds 127.0.0.1, but a malicious website can use DNS rebinding to point its
/// own domain at 127.0.0.1 and drive the API from the victim's browser; the
/// browser then sends the attacker's domain in the Host header, which this
/// check catches.
async fn validate_host(req: Request, next: Next) -> Result<Response, StatusCode> {
    let host_ok = req
        .headers()
        .get(axum::http::header::HOST)
        .and_then(|h| h.to_str().ok())
        .map(host_is_local)
        .unwrap_or(false);
    if host_ok {
        Ok(next.run(req).await)
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}

fn host_is_local(header: &str) -> bool {
    // Strip the port, handling bracketed IPv6 ("[::1]:8080")
    let host = if let Some(rest) = header.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        header.split(':').next().unwrap_or("")
    };
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

pub fn build_router(state: Arc<AppState>) -> Router {
    // No CORS layer — the frontend is served from the same origin via include_str!(),
    // so cross-origin requests are never needed. Allowing Any origin would let
    // malicious websites make requests to the local server.
    Router::new()
        .merge(handlers::frontend::routes())
        .merge(handlers::projects::routes())
        .merge(handlers::config::routes())
        .merge(handlers::templates::routes())
        .merge(handlers::ingest::routes())
        .merge(handlers::model::routes())
        .merge(handlers::finetune::routes())
        .merge(handlers::bundle::routes())
        .merge(handlers::tasks::routes())
        .layer(axum::middleware::from_fn(validate_host))
        .with_state(state)
}


#[cfg(test)]
mod tests {
    use super::host_is_local;

    #[test]
    fn test_host_validation() {
        assert!(host_is_local("127.0.0.1:8080"));
        assert!(host_is_local("127.0.0.1"));
        assert!(host_is_local("localhost:3000"));
        assert!(host_is_local("localhost"));
        assert!(host_is_local("[::1]:8080"));
        assert!(!host_is_local("evil.example.com"));
        assert!(!host_is_local("evil.example.com:8080"));
        assert!(!host_is_local("127.0.0.1.evil.example.com"));
        assert!(!host_is_local(""));
    }
}
