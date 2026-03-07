use std::sync::Arc;

use axum::Router;

use crate::app_state::AppState;
use crate::handlers;

pub fn build_router(state: Arc<AppState>) -> Router {
    // No CORS layer — the frontend is served from the same origin via include_str!(),
    // so cross-origin requests are never needed. Allowing Any origin would let
    // malicious websites make requests to the local server.
    Router::new()
        .merge(handlers::frontend::routes())
        .merge(handlers::bundles::routes())
        .merge(handlers::courses::routes())
        .merge(handlers::chat::routes())
        .merge(handlers::conversations::routes())
        .merge(handlers::settings::routes())
        .with_state(state)
}
