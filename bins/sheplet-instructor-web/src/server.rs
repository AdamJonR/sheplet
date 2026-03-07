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
        .merge(handlers::projects::routes())
        .merge(handlers::config::routes())
        .merge(handlers::templates::routes())
        .merge(handlers::ingest::routes())
        .merge(handlers::model::routes())
        .merge(handlers::finetune::routes())
        .merge(handlers::bundle::routes())
        .merge(handlers::tasks::routes())
        .with_state(state)
}
