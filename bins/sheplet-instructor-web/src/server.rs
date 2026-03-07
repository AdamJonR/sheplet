use std::sync::Arc;

use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use crate::app_state::AppState;
use crate::handlers;

pub fn build_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

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
        .layer(cors)
        .with_state(state)
}
