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
        .merge(handlers::bundles::routes())
        .merge(handlers::courses::routes())
        .merge(handlers::chat::routes())
        .merge(handlers::conversations::routes())
        .merge(handlers::settings::routes())
        .layer(cors)
        .with_state(state)
}
