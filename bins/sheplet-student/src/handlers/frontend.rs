use std::sync::Arc;

use axum::response::Html;
use axum::routing::get;
use axum::Router;
use axum::response::IntoResponse;
use axum::http::{StatusCode, header};

use crate::app_state::AppState;

const INDEX_HTML: &str = include_str!("../frontend/index.html");
const STYLE_CSS: &str = include_str!("../frontend/style.css");
const APP_JS: &str = include_str!("../frontend/app.js");

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(index))
        .route("/static/style.css", get(style))
        .route("/static/app.js", get(app_js))
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn style() -> impl IntoResponse {
    (StatusCode::OK, [(header::CONTENT_TYPE, "text/css")], STYLE_CSS)
}

async fn app_js() -> impl IntoResponse {
    (StatusCode::OK, [(header::CONTENT_TYPE, "application/javascript")], APP_JS)
}
