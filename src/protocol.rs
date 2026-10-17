//! Wire protocol: three task endpoints, validation matrix, unified error
//! format, usage accounting, body limit. Contract rulings:
//!   - /v1/score takes caller-supplied `levels` (pinned by the golden corpus)
//!   - options/levels lower bound is 2 (official forward topk(2) cannot run k=1)

use std::sync::Arc;

use axum::{
    Router,
    extract::{DefaultBodyLimit, FromRequest, Request, State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;

use crate::pipeline::Pool;

/// Request body cap: larger bodies get 413 payload_too_large.
pub const MAX_BODY_BYTES: usize = 1024 * 1024;
/// options/levels upper bound (protocol contract).
pub const MAX_OPTIONS: usize = 32;
/// options/levels lower bound: k=1 crashes the official forward (topk(2) on
/// a K=1 axis); bug-for-bug, Verdict rejects what the reference cannot run.
pub const MIN_OPTIONS: usize = 2;

// ---------------------------------------------------------------- requests

#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceRequest {
    pub question: String,
    #[serde(default)]
    pub state: String,
    pub options: Vec<String>,
    /// Scheduling priority, vLLM-aligned: smaller = admitted earlier.
    #[serde(default)]
    pub priority: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScoreRequest {
    pub question: String,
    #[serde(default)]
    pub state: String,
    pub levels: Vec<String>,
    /// Scheduling priority, vLLM-aligned: smaller = admitted earlier.
    #[serde(default)]
    pub priority: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoulRequest {
    pub claim: String,
    #[serde(default)]
    pub state: String,
    /// Scheduling priority, vLLM-aligned: smaller = admitted earlier.
    #[serde(default)]
    pub priority: Option<i64>,
}

impl ChoiceRequest {
    /// Effective scheduling priority (default 0, vLLM-aligned).
    pub fn priority(&self) -> i64 {
        self.priority.unwrap_or(0)
    }
}

impl ScoreRequest {
    pub fn priority(&self) -> i64 {
        self.priority.unwrap_or(0)
    }
}

impl NoulRequest {
    pub fn priority(&self) -> i64 {
        self.priority.unwrap_or(0)
    }
}

// ---------------------------------------------------------------- responses

#[derive(Debug, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
}

#[derive(Debug, Serialize)]
pub struct ChoiceResponse {
    pub choice: usize,
    pub probabilities: Vec<f64>,
    pub usage: Usage,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct ScoreResponse {
    pub score: f64,
    pub probabilities: Vec<f64>,
    pub usage: Usage,
    pub latency_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct NoulResponse {
    pub probability: f64,
    pub usage: Usage,
    pub latency_ms: u64,
}

// ---------------------------------------------------------------- errors

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    InvalidRequest(String),
    #[error("not found")]
    NotFound,
    #[error("request body exceeds the 1 MiB limit")]
    PayloadTooLarge,
    #[error("internal error")]
    Internal(String),
    #[error("server overloaded, try again later")]
    Overloaded,
}

impl ApiError {
    fn status(&self) -> StatusCode {
        match self {
            ApiError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound => StatusCode::NOT_FOUND,
            ApiError::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            ApiError::InvalidRequest(_) => "invalid_request",
            ApiError::NotFound => "not_found",
            ApiError::PayloadTooLarge => "payload_too_large",
            ApiError::Internal(_) => "internal",
            ApiError::Overloaded => "overloaded",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // Internal errors never leak internals to the wire (message stays generic).
        let message = match &self {
            ApiError::Internal(_) => "internal error".to_string(),
            other => other.to_string(),
        };
        let status = self.status();
        let body = Json(json!({"error": {"type": self.kind(), "message": message}}));
        (status, body).into_response()
    }
}

// ---------------------------------------------------------------- validation

pub(crate) fn require_non_empty(field: &str, value: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(ApiError::InvalidRequest(format!(
            "{field} must be a non-empty string"
        )));
    }
    Ok(())
}

/// Shared options/levels rules: MIN_OPTIONS..=MAX_OPTIONS items, each non-empty.
pub(crate) fn validate_option_list(field: &str, items: &[String]) -> Result<(), ApiError> {
    if items.len() < MIN_OPTIONS {
        return Err(ApiError::InvalidRequest(format!(
            "{field} must contain at least {MIN_OPTIONS} items (got {})",
            items.len()
        )));
    }
    if items.len() > MAX_OPTIONS {
        return Err(ApiError::InvalidRequest(format!(
            "{field} must contain at most {MAX_OPTIONS} items (got {})",
            items.len()
        )));
    }
    for (i, item) in items.iter().enumerate() {
        if item.trim().is_empty() {
            return Err(ApiError::InvalidRequest(format!(
                "{field}[{i}] must be a non-empty string"
            )));
        }
    }
    Ok(())
}

impl ChoiceRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        require_non_empty("question", &self.question)?;
        validate_option_list("options", &self.options)
    }
}

impl ScoreRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        require_non_empty("question", &self.question)?;
        validate_option_list("levels", &self.levels)
    }
}

impl NoulRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        require_non_empty("claim", &self.claim)
    }
}

// ---------------------------------------------------------------- extractor

/// `Json<T>` that maps every serde rejection onto the unified error format
/// (missing fields, wrong types, malformed JSON -> 400 invalid_request).
pub struct ValidJson<T>(pub T);

impl<S, T> FromRequest<S> for ValidJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(ValidJson(value)),
            Err(rejection) => Err(map_json_rejection(rejection)),
        }
    }
}

fn map_json_rejection(rejection: JsonRejection) -> ApiError {
    ApiError::InvalidRequest(rejection.to_string())
}

// ---------------------------------------------------------------- middleware

/// Fast-path 413 for requests that declare an oversized Content-Length.
/// `DefaultBodyLimit` (below) is the backstop for chunked bodies.
async fn body_limit(req: Request, next: axum::middleware::Next) -> Response {
    let oversized = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|len| len > MAX_BODY_BYTES);
    if oversized {
        return ApiError::PayloadTooLarge.into_response();
    }
    next.run(req).await
}

// ---------------------------------------------------------------- handlers

/// Prometheus exposition (ops surface; plain text by design).
async fn metrics_text() -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        crate::metrics::render(),
    )
        .into_response()
}

async fn health(State(pipeline): State<Arc<Pool>>) -> Response {
    if pipeline.health() {
        Json(json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")})).into_response()
    } else {
        // Real liveness probe (TEI Infer::health pattern): an unhealthy engine
        // answers 503, not a static "ok".
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unavailable", "version": env!("CARGO_PKG_VERSION")})),
        )
            .into_response()
    }
}

async fn not_found() -> ApiError {
    ApiError::NotFound
}

async fn choice(
    State(pipeline): State<Arc<Pool>>,
    ValidJson(req): ValidJson<ChoiceRequest>,
) -> Result<Json<ChoiceResponse>, ApiError> {
    req.validate()?;
    Ok(Json(pipeline.choice(req).await?))
}

async fn score(
    State(pipeline): State<Arc<Pool>>,
    ValidJson(req): ValidJson<ScoreRequest>,
) -> Result<Json<ScoreResponse>, ApiError> {
    req.validate()?;
    Ok(Json(pipeline.score(req).await?))
}

async fn noul(
    State(pipeline): State<Arc<Pool>>,
    ValidJson(req): ValidJson<NoulRequest>,
) -> Result<Json<NoulResponse>, ApiError> {
    req.validate()?;
    Ok(Json(pipeline.noul(req).await?))
}

// ---------------------------------------------------------------- router

pub fn router(pool: Arc<Pool>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_text))
        .route("/v1/choice", post(choice))
        .route("/v1/score", post(score))
        .route("/v1/noul", post(noul))
        // OpenAI-compatible facade; the native endpoints above stay the
        // golden-graded contract.
        .merge(crate::compat::routes())
        // The category's own Jev shape (one state, many typed questions).
        .merge(crate::systemone::routes())
        // OpenAI's cross-ecosystem Decisions dialect.
        .merge(crate::decisions::routes())
        .fallback(not_found)
        .layer(axum::middleware::from_fn(body_limit))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(pool)
}
