use crate::pipeline::Pool;
use axum::{
    Router,
    extract::{DefaultBodyLimit, FromRequest, Request, State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use std::sync::Arc;

pub const MAX_OPTIONS: usize = 32;

pub const MIN_OPTIONS: usize = 2;
#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceRequest {
    pub question: String,
    #[serde(default)]
    pub state: String,
    pub options: Vec<String>,

    #[serde(default)]
    pub priority: Option<i64>,
}
#[derive(Debug, Clone, Deserialize)]
pub struct ScoreRequest {
    pub question: String,
    #[serde(default)]
    pub state: String,
    pub levels: Vec<String>,

    #[serde(default)]
    pub priority: Option<i64>,
}
#[derive(Debug, Clone, Deserialize)]
pub struct NoulRequest {
    pub claim: String,
    #[serde(default)]
    pub state: String,

    #[serde(default)]
    pub priority: Option<i64>,
}
impl ChoiceRequest {
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
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    InvalidRequest(String),
    #[error("not found")]
    NotFound,
    #[error("request body exceeds the {0} MiB limit")]
    PayloadTooLarge(usize),
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
            ApiError::PayloadTooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            ApiError::InvalidRequest(_) => "invalid_request",
            ApiError::NotFound => "not_found",
            ApiError::PayloadTooLarge(_) => "payload_too_large",
            ApiError::Internal(_) => "internal",
            ApiError::Overloaded => "overloaded",
        }
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let message = match &self {
            ApiError::Internal(_) => "internal error".to_string(),
            other => other.to_string(),
        };
        let status = self.status();
        let body = Json(json!({"error": {"type": self.kind(), "message": message}}));
        (status, body).into_response()
    }
}
pub(crate) fn require_non_empty(field: &str, value: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(ApiError::InvalidRequest(format!(
            "{field} must be a non-empty string"
        )));
    }
    Ok(())
}

pub(crate) fn validate_option_list_max(
    field: &str,
    items: &[String],
    max: usize,
) -> Result<(), ApiError> {
    if items.len() < MIN_OPTIONS {
        return Err(ApiError::InvalidRequest(format!(
            "{field} must contain at least {MIN_OPTIONS} items (got {})",
            items.len()
        )));
    }
    if items.len() > max {
        return Err(ApiError::InvalidRequest(format!(
            "{field} must contain at most {max} items (got {})",
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
pub(crate) fn validate_option_list(field: &str, items: &[String]) -> Result<(), ApiError> {
    validate_option_list_max(field, items, MAX_OPTIONS)
}

fn choice_ceiling(family: crate::family::Family) -> usize {
    match family {
        crate::family::Family::LetterBranches => crate::family::letter_branches::MAX_CHOICE_OPTIONS,
        _ => MAX_OPTIONS,
    }
}

fn score_ceiling(family: crate::family::Family) -> usize {
    match family {
        crate::family::Family::LetterBranches => crate::family::letter_branches::MAX_SCORE_LEVELS,
        _ => MAX_OPTIONS,
    }
}
impl ChoiceRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        require_non_empty("question", &self.question)?;
        validate_option_list("options", &self.options)
    }

    pub fn validate_with_family(&self, family: crate::family::Family) -> Result<(), ApiError> {
        if family != crate::family::Family::LetterBranches {
            require_non_empty("question", &self.question)?;
        }
        validate_option_list_max("options", &self.options, choice_ceiling(family))
    }
}
impl ScoreRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        require_non_empty("question", &self.question)?;
        validate_option_list("levels", &self.levels)
    }

    pub fn validate_with_family(&self, family: crate::family::Family) -> Result<(), ApiError> {
        if family != crate::family::Family::LetterBranches {
            require_non_empty("question", &self.question)?;
        }
        validate_option_list_max("levels", &self.levels, score_ceiling(family))
    }
}
impl NoulRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        require_non_empty("claim", &self.claim)
    }

    pub fn validate_with_family(&self, family: crate::family::Family) -> Result<(), ApiError> {
        if family != crate::family::Family::LetterBranches {
            require_non_empty("claim", &self.claim)?;
        }
        Ok(())
    }
}

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

async fn body_limit(
    State(limit_bytes): State<usize>,
    req: Request,
    next: axum::middleware::Next,
) -> Response {
    let oversized = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|len| len > limit_bytes);
    if oversized {
        return ApiError::PayloadTooLarge(limit_bytes / (1024 * 1024)).into_response();
    }
    next.run(req).await
}

async fn graph_coverage() -> Response {
    Json(crate::graph_stats::render()).into_response()
}

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
    req.validate_with_family(pipeline.family())?;
    Ok(Json(pipeline.choice(req).await?))
}
async fn score(
    State(pipeline): State<Arc<Pool>>,
    ValidJson(req): ValidJson<ScoreRequest>,
) -> Result<Json<ScoreResponse>, ApiError> {
    req.validate_with_family(pipeline.family())?;
    Ok(Json(pipeline.score(req).await?))
}
async fn noul(
    State(pipeline): State<Arc<Pool>>,
    ValidJson(req): ValidJson<NoulRequest>,
) -> Result<Json<NoulResponse>, ApiError> {
    req.validate_with_family(pipeline.family())?;
    Ok(Json(pipeline.noul(req).await?))
}
pub fn router(pool: Arc<Pool>, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_text))
        .route("/graph", get(graph_coverage))
        .route("/v1/choice", post(choice))
        .route("/v1/score", post(score))
        .route("/v1/noul", post(noul))
        .merge(crate::compat::routes())
        .merge(crate::systemone::routes())
        .merge(crate::decisions::routes())
        .fallback(not_found)
        .layer(axum::middleware::from_fn_with_state(
            max_body_bytes,
            body_limit,
        ))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(pool)
}
