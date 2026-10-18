//! OpenAI-compatible surface: the vLLM/OpenAI facade over the native
//! protocol, whose endpoints stay the golden-graded contract. This layer
//! adapts ALREADY-GRADED semantics: its correctness gate is mapping tests
//! (replay engine + per-field shape assertions), not golden vectors.
//!
//! Surface rulings: decision fields ride as top-level request extensions
//! (`state`/`options`/`levels`/`claim`; SDK users pass them via `extra_body`);
//! the chat `content` is the human-readable ANSWER text (drop-in for vLLM
//! structured-choice clients) while the `verdict` extension carries the full
//! native payload; usage is prompt = total, completion = 0 (vLLM pooling
//! behavior); `structured_outputs: {"choice": [...]}`
//! is accepted as an `options` alias.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use crate::pipeline::Pool;

// ---------------------------------------------------------------- errors

/// OpenAI-shaped error envelope for the compat surface (the native protocol
/// keeps its own `{"error":{type,message}}` shape: two protocol families,
/// two envelopes).
#[derive(Debug)]
pub struct CompatError {
    status: StatusCode,
    err_type: &'static str,
    message: String,
    code: Option<&'static str>,
}

impl CompatError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            err_type: "invalid_request_error",
            message: message.into(),
            code: None,
        }
    }

    /// OpenAI model-not-found: 404 with code `model_not_found`
    /// (vLLM/openai clients match on the code).
    pub fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            err_type: "invalid_request_error",
            message: format!("The model `{model}` does not exist."),
            code: Some("model_not_found"),
        }
    }

    pub fn not_found() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            err_type: "not_found_error",
            message: "Unknown request URL.".to_string(),
            code: None,
        }
    }

    /// Internal errors never leak internals to the wire.
    pub fn internal() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            err_type: "server_error",
            message: "internal error".to_string(),
            code: None,
        }
    }

    pub fn overloaded() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            err_type: "server_error",
            message: "server overloaded, try again later".to_string(),
            code: None,
        }
    }
}

impl IntoResponse for CompatError {
    fn into_response(self) -> Response {
        let body = json!({"error": {
            "message": self.message,
            "type": self.err_type,
            "param": null,
            "code": self.code,
        }});
        (self.status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------- /v1/models

/// Process start as the OpenAI `created` field on model cards (there is no
/// per-model timestamp: the server loads whatever the model dir holds).
fn created_unix() -> u64 {
    static CREATED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CREATED.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    })
}

fn model_card(pipeline: &Pool) -> Value {
    json!({
        "id": pipeline.served_model(),
        "object": "model",
        "created": created_unix(),
        "owned_by": "verdict",
    })
}

async fn list_models(State(pipeline): State<Arc<Pool>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [model_card(&pipeline)]}))
}

async fn retrieve_model(
    State(pipeline): State<Arc<Pool>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, CompatError> {
    if id == pipeline.served_model() {
        Ok(Json(model_card(&pipeline)))
    } else {
        Err(CompatError::model_not_found(&id))
    }
}

// ---------------------------------------------------------------- wiring

/// Compat routes, merged into the native router (state applied there).
pub fn routes() -> Router<Arc<Pool>> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(retrieve_model))
        .route("/v1/chat/completions", post(chat_completions))
}

// ------------------------------------------------- chat streaming

#[derive(Debug, serde::Serialize)]
struct Delta {
    role: Option<&'static str>,
    content: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct ChunkChoice {
    index: u8,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, serde::Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<ChatUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verdict: Option<Value>,
}

/// Single-chunk streaming: the decision completes, then ONE delta chunk
/// (role + full content + finish stop, verdict attached) streams out -- no
/// fake token-by-token carving. With stream_options.include_usage, an
/// OpenAI usage chunk (empty choices) follows, then [DONE].
#[allow(clippy::too_many_arguments)]
fn sse_response(
    id: String,
    created: u64,
    model: String,
    answer: String,
    verdict: Value,
    usage: Option<ChatUsage>,
) -> Response {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use futures_util::StreamExt;
    use futures_util::stream;

    let chunk = |choices, usage, verdict: Option<Value>| {
        Event::default().data(
            serde_json::to_string(&ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices,
                usage,
                verdict,
            })
            .expect("chunk serializes"),
        )
    };
    let mut events = vec![chunk(
        vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: Some(answer),
            },
            finish_reason: Some("stop"),
        }],
        None,
        Some(verdict),
    )];
    if let Some(usage) = usage {
        events.push(chunk(Vec::new(), Some(usage), None));
    }
    events.push(Event::default().data("[DONE]"));
    Sse::new(stream::iter(events).map(Ok::<_, std::convert::Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ------------------------------------------------- /v1/chat/completions

/// An inbound chat message. Only `role` and string `content` are meaningful
/// for decision tasks; other OpenAI message fields (names, tool calls,
/// multimodal parts) are ignored by serde, and non-string `content` fails
/// extraction with an OpenAI-shaped 400.
#[derive(Debug, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// `structured_outputs: {"choice": [...]}` -- vLLM's guided-decoding shape,
/// accepted as an `options` alias (zero request changes for vLLM clients).
#[derive(Debug, serde::Deserialize)]
pub struct StructuredOutputs {
    pub choice: Option<Vec<String>>,
}

/// Chat request. Decision fields ride as top-level extensions (SDK users
/// pass them via `extra_body`); sampling parameters (temperature, ...) are
/// accepted and ignored -- they have no meaning for a single-forward
/// decision. Task selection is by field presence:
/// `options`/`structured_outputs.choice` -> choice, `levels` -> score,
/// `claim` -> noul; exactly one must be present.
#[derive(Debug, serde::Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub options: Option<Vec<String>>,
    #[serde(default)]
    pub levels: Option<Vec<String>>,
    #[serde(default)]
    pub claim: Option<String>,
    #[serde(default)]
    pub structured_outputs: Option<StructuredOutputs>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Scheduling priority, vLLM-aligned: smaller = admitted earlier.
    #[serde(default)]
    pub priority: Option<i64>,
}

/// OpenAI `stream_options` (only `include_usage` exists today).
#[derive(Debug, serde::Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ChatUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Debug, serde::Serialize)]
struct ChatMessageOut {
    role: &'static str,
    content: String,
}

#[derive(Debug, serde::Serialize)]
struct ChatChoice {
    index: u8,
    message: ChatMessageOut,
    finish_reason: &'static str,
}

#[derive(Debug, serde::Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
    /// Full native decision payload (same source as `content`).
    verdict: Value,
}

/// Process-unique completion id without a uuid dependency: nanosecond
/// timestamp + monotonic counter.
fn completion_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("chatcmpl-{t:x}-{n:x}")
}

fn created_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// JSON extractor whose rejections carry the OpenAI error envelope (the
/// native `ValidJson` speaks the native one).
struct CompatJson<T>(T);

impl<S, T> axum::extract::FromRequest<S> for CompatJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = CompatError;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(v)) => Ok(CompatJson(v)),
            Err(rej) => Err(CompatError::invalid_request(rej.body_text())),
        }
    }
}

/// The single task discriminator, from field presence.
enum Task {
    Choice(Vec<String>),
    Score(Vec<String>),
    Noul(String),
}

fn discriminate(req: &ChatCompletionRequest) -> Result<Task, CompatError> {
    let alias = req
        .structured_outputs
        .as_ref()
        .and_then(|s| s.choice.clone());
    if req.options.is_some() && alias.is_some() {
        return Err(CompatError::invalid_request(
            "'options' and 'structured_outputs.choice' are two spellings of the same field; provide one",
        ));
    }
    let options = req.options.clone().or(alias);
    let fields = [options.is_some(), req.levels.is_some(), req.claim.is_some()];
    match fields.iter().filter(|b| **b).count() {
        1 => Ok(match (options, &req.levels, &req.claim) {
            (Some(options), _, _) => Task::Choice(options),
            (None, Some(levels), _) => Task::Score(levels.clone()),
            (None, None, Some(claim)) => Task::Noul(claim.clone()),
            _ => unreachable!("count checked above"),
        }),
        0 => Err(CompatError::invalid_request(
            "one of 'options', 'levels', 'claim' must be provided (via extra_body from OpenAI SDK clients) to select the task type",
        )),
        _ => Err(CompatError::invalid_request(
            "at most one of 'options', 'levels', 'claim' may be provided",
        )),
    }
}

/// The question text: the LAST user message (earlier turns are not
/// consumed).
fn last_user_content(req: &ChatCompletionRequest) -> Result<&str, CompatError> {
    req.messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .ok_or_else(|| CompatError::invalid_request("no user message found in 'messages'"))
        .and_then(|c| {
            if c.trim().is_empty() {
                Err(CompatError::invalid_request(
                    "the user message content must be a non-empty string",
                ))
            } else {
                Ok(c)
            }
        })
}

/// Map native pipeline failures onto the OpenAI envelope.
fn map_api_error(err: crate::protocol::ApiError) -> CompatError {
    match err {
        crate::protocol::ApiError::Overloaded => CompatError::overloaded(),
        crate::protocol::ApiError::InvalidRequest(msg) => CompatError::invalid_request(msg),
        other => {
            tracing::error!(%other, "chat completion failed");
            CompatError::internal()
        }
    }
}

async fn chat_completions(
    State(pipeline): State<Arc<Pool>>,
    CompatJson(req): CompatJson<ChatCompletionRequest>,
) -> Result<Response, CompatError> {
    if req.model != pipeline.served_model() {
        return Err(CompatError::model_not_found(&req.model));
    }
    // The decision runs BEFORE any stream starts, so request errors surface
    // as normal OpenAI JSON errors -- nothing fakes success mid-stream.
    let (answer, verdict, input_tokens) = run_decision(&pipeline, &req).await?;
    let usage = ChatUsage {
        prompt_tokens: input_tokens,
        completion_tokens: 0,
        total_tokens: input_tokens,
    };
    if req.stream {
        let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
        return Ok(sse_response(
            completion_id(),
            created_now(),
            pipeline.served_model().to_string(),
            answer,
            verdict,
            include_usage.then_some(usage),
        ));
    }
    Ok(Json(ChatCompletionResponse {
        id: completion_id(),
        object: "chat.completion",
        created: created_now(),
        model: pipeline.served_model().to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant",
                content: answer,
            },
            finish_reason: "stop",
        }],
        usage,
        verdict,
    })
    .into_response())
}

/// Shared decision core for the streaming and non-streaming shapes: content
/// and verdict share one source (the native response).
async fn run_decision(
    pipeline: &Pool,
    req: &ChatCompletionRequest,
) -> Result<(String, Value, u64), CompatError> {
    let state = req.state.clone().unwrap_or_default();
    match discriminate(req)? {
        Task::Choice(options) => {
            let question = last_user_content(req)?.to_owned();
            let native_req = crate::protocol::ChoiceRequest {
                question,
                state,
                options: options.clone(),
                priority: req.priority,
            };
            // Same validation the native handler runs -- the adapter reuses
            // native semantics, it never re-derives them.
            native_req.validate().map_err(map_api_error)?;
            let res = pipeline.choice(native_req).await.map_err(map_api_error)?;
            let answer = options
                .get(res.choice)
                .cloned()
                .expect("native layer validates the choice index");
            Ok((
                answer,
                json!({"choice": res.choice, "probabilities": res.probabilities}),
                res.usage.input_tokens,
            ))
        }
        Task::Score(levels) => {
            let question = last_user_content(req)?.to_owned();
            let native_req = crate::protocol::ScoreRequest {
                question,
                state,
                levels: levels.clone(),
                priority: req.priority,
            };
            native_req.validate().map_err(map_api_error)?;
            let res = pipeline.score(native_req).await.map_err(map_api_error)?;
            let top = res
                .probabilities
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .expect("native layer returns one probability per level");
            Ok((
                levels[top].clone(),
                json!({"score": res.score, "probabilities": res.probabilities}),
                res.usage.input_tokens,
            ))
        }
        Task::Noul(claim) => {
            // Statement source: the explicit `claim` field wins; an empty
            // claim falls back to the user message content.
            let claim = if claim.trim().is_empty() {
                last_user_content(req)?.to_owned()
            } else {
                claim
            };
            let native_req = crate::protocol::NoulRequest {
                claim,
                state,
                priority: req.priority,
            };
            native_req.validate().map_err(map_api_error)?;
            let res = pipeline.noul(native_req).await.map_err(map_api_error)?;
            let answer = if res.probability >= 0.5 {
                "true"
            } else {
                "false"
            };
            Ok((
                answer.to_string(),
                json!({"probability": res.probability}),
                res.usage.input_tokens,
            ))
        }
    }
}
