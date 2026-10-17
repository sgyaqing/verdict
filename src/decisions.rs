//! `/v1/decisions`: OpenAI's cross-ecosystem Decisions API -- the dialect
//! LiteLLM's unified decision endpoint and SGLang serve alongside
//! `/v1/systemone`. One request carries one `input` (shared evidence) plus a
//! LIST of typed questions; answers come back as a parallel array.
//!
//! Schema source of truth: the installed OpenAI SDK 3.26.0
//! (`types/decision_create_params.py` + `types/decision.py`, generated from
//! OpenAI's OpenAPI spec). Dialect notes:
//!   - `input` accepts a string or an array of user messages; Verdict
//!     flattens the message form to its text parts joined with newlines and
//!     rejects image parts (text-only decision models).
//!   - `choice` values are TYPED (string or boolean; "a string and a boolean
//!     with the same text are distinct"): the rendered option text is the
//!     value plus its optional description ("value: description", the same
//!     rendering as the Jev criteria), and the answer echoes the original
//!     JSON type.
//!   - The dialect defines a fourth answer type, `refusal` ("the host may
//!     decline one question"); Verdict never emits it -- no host-side safety
//!     layer stands between the caller and the model.
//!   - Like `/v1/systemone`, an adapter over already-graded semantics: every
//!     question is one engine task; the correctness gate is mapping tests
//!     Malformed questions answer 400 in Verdict's
//!     unified error envelope, the message naming the question (by name when
//!     present, else by index).

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::post;
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::calibration::Calibrated;
use crate::metrics;
use crate::pipeline::Pool;
use crate::protocol::{ApiError, MAX_OPTIONS, MIN_OPTIONS, ValidJson};

/// Decisions routes, merged into the native router (state applied there).
pub fn routes() -> Router<Arc<Pool>> {
    Router::new().route("/v1/decisions", post(create_decision))
}

// ---------------------------------------------------------------- request

#[derive(Debug, Deserialize)]
pub struct DecisionsRequest {
    /// Required by the dialect; ignored -- one process serves one model.
    pub model: String,
    /// Shared evidence: a string, or an array of user messages (text parts).
    pub input: Value,
    /// Questions in order; answers are a parallel array.
    pub questions: Vec<Value>,
    /// Accepted for shape compatibility, ignored.
    #[serde(default)]
    pub safety_identifier: Option<String>,
}

/// One question, kept raw through deserialization so every rejection can
/// name the offending question.
#[derive(Debug, Deserialize)]
struct RawQuestion {
    #[serde(rename = "type")]
    qtype: String,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    choices: Option<Value>,
    #[serde(default)]
    levels: Option<Value>,
}

// ---------------------------------------------------------------- input

/// Flatten the dialect's `input` to the state string: a string passes
/// through; the message form contributes its text parts, all joined with
/// newlines (our rendering choice -- the dialect does not define a join).
/// Image parts are rejected: Verdict serves text-only decision models.
pub fn flatten_input(input: &Value) -> Result<String, ApiError> {
    match input {
        Value::String(s) => Ok(s.clone()),
        Value::Array(messages) => {
            let mut texts: Vec<String> = Vec::new();
            for (i, message) in messages.iter().enumerate() {
                if message.get("role").and_then(Value::as_str) != Some("user") {
                    return Err(ApiError::InvalidRequest(format!(
                        "input[{i}]: only user messages are supported"
                    )));
                }
                match message.get("content") {
                    Some(Value::String(text)) => texts.push(text.clone()),
                    Some(Value::Array(parts)) => {
                        for part in parts {
                            match part.get("type").and_then(Value::as_str) {
                                Some("input_text") => {
                                    texts.push(
                                        part.get("text")
                                            .and_then(Value::as_str)
                                            .ok_or_else(|| {
                                                ApiError::InvalidRequest(
                                                    "input text parts must carry a text string"
                                                        .to_string(),
                                                )
                                            })?
                                            .to_string(),
                                    );
                                }
                                Some("input_image") => {
                                    return Err(ApiError::InvalidRequest(
                                        "input image parts are not supported: Verdict serves \
                                         text-only decision models"
                                            .to_string(),
                                    ));
                                }
                                other => {
                                    return Err(ApiError::InvalidRequest(format!(
                                        "input[{i}]: unknown part type {other:?}"
                                    )));
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(ApiError::InvalidRequest(format!(
                            "input[{i}]: content must be a string or an array of parts"
                        )));
                    }
                }
            }
            Ok(texts.join("\n"))
        }
        other => Err(ApiError::InvalidRequest(format!(
            "input must be a string or an array of user messages (got {other})"
        ))),
    }
}

/// The rendered option text of a typed choice value: booleans spell out,
/// strings pass through (the answer echoes the original type, not this).
pub fn choice_value_text(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(ApiError::InvalidRequest(format!(
            "choice values must be strings or booleans (got {other})"
        ))),
    }
}

// ------------------------------------------------------------- preparation

/// A validated question, ready to become one engine task.
struct Prepared {
    name: Option<String>,
    task: &'static str,
    instructions: String,
    /// Rendered option texts in marker order (empty for predicate).
    options: Vec<String>,
    /// choice only: the TYPED values, echoed verbatim in the answer.
    values: Vec<Value>,
    /// score only: the level labels, echoed in the answer probabilities.
    labels: Vec<String>,
}

fn question_error(pre: &Prepared, index: usize, detail: impl std::fmt::Display) -> ApiError {
    match &pre.name {
        Some(name) => ApiError::InvalidRequest(format!("question {name:?}: {detail}")),
        None => ApiError::InvalidRequest(format!("question {index}: {detail}")),
    }
}

/// A name-bearing error before the question body itself parses.
fn raw_question_error(
    name: Option<&str>,
    index: usize,
    detail: impl std::fmt::Display,
) -> ApiError {
    match name {
        Some(name) => ApiError::InvalidRequest(format!("question {name:?}: {detail}")),
        None => ApiError::InvalidRequest(format!("question {index}: {detail}")),
    }
}

fn prepare(index: usize, raw: &Value) -> Result<Prepared, ApiError> {
    // Fish out name first so even parse failures can name the question.
    let name = raw.get("name").and_then(Value::as_str).map(str::to_string);
    let q: RawQuestion = serde_json::from_value(raw.clone())
        .map_err(|err| raw_question_error(name.as_deref(), index, err))?;
    let mut pre = Prepared {
        name,
        task: "",
        instructions: String::new(),
        options: Vec::new(),
        values: Vec::new(),
        labels: Vec::new(),
    };
    pre.instructions = q
        .instructions
        .clone()
        .ok_or_else(|| raw_question_error(q.name.as_deref(), index, "instructions is required"))?;
    if pre.instructions.trim().is_empty() {
        return Err(raw_question_error(
            q.name.as_deref(),
            index,
            "instructions must not be empty",
        ));
    }
    match q.qtype.as_str() {
        "predicate" => {
            // No criteria in this dialect: the engine's fixed noul pair.
            pre.task = "noul";
        }
        "choice" => {
            pre.task = "choice";
            let choices = q.choices.as_ref().ok_or_else(|| {
                raw_question_error(q.name.as_deref(), index, "choices is required")
            })?;
            let items = choices.as_array().ok_or_else(|| {
                raw_question_error(q.name.as_deref(), index, "choices must be an array")
            })?;
            check_count(&pre, index, items.len(), "choices")?;
            for item in items {
                let value = item.get("value").cloned().ok_or_else(|| {
                    raw_question_error(q.name.as_deref(), index, "each choice needs a value")
                })?;
                let text = choice_value_text(&value)?;
                let description = item.get("description").and_then(Value::as_str);
                pre.options.push(match description {
                    Some(desc) => format!("{text}: {desc}"),
                    None => text,
                });
                pre.values.push(value);
            }
        }
        "score" => {
            pre.task = "score";
            let levels = q.levels.as_ref().ok_or_else(|| {
                raw_question_error(q.name.as_deref(), index, "levels is required")
            })?;
            let items = levels.as_array().ok_or_else(|| {
                raw_question_error(q.name.as_deref(), index, "levels must be an array")
            })?;
            check_count(&pre, index, items.len(), "levels")?;
            for item in items {
                let label = item
                    .get("label")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        raw_question_error(q.name.as_deref(), index, "each level needs a label")
                    })?
                    .to_string();
                if label.trim().is_empty() {
                    return Err(raw_question_error(
                        q.name.as_deref(),
                        index,
                        "level labels must be non-empty",
                    ));
                }
                let description = item.get("description").and_then(Value::as_str);
                pre.options.push(match description {
                    Some(desc) => format!("{label}: {desc}"),
                    None => label.clone(),
                });
                pre.labels.push(label);
            }
        }
        other => {
            return Err(raw_question_error(
                q.name.as_deref(),
                index,
                format!(
                    "unknown type {other:?} (expected \"predicate\", \"choice\", or \"score\")"
                ),
            ));
        }
    }
    Ok(pre)
}

fn check_count(pre: &Prepared, index: usize, count: usize, what: &str) -> Result<(), ApiError> {
    if !(MIN_OPTIONS..=MAX_OPTIONS).contains(&count) {
        return Err(question_error(
            pre,
            index,
            format!("{what} must name {MIN_OPTIONS}..={MAX_OPTIONS} options (got {count})"),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------- response

#[derive(Debug, Serialize)]
struct UsageInputDetails {
    cache_write_tokens: u64,
    cached_tokens: u64,
}

#[derive(Debug, Serialize)]
struct UsageOutputDetails {
    reasoning_tokens: u64,
}

#[derive(Debug, Serialize)]
struct DecisionsUsage {
    input_tokens: u64,
    input_tokens_details: UsageInputDetails,
    output_tokens: u64,
    output_tokens_details: UsageOutputDetails,
    total_tokens: u64,
}

#[derive(Debug, Serialize)]
struct ChoiceProbability {
    value: Value,
    probability: f64,
}

#[derive(Debug, Serialize)]
struct ScoreProbability {
    label: String,
    value: u64,
    probability: f64,
}

/// One typed answer (SDK decision.py): predicate is a probability, choice
/// echoes the TYPED value with per-value probabilities, score carries the
/// expected level over label-keyed probabilities. `name` echoes the question
/// when it had one; `refusal` is deliberately never emitted (module docs).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Answer {
    Predicate {
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        probability: f64,
    },
    Choice {
        choice: Value,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        probabilities: Vec<ChoiceProbability>,
    },
    Score {
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        probabilities: Vec<ScoreProbability>,
        score: f64,
    },
}

fn answer(p: &Prepared, c: &Calibrated) -> Answer {
    match p.task {
        "choice" => {
            let best = c.choice.unwrap_or(0);
            Answer::Choice {
                choice: p.values.get(best).cloned().unwrap_or(Value::Null),
                confidence: c.confidence,
                name: p.name.clone(),
                probabilities: p
                    .values
                    .iter()
                    .enumerate()
                    .map(|(i, value)| ChoiceProbability {
                        value: value.clone(),
                        probability: c.probabilities.get(i).copied().unwrap_or(0.0),
                    })
                    .collect(),
            }
        }
        "score" => Answer::Score {
            confidence: c.confidence,
            name: p.name.clone(),
            probabilities: p
                .labels
                .iter()
                .enumerate()
                .map(|(i, label)| ScoreProbability {
                    label: label.clone(),
                    value: i as u64,
                    probability: c.probabilities.get(i).copied().unwrap_or(0.0),
                })
                .collect(),
            score: c.score.unwrap_or(0.0),
        },
        _ => Answer::Predicate {
            name: p.name.clone(),
            probability: c.probability.unwrap_or(0.0),
        },
    }
}

// ---------------------------------------------------------------- handler

#[tracing::instrument(skip_all, fields(task = "decisions", questions = req.questions.len()))]
async fn create_decision(
    State(pool): State<Arc<Pool>>,
    ValidJson(req): ValidJson<DecisionsRequest>,
) -> Result<Json<DecisionsResponse>, ApiError> {
    if req.questions.is_empty() {
        return Err(ApiError::InvalidRequest(
            "questions must contain at least one question".to_string(),
        ));
    }
    let state = flatten_input(&req.input)?;
    // Validate everything before submitting anything: one malformed question
    // rejects the whole request without spending engine work.
    let prepared: Vec<Prepared> = req
        .questions
        .iter()
        .enumerate()
        .map(|(index, raw)| prepare(index, raw))
        .collect::<Result<_, _>>()?;

    // Fire every question concurrently: the batcher packs them into one
    // batch, exactly like /v1/systemone. Each question records its own
    // decision metric (one decision = one record, wherever it came from).
    let results = try_join_all(prepared.iter().map(|p| async {
        let started = Instant::now();
        let result = pool
            .run_task(p.task, &p.instructions, &state, &p.options, 0)
            .await;
        if result.is_ok() {
            metrics::record_decision(p.task, started.elapsed().as_millis() as u64);
        }
        result
    }))
    .await?;

    let mut answers = Vec::with_capacity(prepared.len());
    let mut input_tokens = 0u64;
    for (p, (calibrated, tokens)) in prepared.iter().zip(results) {
        input_tokens += tokens;
        answers.push(answer(p, &calibrated));
    }
    Ok(Json(DecisionsResponse {
        model: pool.served_model().to_string(),
        answers,
        usage: DecisionsUsage {
            input_tokens,
            input_tokens_details: UsageInputDetails {
                cache_write_tokens: 0,
                cached_tokens: 0,
            },
            output_tokens: 0,
            output_tokens_details: UsageOutputDetails {
                reasoning_tokens: 0,
            },
            total_tokens: input_tokens,
        },
    }))
}

#[derive(Debug, Serialize)]
struct DecisionsResponse {
    model: String,
    answers: Vec<Answer>,
    usage: DecisionsUsage,
}
