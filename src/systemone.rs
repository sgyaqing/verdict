//! `/v1/systemone`: the System One category's native API -- TypeSafe Jev's
//! request/response shape, which Laya's official runtime reproduces and
//! Ollama 0.35 / Runware / LiteLLM serve. One request carries a `state` plus
//! a map of typed questions; every question becomes exactly one engine task
//! (the vendor's "every question in one forward pass" is the batch dimension;
//! Verdict's cross-request batcher assembles the same batch, pinned
//! response-bitwise by the batch-invariance suites, both families).
//!
//! Parity ruling (same as the OpenAI compat layer): an adapter over
//! already-graded semantics. Per-question parity is the native golden
//! contract; the correctness gate here is the request/response mapping.
//!
//! Wire shapes are re-typed from laya-export/src/{types,sequence,laya}.ts,
//! the vendor's own Jev port. Two deliberate deviations:
//!   - `rl_agent.act_probability` is omitted: the graph's act_probs output is
//!     intentionally unread (see engine.rs) and the vendor documents
//!     the field as carrying no usable signal (~1.0 for almost every input,
//!     AUROC 0.30 against correctness).
//!   - A malformed question answers 400 in Verdict's unified error envelope
//!     (the message names the question), not laya-serve's bare 422: one error
//!     shape per protocol family is the product contract.

use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::post;
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::calibration::Calibrated;
use crate::metrics;
use crate::pipeline::Pool;
use crate::protocol::{ApiError, MAX_OPTIONS, MIN_OPTIONS, ValidJson};
use crate::tokenize::NOUL_DEFAULT_DESCRIPTIONS;

/// Systemone routes, merged into the native router (state applied there).
pub fn routes() -> Router<Arc<Pool>> {
    Router::new().route("/v1/systemone", post(system_one))
}

// ---------------------------------------------------------------- request

#[derive(Debug, Deserialize)]
pub struct SystemOneRequest {
    /// Accepted for shape compatibility (Jev clients send it); ignored
    /// (one process serves one model).
    #[serde(default)]
    pub model: Option<Value>,
    /// The state every question is asked about: a string passes through,
    /// anything else is serialized the way the reference path does.
    pub state: Value,
    /// Question id -> question, insertion-ordered (answers echo this order).
    pub questions: Map<String, Value>,
}

/// One question, kept raw through deserialization so every rejection can
/// name the offending question id (the vendor's "naming the problem"
/// promise, in Verdict's envelope).
#[derive(Debug, Deserialize)]
struct RawQuestion {
    #[serde(rename = "type")]
    qtype: String,
    #[serde(default)]
    instructions: Option<Value>,
    #[serde(default)]
    criteria: Option<Value>,
}

// ------------------------------------------------- vendor serialization ports

/// The reference state serializer (rl_common.serialize_state / sequence.ts):
/// a string passes through; anything else is dumped Python-style -- ", " and
/// ": " separators, insertion key order, non-ASCII raw (ensure_ascii=False).
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => py_json_dumps(other),
    }
}

/// Python `json.dumps(v, ensure_ascii=False)` over an already-parsed value
/// (sequence.ts pyJsonDumps, re-typed; the unit tests pin the wire format).
pub fn py_json_dumps(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                // Floats ride the JSON shortest round-trip; exotic JS
                // spellings (1e+21) are out of scope for real states.
                n.to_string()
            }
        }
        Value::String(s) => serde_json::to_string(s).expect("a JSON string serializes"),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_json_dumps).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(key).expect("a JSON string serializes"),
                        py_json_dumps(value)
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Instruction flattening (toInternal): a string passes through, an object is
/// JSON-stringified COMPACT (JS JSON.stringify, no spaces) -- deliberately
/// asymmetric with the Python-style state serializer above; both re-typed
/// from the vendor source.
pub fn instructions_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).expect("a JSON value serializes"),
    }
}

/// The noul option pair. No criteria (or an empty object) yields no options,
/// letting build_sequence render its default pair; the Jev criteria
/// `{"true": ..., "false": ...}` yield the two sentences, each side falling
/// back to its default when absent or empty (the vendor's JS-truthy `||`).
pub fn noul_pair(criteria: Option<&Value>) -> Result<Vec<String>, String> {
    let (t, f) = match criteria {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Object(map)) => {
            let side = |key: &str| -> Result<Option<String>, String> {
                match map.get(key) {
                    None | Some(Value::Null) => Ok(None),
                    // JS truthiness: an empty string falls back to the default.
                    Some(Value::String(s)) if !s.is_empty() => Ok(Some(s.clone())),
                    Some(Value::String(_)) => Ok(None),
                    Some(other) => Err(format!(
                        "criteria.{key} must be a description string (got {other})"
                    )),
                }
            };
            let t = side("true")?;
            let f = side("false")?;
            if t.is_none() && f.is_none() {
                return Ok(Vec::new());
            }
            (t, f)
        }
        Some(other) => {
            return Err(format!(
                "criteria must be an object with optional \"true\"/\"false\" descriptions (got {other})"
            ));
        }
    };
    Ok(vec![
        format!(
            "false: {}",
            f.unwrap_or_else(|| NOUL_DEFAULT_DESCRIPTIONS[0].to_string())
        ),
        format!(
            "true: {}",
            t.unwrap_or_else(|| NOUL_DEFAULT_DESCRIPTIONS[1].to_string())
        ),
    ])
}

// ------------------------------------------------------------- preparation

/// A validated question, ready to become one engine task.
struct Prepared {
    qid: String,
    task: &'static str,
    instructions: String,
    /// Rendered option texts in marker order (empty = server-rendered noul).
    options: Vec<String>,
    /// choice: option keys (the answer vocabulary). score: level indices.
    keys: Vec<String>,
    /// score only: the level texts, for the legend.
    legend: Option<Vec<String>>,
}

fn question_error(qid: &str, detail: impl std::fmt::Display) -> ApiError {
    ApiError::InvalidRequest(format!("question {qid:?}: {detail}"))
}

fn prepare(qid: &str, raw: &Value) -> Result<Prepared, ApiError> {
    let q: RawQuestion =
        serde_json::from_value(raw.clone()).map_err(|err| question_error(qid, err))?;
    let instructions =
        instructions_string(q.instructions.as_ref().ok_or_else(|| {
            question_error(qid, "instructions is required (a string or an object)")
        })?);
    if instructions.trim().is_empty() {
        return Err(question_error(qid, "instructions must not be empty"));
    }
    match q.qtype.as_str() {
        "choice" => {
            let (keys, descriptions) = choice_criteria(qid, q.criteria.as_ref())?;
            let options = keys
                .iter()
                .zip(&descriptions)
                .map(|(key, desc)| match desc {
                    Some(text) => format!("{key}: {text}"),
                    None => key.clone(),
                })
                .collect();
            Ok(Prepared {
                qid: qid.to_string(),
                task: "choice",
                instructions,
                options,
                keys,
                legend: None,
            })
        }
        "score" => {
            let levels = score_levels(qid, q.criteria.as_ref())?;
            let keys = (0..levels.len()).map(|i| i.to_string()).collect();
            Ok(Prepared {
                qid: qid.to_string(),
                task: "score",
                instructions,
                options: levels.clone(),
                keys,
                legend: Some(levels),
            })
        }
        "noul" => {
            let options = noul_pair(q.criteria.as_ref()).map_err(|err| question_error(qid, err))?;
            Ok(Prepared {
                qid: qid.to_string(),
                task: "noul",
                instructions,
                options,
                keys: Vec::new(),
                legend: None,
            })
        }
        other => Err(question_error(
            qid,
            format!("unknown type {other:?} (expected \"choice\", \"score\", or \"noul\")"),
        )),
    }
}

/// choice criteria: an object (option -> description | null, insertion
/// order) or the Jev list shorthand (bare option names).
fn choice_criteria(
    qid: &str,
    criteria: Option<&Value>,
) -> Result<(Vec<String>, Vec<Option<String>>), ApiError> {
    let pairs: Vec<(String, Option<String>)> = match criteria {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| (s.to_string(), None))
                    .ok_or_else(|| question_error(qid, "criteria list items must be strings"))
            })
            .collect::<Result<_, _>>()?,
        Some(Value::Object(map)) => map
            .iter()
            .map(|(key, value)| match value {
                Value::Null => Ok((key.clone(), None)),
                Value::String(desc) => Ok((key.clone(), Some(desc.clone()))),
                other => Err(question_error(
                    qid,
                    format!("criteria[{key:?}] must be a description string or null (got {other})"),
                )),
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(question_error(
                qid,
                "criteria is required: an object of option -> description (or null), or a list of option names",
            ));
        }
    };
    if pairs.len() < MIN_OPTIONS || pairs.len() > MAX_OPTIONS {
        return Err(question_error(
            qid,
            format!(
                "criteria must name {MIN_OPTIONS}..={MAX_OPTIONS} options (got {})",
                pairs.len()
            ),
        ));
    }
    if let Some((key, _)) = pairs.iter().find(|(key, _)| key.trim().is_empty()) {
        return Err(question_error(
            qid,
            format!("criteria option {key:?} must be a non-empty string"),
        ));
    }
    Ok(pairs.into_iter().unzip())
}

fn score_levels(qid: &str, criteria: Option<&Value>) -> Result<Vec<String>, ApiError> {
    let levels: Vec<String> = match criteria {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| question_error(qid, "criteria levels must be strings"))
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(question_error(
                qid,
                "criteria is required: a list of level descriptions, lowest first",
            ));
        }
    };
    if levels.len() < MIN_OPTIONS || levels.len() > MAX_OPTIONS {
        return Err(question_error(
            qid,
            format!(
                "criteria must name {MIN_OPTIONS}..={MAX_OPTIONS} levels (got {})",
                levels.len()
            ),
        ));
    }
    if levels.iter().any(|level| level.trim().is_empty()) {
        return Err(question_error(
            qid,
            "criteria level descriptions must be non-empty",
        ));
    }
    Ok(levels)
}

// ---------------------------------------------------------------- response

#[derive(Debug, Serialize)]
struct SystemOneUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Serialize)]
struct SystemOneResponse {
    model: String,
    answers: Map<String, Value>,
    usage: SystemOneUsage,
}

/// One typed answer (types.ts Answer): choice speaks in option keys, score in
/// level indices with its legend, noul is P(true). `confidence` is the Jev
/// 1-minus-normalized-entropy; `rl_agent` is deliberately absent (module
/// docs). Maps are serde_json Maps (insertion-ordered under preserve_order).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Answer {
    Choice {
        choice: String,
        probabilities: Map<String, Value>,
        confidence: f64,
    },
    Score {
        score: f64,
        legend: Map<String, Value>,
        probabilities: Map<String, Value>,
        confidence: f64,
    },
    Noul {
        noul: f64,
    },
}

fn answer(p: &Prepared, c: &Calibrated) -> Answer {
    match p.task {
        "choice" => {
            let mut probabilities = Map::new();
            for (i, key) in p.keys.iter().enumerate() {
                probabilities.insert(
                    key.clone(),
                    c.probabilities.get(i).copied().unwrap_or(0.0).into(),
                );
            }
            Answer::Choice {
                choice: p
                    .keys
                    .get(c.choice.unwrap_or(0))
                    .cloned()
                    .unwrap_or_default(),
                probabilities,
                confidence: c.confidence,
            }
        }
        "score" => {
            let mut legend = Map::new();
            let mut probabilities = Map::new();
            if let Some(levels) = &p.legend {
                for (i, text) in levels.iter().enumerate() {
                    legend.insert(i.to_string(), text.clone().into());
                    probabilities.insert(
                        i.to_string(),
                        c.probabilities.get(i).copied().unwrap_or(0.0).into(),
                    );
                }
            }
            Answer::Score {
                score: c.score.unwrap_or(0.0),
                legend,
                probabilities,
                confidence: c.confidence,
            }
        }
        _ => Answer::Noul {
            noul: c.probability.unwrap_or(0.0),
        },
    }
}

// ---------------------------------------------------------------- handler

#[tracing::instrument(skip_all, fields(task = "systemone", questions = req.questions.len()))]
async fn system_one(
    State(pool): State<Arc<Pool>>,
    ValidJson(req): ValidJson<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, ApiError> {
    if req.questions.is_empty() {
        return Err(ApiError::InvalidRequest(
            "questions must contain at least one question".to_string(),
        ));
    }
    let state = serialize_state(&req.state);
    // Validate everything before submitting anything: one malformed question
    // rejects the whole request without spending engine work.
    let prepared: Vec<Prepared> = req
        .questions
        .iter()
        .map(|(qid, raw)| prepare(qid, raw))
        .collect::<Result<_, _>>()?;

    // Fire every question concurrently: the batcher packs them into one
    // batch -- the same batch the vendor's packed forward would run. Each
    // question records its own decision metric (one decision = one record,
    // wherever the decision came from).
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

    let mut answers = Map::new();
    let mut input_tokens = 0u64;
    for (p, (calibrated, tokens)) in prepared.iter().zip(results) {
        input_tokens += tokens;
        answers.insert(
            p.qid.clone(),
            serde_json::to_value(answer(p, &calibrated)).expect("an answer serializes"),
        );
    }
    Ok(Json(SystemOneResponse {
        model: pool.served_model().to_string(),
        answers,
        usage: SystemOneUsage {
            input_tokens,
            output_tokens: 0,
        },
    }))
}
