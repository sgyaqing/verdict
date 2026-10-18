//! `/v1/systemone`: the System One category's native API -- TypeSafe Jev's
//! request/response shape, which Laya's official runtime reproduces and
//! Ollama, Runware and LiteLLM serve. One request carries a `state` plus a
//! map of typed questions; every question becomes exactly one engine task (the
//! category's "every question in one forward pass" is the batch dimension;
//! Verdict's cross-request batcher assembles the same batch, pinned
//! response-bitwise by the batch-invariance suites).
//!
//! Parity ruling (same as the OpenAI compat layer): an adapter over
//! already-graded semantics. Per-question parity is the native golden
//! contract; the correctness gate here is mapping tests (tests/systemone.rs).
//!
//! Wire shapes follow the reference Jev implementation. Two deliberate
//! deviations:
//!   - `rl_agent.act_probability` is omitted: the graph's act_probs output is
//!     intentionally unread (see engine.rs), and the reference documents the
//!     field as carrying no usable signal.
//!   - A malformed question answers 400 in Verdict's unified error envelope
//!     (the message names the question), not a bare 422: one error
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
use crate::family::Family;
pub use crate::family::marker_head::noul_pair;
use crate::metrics;
use crate::pipeline::Pool;
use crate::protocol::{ApiError, MAX_OPTIONS, MIN_OPTIONS, ValidJson};

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
/// name the offending question id (naming the problem is the contract, in
/// Verdict's envelope).
#[derive(Debug, Deserialize)]
struct RawQuestion {
    #[serde(rename = "type")]
    qtype: String,
    #[serde(default)]
    instructions: Option<Value>,
    #[serde(default)]
    criteria: Option<Value>,
}

// ------------------------------------------------- reference serialization ports

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
/// asymmetric with the Python-style state serializer above; both follow the
/// reference implementation.
pub fn instructions_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).expect("a JSON value serializes"),
    }
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
    /// Which family recipe composed (and answers) this question.
    family: Family,
}

fn question_error(qid: &str, detail: impl std::fmt::Display) -> ApiError {
    ApiError::InvalidRequest(format!("question {qid:?}: {detail}"))
}

/// The letter-readout family flattens instruction objects PYTHON-style
/// (json.dumps, not JS stringify) and accepts an absent instruction on noul
/// questions whose criteria describe a side (the fixed fallback sentence
/// then asks the question). Re-typed from the family's render_question.
fn instructions_of(family: Family, raw: Option<&Value>) -> String {
    match family {
        Family::LetterReadout => match raw {
            Some(Value::String(s)) => s.clone(),
            Some(other) => py_json_dumps(other),
            None => String::new(),
        },
        _ => instructions_string(raw.expect("checked present for marker families")),
    }
}

fn prepare(qid: &str, raw: &Value, family: Family) -> Result<Prepared, ApiError> {
    let q: RawQuestion =
        serde_json::from_value(raw.clone()).map_err(|err| question_error(qid, err))?;
    if family != Family::LetterReadout {
        // The marker families require a non-empty instruction up front.
        let instructions = instructions_string(
            q.instructions
                .as_ref()
                .ok_or_else(|| question_error(qid, "instructions is required"))?,
        );
        if instructions.trim().is_empty() {
            return Err(question_error(qid, "instructions must not be empty"));
        }
        return prepare_marker(
            qid,
            q.qtype.as_str(),
            &instructions,
            q.criteria.as_ref(),
            family,
        );
    }
    let instructions = instructions_of(family, q.instructions.as_ref());
    match q.qtype.as_str() {
        "choice" => {
            if instructions.trim().is_empty() {
                return Err(question_error(qid, "question without instructions"));
            }
            let (keys, options) = letter_choice_criteria(qid, q.criteria.as_ref())?;
            Ok(Prepared {
                qid: qid.to_string(),
                task: "choice",
                instructions,
                options,
                keys,
                legend: None,
                family,
            })
        }
        "score" => {
            if instructions.trim().is_empty() {
                return Err(question_error(qid, "question without instructions"));
            }
            let levels = letter_score_levels(qid, q.criteria.as_ref())?;
            let keys = (0..levels.len()).map(|i| i.to_string()).collect();
            Ok(Prepared {
                qid: qid.to_string(),
                task: "score",
                instructions,
                options: levels.clone(),
                keys,
                legend: Some(levels),
                family,
            })
        }
        "noul" => {
            let sides = letter_noul_sides(qid, q.criteria.as_ref())?;
            // No instruction: the descriptions must carry the question
            // (render_question's NOUL_WITHOUT_INSTRUCTIONS rule).
            let instructions = if instructions.trim().is_empty() {
                if sides.is_none() {
                    return Err(question_error(
                        qid,
                        "noul question without instructions: criteria must describe \
                         true or false",
                    ));
                }
                crate::family::letter_readout::NOUL_WITHOUT_INSTRUCTIONS.to_string()
            } else {
                instructions
            };
            let options = letter_noul_pair(sides);
            Ok(Prepared {
                qid: qid.to_string(),
                task: "noul",
                instructions,
                options,
                keys: Vec::new(),
                legend: None,
                family,
            })
        }
        other => Err(question_error(
            qid,
            format!("unknown type {other:?} (expected \"choice\", \"score\", or \"noul\")"),
        )),
    }
}

/// The marker-family preparation (v0.1 semantics; identical to the previous
/// single path).
fn prepare_marker(
    qid: &str,
    qtype: &str,
    instructions: &str,
    criteria: Option<&Value>,
    family: Family,
) -> Result<Prepared, ApiError> {
    match qtype {
        "choice" => {
            let (keys, descriptions) = choice_criteria(qid, criteria)?;
            // Criteria composition is family data: the marker-head recipe
            // renders "key: description" pairs, the independent-options
            // recipe packs the description itself (falling back to the key).
            let options = keys
                .iter()
                .zip(&descriptions)
                .map(|(key, desc)| match (family, desc) {
                    (Family::MarkerHead, Some(text)) => format!("{key}: {text}"),
                    (Family::IndependentOptions, Some(text)) => text.trim().to_string(),
                    (Family::MarkerHead, None) => key.clone(),
                    (Family::IndependentOptions, None) => key.trim().to_string(),
                    (Family::LetterReadout, _) => unreachable!("dispatched above"),
                })
                .collect();
            Ok(Prepared {
                qid: qid.to_string(),
                task: "choice",
                instructions: instructions.to_string(),
                options,
                keys,
                legend: None,
                family,
            })
        }
        "score" => {
            let levels = score_levels(qid, criteria)?;
            let keys = (0..levels.len()).map(|i| i.to_string()).collect();
            Ok(Prepared {
                qid: qid.to_string(),
                task: "score",
                instructions: instructions.to_string(),
                options: levels.clone(),
                keys,
                legend: Some(levels),
                family,
            })
        }
        "noul" => {
            let options = match family {
                Family::MarkerHead => {
                    noul_pair(criteria).map_err(|err| question_error(qid, err))?
                }
                Family::IndependentOptions => {
                    independent_noul_pair(criteria).map_err(|err| question_error(qid, err))?
                }
                Family::LetterReadout => unreachable!("dispatched above"),
            };
            Ok(Prepared {
                qid: qid.to_string(),
                task: "noul",
                instructions: instructions.to_string(),
                options,
                keys: Vec::new(),
                legend: None,
                family,
            })
        }
        other => Err(question_error(
            qid,
            format!("unknown type {other:?} (expected \"choice\", \"score\", or \"noul\")"),
        )),
    }
}

/// Letter choice criteria (render_question): a map of 2..=MAX_OPTIONS
/// options whose descriptions may be ANY JSON value (rendered "key: value"
/// Python-style; a null or empty-string description leaves the bare key), or
/// the list shorthand of bare names. -> (names, rendered option texts).
fn letter_choice_criteria(
    qid: &str,
    criteria: Option<&Value>,
) -> Result<(Vec<String>, Vec<String>), ApiError> {
    let (keys, options): (Vec<String>, Vec<String>) = match criteria {
        // List shorthand: bare option names (official: the map {name: null}).
        Some(Value::Array(items)) => {
            let names = items
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| question_error(qid, "criteria list items must be strings"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let options = names.clone();
            (names, options)
        }
        Some(Value::Object(map)) => {
            let keys: Vec<String> = map.keys().cloned().collect();
            let options = map
                .iter()
                .map(|(key, desc)| match desc {
                    Value::Null => key.clone(),
                    Value::String(s) if s.is_empty() => key.clone(),
                    Value::String(s) => format!("{key}: {s}"),
                    other => format!("{key}: {}", py_json_dumps(other)),
                })
                .collect();
            (keys, options)
        }
        _ => {
            return Err(question_error(
                qid,
                "criteria is required: an object of option -> description (or null), or a list of option names",
            ));
        }
    };
    if keys.len() < MIN_OPTIONS || keys.len() > MAX_OPTIONS {
        return Err(question_error(
            qid,
            format!(
                "criteria must name {MIN_OPTIONS}..={MAX_OPTIONS} options (got {})",
                keys.len()
            ),
        ));
    }
    if let Some(key) = keys.iter().find(|key| key.trim().is_empty()) {
        return Err(question_error(
            qid,
            format!("criteria option {key:?} must be a non-empty string"),
        ));
    }
    Ok((keys, options))
}

/// Letter score criteria (render_question): an ordered list of level
/// descriptions, or the legend map {"0": "...", "1": "..."} sorted by its
/// float keys. Descriptions may be any JSON value (Python-style rendering).
fn letter_score_levels(qid: &str, criteria: Option<&Value>) -> Result<Vec<String>, ApiError> {
    let raw: Vec<&Value> = match criteria {
        Some(Value::Array(items)) => items.iter().collect(),
        Some(Value::Object(map)) => {
            let mut keyed: Vec<(f64, &Value)> = Vec::with_capacity(map.len());
            for (key, value) in map {
                let num: f64 = key.parse().map_err(|_| {
                    question_error(qid, format!("legend key {key:?} is not a number"))
                })?;
                keyed.push((num, value));
            }
            keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("parsed floats are total"));
            keyed.into_iter().map(|(_, v)| v).collect()
        }
        _ => {
            return Err(question_error(
                qid,
                "criteria is required: a list of level descriptions, lowest first",
            ));
        }
    };
    if raw.len() < MIN_OPTIONS || raw.len() > MAX_OPTIONS {
        return Err(question_error(
            qid,
            format!(
                "criteria must name {MIN_OPTIONS}..={MAX_OPTIONS} levels (got {})",
                raw.len()
            ),
        ));
    }
    // Bug-for-bug: the official renderer caps a score legend at 10 levels
    // (MAX_LEVELS) and rejects more; the protocol's 32 bound is the NATIVE
    // endpoint's contract, not this family's.
    if raw.len() > 10 {
        return Err(question_error(
            qid,
            format!(
                "criteria must name 2..=10 level descriptions (got {}) on this family",
                raw.len()
            ),
        ));
    }
    let levels = raw
        .iter()
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => py_json_dumps(other),
        })
        .collect::<Vec<_>>();
    if levels.iter().any(|level| level.trim().is_empty()) {
        return Err(question_error(
            qid,
            "criteria level descriptions must be non-empty",
        ));
    }
    Ok(levels)
}

/// Letter noul sides (render_question): the criteria map's optional
/// "true"/"false" descriptions; None when there are no criteria at all or
/// neither side is described (the fixed no/yes pair). A non-map is rejected.
/// -> Some((true_desc, false_desc)) with each side Option.
type NoulSides = (Option<String>, Option<String>);

fn letter_noul_sides(qid: &str, criteria: Option<&Value>) -> Result<Option<NoulSides>, ApiError> {
    let sides = match criteria {
        None | Some(Value::Null) => None,
        Some(Value::Object(map)) => {
            let side = |key: &str| -> Result<Option<String>, ApiError> {
                match map.get(key) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::String(s)) if s.is_empty() => Ok(None),
                    Some(Value::String(s)) => Ok(Some(s.clone())),
                    Some(other) => Err(question_error(
                        qid,
                        format!("criteria.{key} must be a description (got {other})"),
                    )),
                }
            };
            let t = side("true")?;
            let f = side("false")?;
            (t.is_some() || f.is_some()).then_some((t, f))
        }
        Some(other) => {
            return Err(question_error(
                qid,
                format!(
                    "criteria must be an object with optional \"true\"/\"false\" descriptions (got {other})"
                ),
            ));
        }
    };
    Ok(sides)
}

/// The letter noul option pair: ["no"/"no: {false}", "yes"/"yes: {true}"]
/// (option 1 is yes, so p[1] == P(true)).
fn letter_noul_pair(sides: Option<NoulSides>) -> Vec<String> {
    match sides {
        None => crate::family::letter_readout::NOUL_DEFAULT_PAIR
            .map(|s| s.to_string())
            .to_vec(),
        Some((t, f)) => vec![
            f.map(|f| format!("no: {f}"))
                .unwrap_or_else(|| "no".to_string()),
            t.map(|t| format!("yes: {t}"))
                .unwrap_or_else(|| "yes".to_string()),
        ],
    }
}

/// The independent-options noul pair: `[true, false]` descriptions, each
/// side falling back to its default; NO criteria at all yields no options
/// (the zero-shot path -- the pipeline then runs the context-free debias
/// forward). One-sided criteria fill the other side with its default,
/// mirroring the official truthiness handling.
fn independent_noul_pair(criteria: Option<&Value>) -> Result<Vec<String>, String> {
    use crate::family::independent_options::NOUL_DEFAULT_PAIR;
    let sides: Option<(Option<String>, Option<String>)> = match criteria {
        None | Some(Value::Null) => None,
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
            Some((side("true")?, side("false")?))
        }
        Some(other) => {
            return Err(format!(
                "criteria must be an object with optional \"true\"/\"false\" descriptions (got {other})"
            ));
        }
    };
    match sides {
        None => Ok(Vec::new()),
        Some((t, f)) if t.is_none() && f.is_none() => Ok(Vec::new()),
        Some((t, f)) => Ok(vec![
            t.unwrap_or_else(|| NOUL_DEFAULT_PAIR[0].to_string()),
            f.unwrap_or_else(|| NOUL_DEFAULT_PAIR[1].to_string()),
        ]),
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
/// level indices with its legend, noul is P(true). `confidence` is each
/// family's own recipe (Jev entropy for the marker families, the official
/// margin/distance formulas for letter readout); `rl_agent` is deliberately
/// absent (module docs). The letter family's official answer extras
/// (x_p_max/certainty and the isolated-score recombination) ride along as
/// optional fields, absent for the marker families. Maps are serde_json Maps
/// (insertion-ordered under preserve_order).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Answer {
    Choice {
        choice: String,
        probabilities: Map<String, Value>,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        x_p_max: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        certainty: Option<f64>,
    },
    Score {
        score: f64,
        legend: Map<String, Value>,
        probabilities: Map<String, Value>,
        confidence: f64,
        #[serde(skip_serializing_if = "Option::is_none")]
        x_p_max: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        certainty: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        level_fit: Option<Map<String, Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        fit_mass: Option<f64>,
    },
    Noul {
        noul: f64,
        /// Calibrated P(true) before the commit band, when the family has
        /// one (the independent-options recipe); absent otherwise.
        #[serde(skip_serializing_if = "Option::is_none")]
        noul_raw: Option<f64>,
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
                x_p_max: c.x_p_max,
                certainty: c.certainty,
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
            let level_fit = c.level_fit.as_ref().map(|fit| {
                fit.iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), (*v).into()))
                    .collect::<Map<String, Value>>()
            });
            Answer::Score {
                score: c.score.unwrap_or(0.0),
                legend,
                probabilities,
                confidence: c.confidence,
                x_p_max: c.x_p_max,
                certainty: c.certainty,
                level_fit,
                fit_mass: c.fit_mass,
            }
        }
        _ => Answer::Noul {
            noul: c.probability.unwrap_or(0.0),
            noul_raw: match p.family {
                Family::MarkerHead | Family::LetterReadout => None,
                // probabilities[0] is the rounded pre-band P(true).
                Family::IndependentOptions => Some(c.probabilities.first().copied().unwrap_or(0.0)),
            },
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
    let family = pool.family();
    // State serialization is family data: the marker families keep plain
    // Python-style dumps; the letter family additionally annotates
    // long arrays with their indices (official annotate_indices).
    let state = match family {
        Family::LetterReadout => crate::family::letter_readout::serialize_state(&req.state),
        _ => serialize_state(&req.state),
    };
    // Validate everything before submitting anything: one malformed question
    // rejects the whole request without spending engine work.
    let prepared: Vec<Prepared> = req
        .questions
        .iter()
        .map(|(qid, raw)| prepare(qid, raw, family))
        .collect::<Result<_, _>>()?;

    // Fire every question concurrently: the batcher packs them into one
    // batch -- a single packed forward over every question. Each
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

    // Usage is family accounting: the marker families sum each question's
    // rendered tokens; the letter family counts the shared state prefix
    // once across the whole request (official unique_tokens over all rows).
    let input_tokens = match family {
        Family::LetterReadout => {
            let rows: Vec<Vec<u32>> = prepared
                .iter()
                .flat_map(|p| pool.build_row_ids(p.task, &p.instructions, &state, &p.options))
                .collect();
            crate::family::letter_readout::unique_tokens(
                &rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>(),
            )
        }
        _ => results.iter().map(|(_, tokens)| *tokens).sum(),
    };
    let mut answers = Map::new();
    for (p, (calibrated, _)) in prepared.iter().zip(results) {
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
