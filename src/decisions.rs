use crate::calibration::Calibrated;
use crate::metrics;
use crate::pipeline::Pool;
use crate::protocol::{ApiError, MAX_OPTIONS, MIN_OPTIONS, ValidJson};
use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::post;
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;

pub fn routes() -> Router<Arc<Pool>> {
    Router::new().route("/v1/decisions", post(create_decision))
}
#[derive(Debug, Deserialize)]
pub struct DecisionsRequest {
    pub model: String,

    pub input: Value,

    pub questions: Vec<Value>,

    #[serde(default)]
    pub safety_identifier: Option<String>,
}

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

pub fn flatten_input(
    input: &Value,
    media_capable: bool,
) -> Result<(String, Vec<crate::tokenize::MediaItem>), ApiError> {
    let mut media = Vec::new();
    let mut segments: Vec<String> = Vec::new();
    let mut image_part = |url: &str| -> Result<String, ApiError> {
        if !media_capable {
            return Err(ApiError::InvalidRequest(
                "this model does not accept image input (no processor in the package)".to_string(),
            ));
        }
        let data = parse_data_uri(url)?;
        media.push(data);
        Ok(crate::family::letter_branches::VISION_BLOCK.to_string())
    };
    match input {
        Value::String(s) => segments.push(s.clone()),
        Value::Array(messages) => {
            for (i, message) in messages.iter().enumerate() {
                if message.get("role").and_then(Value::as_str) != Some("user") {
                    return Err(ApiError::InvalidRequest(format!(
                        "input[{i}]: only user messages are supported"
                    )));
                }
                match message.get("content") {
                    Some(Value::String(text)) => segments.push(text.clone()),
                    Some(Value::Array(parts)) => {
                        for part in parts {
                            match part.get("type").and_then(Value::as_str) {
                                Some("input_text") => {
                                    segments.push(
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
                                    let url = part.get("image_url").ok_or_else(|| {
                                        ApiError::InvalidRequest(
                                            "input_image parts must carry image_url".to_string(),
                                        )
                                    })?;
                                    let url = url.as_str().ok_or_else(|| {
                                        ApiError::InvalidRequest(
                                            "input_image image_url must be a string (the \
                                             Responses-API shape)"
                                                .to_string(),
                                        )
                                    })?;
                                    segments.push(image_part(url)?);
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
        }
        other => {
            return Err(ApiError::InvalidRequest(format!(
                "input must be a string or an array of user messages (got {other})"
            )));
        }
    }
    Ok((segments.join("\n"), media))
}

fn parse_data_uri(url: &str) -> Result<crate::tokenize::MediaItem, ApiError> {
    use base64::Engine as _;
    if !url.starts_with("data:") {
        return Err(ApiError::InvalidRequest(
            "input_image supports data URIs only for now (got a non-data url); inline the \
             image as data:image/png;base64,..."
                .to_string(),
        ));
    }
    let (head, payload) = url
        .split_once(',')
        .ok_or_else(|| ApiError::InvalidRequest("input_image data URI has no comma".to_string()))?;
    let mime = head
        .strip_prefix("data:")
        .and_then(|h| h.split(';').next())
        .unwrap_or("");
    let format = match mime {
        "image/png" => "png",
        "image/jpeg" => "jpeg",
        "image/jpg" => "jpg",
        "image/webp" => "webp",
        other => {
            return Err(ApiError::InvalidRequest(format!(
                "input_image mime {other:?} is not a supported image type"
            )));
        }
    };
    let data = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|err| {
            ApiError::InvalidRequest(format!("input_image base64 is invalid ({err})"))
        })?;
    if data.is_empty() {
        return Err(ApiError::InvalidRequest(
            "input_image decoded image is empty".to_string(),
        ));
    }
    Ok(crate::tokenize::MediaItem::Image(
        crate::tokenize::MediaFrame {
            format: format.to_string(),
            data,
        },
    ))
}

pub fn choice_value_text(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(ApiError::InvalidRequest(format!(
            "choice values must be strings or booleans (got {other})"
        ))),
    }
}

struct Prepared {
    name: Option<String>,
    task: &'static str,
    instructions: String,

    options: Vec<String>,

    values: Vec<Value>,

    labels: Vec<String>,
}
fn question_error(pre: &Prepared, index: usize, detail: impl std::fmt::Display) -> ApiError {
    match &pre.name {
        Some(name) => ApiError::InvalidRequest(format!("question {name:?}: {detail}")),
        None => ApiError::InvalidRequest(format!("question {index}: {detail}")),
    }
}

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
    let (state, media) = flatten_input(&req.input, pool.media_capable())?;
    let prepared: Vec<Prepared> = req
        .questions
        .iter()
        .enumerate()
        .map(|(index, raw)| prepare(index, raw))
        .collect::<Result<_, _>>()?;
    let request = pool.next_request();
    let results = try_join_all(prepared.iter().map(|p| {
        let media = media.clone();
        let state = state.clone();
        let pool = Arc::clone(&pool);
        async move {
            let started = Instant::now();
            let result = pool
                .run_task(
                    p.task,
                    &p.instructions,
                    &state,
                    &p.options,
                    &media,
                    0,
                    request,
                )
                .await;
            if result.is_ok() {
                metrics::record_decision(p.task, started.elapsed().as_millis() as u64);
            }
            result
        }
    }))
    .await?;
    let mut answers = Vec::with_capacity(prepared.len());
    let mut input_tokens = 0u64;
    for (p, (calibrated, tokens)) in prepared.iter().zip(&results) {
        input_tokens += tokens;
        answers.push(answer(p, calibrated));
    }
    if !media.is_empty()
        && matches!(
            pool.family(),
            crate::family::Family::LetterReadout | crate::family::Family::LetterBranches
        )
    {
        let n = results.len() as u64;
        let first = &prepared[0];
        let row_len = pool.build_row_ids(first.task, &first.instructions, &state, &first.options)[0]
            .len() as u64;
        let prefix = results[0].1 - (row_len - pool.state_prefix_len(&state) as u64);
        input_tokens -= (n - 1) * prefix;
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
