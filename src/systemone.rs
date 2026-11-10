use crate::calibration::Calibrated;
use crate::family::Family;
pub use crate::family::marker_head::noul_pair;
use crate::metrics;
use crate::pipeline::Pool;
use crate::protocol::{ApiError, MAX_OPTIONS, MIN_OPTIONS, ValidJson};
use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::post;
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::Arc;
use std::time::Instant;

pub fn routes() -> Router<Arc<Pool>> {
    Router::new().route("/v1/systemone", post(system_one))
}
#[derive(Debug, Deserialize)]
pub struct SystemOneRequest {
    #[serde(default)]
    pub model: Option<Value>,

    pub state: Value,

    pub questions: Map<String, Value>,

    #[serde(default)]
    pub assets: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize)]
struct RawQuestion {
    #[serde(rename = "type")]
    qtype: String,
    #[serde(default)]
    instructions: Option<Value>,
    #[serde(default)]
    criteria: Option<Value>,
}

pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => py_json_dumps(other),
    }
}

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

pub fn instructions_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).expect("a JSON value serializes"),
    }
}

pub fn py_json_dumps_indent2(v: &Value, level: usize) -> String {
    let pad = |n: usize| "  ".repeat(n);
    match v {
        Value::Array(items) if items.is_empty() => "[]".to_string(),
        Value::Array(items) => format!(
            "[\n{}\n{}]",
            items
                .iter()
                .map(|item| format!(
                    "{}{}",
                    pad(level + 1),
                    py_json_dumps_indent2(item, level + 1)
                ))
                .collect::<Vec<_>>()
                .join(",\n"),
            pad(level)
        ),
        Value::Object(map) if map.is_empty() => "{}".to_string(),
        Value::Object(map) => format!(
            "{{\n{}\n{}}}",
            map.iter()
                .map(|(key, value)| format!(
                    "{}{}: {}",
                    pad(level + 1),
                    serde_json::to_string(key).expect("a JSON string serializes"),
                    py_json_dumps_indent2(value, level + 1)
                ))
                .collect::<Vec<_>>()
                .join(",\n"),
            pad(level)
        ),
        other => py_json_dumps(other),
    }
}

fn parse_assets(
    assets: &Map<String, Value>,
) -> Result<Vec<(String, crate::tokenize::MediaItem)>, ApiError> {
    use base64::Engine as _;
    if assets.is_empty() {
        return Err(ApiError::InvalidRequest(
            "assets must name at least one image".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(assets.len());
    let mut total_ref_bytes = 0usize;
    for (r, v) in assets {
        if r.is_empty() || r.len() > 1024 {
            return Err(ApiError::InvalidRequest(format!(
                "asset reference {r:?} must be 1..=1024 bytes"
            )));
        }
        total_ref_bytes += r.len();
        if total_ref_bytes > 65536 {
            return Err(ApiError::InvalidRequest(
                "asset references exceed 65536 bytes in total".to_string(),
            ));
        }
        let ty = v.get("type").and_then(Value::as_str).unwrap_or("image");
        let frame = |v: &Value| -> Result<crate::tokenize::MediaFrame, ApiError> {
            let data_b64 = v
                .get("data_base64")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiError::InvalidRequest(format!(
                        "asset {r:?}: data_base64 is required (base64 image bytes)"
                    ))
                })?;
            let format = v.get("format").and_then(Value::as_str).ok_or_else(|| {
                ApiError::InvalidRequest(format!(
                    "asset {r:?}: format is required (\"png\", \"jpeg\", ...)"
                ))
            })?;
            if data_b64.is_empty() || format.is_empty() {
                return Err(ApiError::InvalidRequest(format!(
                    "asset {r:?}: data_base64 and format must be non-empty"
                )));
            }
            let data = base64::engine::general_purpose::STANDARD
                .decode(data_b64)
                .map_err(|err| {
                    ApiError::InvalidRequest(format!("asset {r:?}: invalid base64 ({err})"))
                })?;
            if data.is_empty() {
                return Err(ApiError::InvalidRequest(format!(
                    "asset {r:?}: decoded frame is empty"
                )));
            }
            Ok(crate::tokenize::MediaFrame {
                format: format.to_string(),
                data,
            })
        };
        let item = match ty {
            "image" => crate::tokenize::MediaItem::Image(frame(v)?),
            "video" => {
                let frames = v
                    .get("frames")
                    .and_then(Value::as_array)
                    .filter(|list| !list.is_empty())
                    .ok_or_else(|| {
                        ApiError::InvalidRequest(format!(
                            "asset {r:?}: video needs a non-empty frames list \
                             (caller-extracted frames; no container decoding)"
                        ))
                    })?;
                let frames = frames.iter().map(frame).collect::<Result<Vec<_>, _>>()?;
                let fps = match v.get("fps") {
                    None | Some(Value::Null) => 1.0,
                    Some(Value::Number(n)) => {
                        let fps = n.as_f64().unwrap_or(0.0);
                        if fps.is_finite() && fps > 0.0 {
                            fps
                        } else {
                            return Err(ApiError::InvalidRequest(format!(
                                "asset {r:?}: fps must be a positive number (got {n})"
                            )));
                        }
                    }
                    Some(other) => {
                        return Err(ApiError::InvalidRequest(format!(
                            "asset {r:?}: fps must be a positive number (got {other})"
                        )));
                    }
                };
                crate::tokenize::MediaItem::Video { frames, fps }
            }
            other => {
                return Err(ApiError::InvalidRequest(format!(
                    "asset {r:?}: unknown type {other:?} (expected \"image\" or \"video\")"
                )));
            }
        };
        out.push((r.clone(), item));
    }
    Ok(out)
}

fn resolve_assets(
    state: &mut Value,
    assets: &[(String, crate::tokenize::MediaItem)],
    img_block: &str,
    vid_block: &str,
) -> Result<Vec<crate::tokenize::MediaItem>, ApiError> {
    let mut used = vec![false; assets.len()];
    let mut media: Vec<crate::tokenize::MediaItem> = Vec::new();
    fn walk(
        v: &mut Value,
        assets: &[(String, crate::tokenize::MediaItem)],
        img_block: &str,
        vid_block: &str,
        used: &mut Vec<bool>,
        media: &mut Vec<crate::tokenize::MediaItem>,
    ) {
        match v {
            Value::String(s) => {
                let mut out = String::with_capacity(s.len());
                let mut rest = s.as_str();
                while !rest.is_empty() {
                    let mut best: Option<(usize, usize)> = None;
                    for (i, (r, _)) in assets.iter().enumerate() {
                        if rest.starts_with(r.as_str()) {
                            let len = r.len();
                            if best.is_none_or(|(_, l)| len > l) {
                                best = Some((i, len));
                            }
                        }
                    }
                    match best {
                        Some((i, len)) => {
                            out.push_str(match &assets[i].1 {
                                crate::tokenize::MediaItem::Image(_) => img_block,
                                crate::tokenize::MediaItem::Video { .. } => vid_block,
                            });
                            media.push(assets[i].1.clone());
                            used[i] = true;
                            rest = &rest[len..];
                        }
                        None => {
                            let cut = rest
                                .char_indices()
                                .nth(1)
                                .map(|(i, _)| i)
                                .unwrap_or(rest.len());
                            let (ch, tail) = rest.split_at(cut);
                            out.push_str(ch);
                            rest = tail;
                        }
                    }
                }
                *s = out;
            }
            Value::Array(items) => {
                for item in items {
                    walk(item, assets, img_block, vid_block, used, media);
                }
            }
            Value::Object(map) => {
                for value in map.values_mut() {
                    walk(value, assets, img_block, vid_block, used, media);
                }
            }
            _ => {}
        }
    }
    walk(state, assets, img_block, vid_block, &mut used, &mut media);
    if let Some(unused) = used
        .iter()
        .zip(assets)
        .find_map(|(u, (r, _))| (!u).then_some(r))
    {
        return Err(ApiError::InvalidRequest(format!(
            "asset {unused:?} is not referenced anywhere in the state"
        )));
    }
    Ok(media)
}

fn refs_in_questions(
    questions: &Map<String, Value>,
    assets: &[(String, crate::tokenize::MediaItem)],
) -> bool {
    fn any_hit(v: &Value, refs: &[String]) -> bool {
        match v {
            Value::String(s) => refs.iter().any(|r| s.contains(r.as_str())),
            Value::Array(items) => items.iter().any(|i| any_hit(i, refs)),
            Value::Object(map) => map.values().any(|val| any_hit(val, refs)),
            _ => false,
        }
    }
    let refs: Vec<String> = assets.iter().map(|(r, _)| r.clone()).collect();
    questions.values().any(|q| any_hit(q, &refs))
}

struct Prepared {
    qid: String,
    task: &'static str,
    instructions: String,

    options: Vec<String>,

    keys: Vec<String>,

    legend: Option<Vec<Value>>,

    family: Family,
}
fn question_error(qid: &str, detail: impl std::fmt::Display) -> ApiError {
    ApiError::InvalidRequest(format!("question {qid:?}: {detail}"))
}

fn instructions_of(family: Family, raw: Option<&Value>) -> String {
    match family {
        Family::LetterReadout => match raw {
            Some(Value::String(s)) => s.clone(),
            Some(other) => py_json_dumps(other),
            None => String::new(),
        },
        Family::LetterBranches => entry_text(raw),
        _ => instructions_string(raw.expect("checked present for marker families")),
    }
}

fn entry_text(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(other) => py_json_dumps_indent2(other, 0),
    }
}
fn prepare(qid: &str, raw: &Value, family: Family) -> Result<Prepared, ApiError> {
    let q: RawQuestion =
        serde_json::from_value(raw.clone()).map_err(|err| question_error(qid, err))?;
    if family == Family::LetterBranches {
        return prepare_branches(
            qid,
            &instructions_of(family, q.instructions.as_ref()),
            q.qtype.as_str(),
            q.criteria.as_ref(),
        );
    }
    if family != Family::LetterReadout {
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
                legend: Some(levels.into_iter().map(Value::String).collect()),
                family,
            })
        }
        "noul" => {
            let sides = letter_noul_sides(qid, q.criteria.as_ref())?;
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

fn prepare_branches(
    qid: &str,
    instructions: &str,
    qtype: &str,
    criteria: Option<&Value>,
) -> Result<Prepared, ApiError> {
    match qtype {
        "choice" => {
            let map = match criteria {
                Some(Value::Object(map)) => map,
                _ => {
                    return Err(question_error(
                        qid,
                        "criteria is required: an object of option -> description (or null)",
                    ));
                }
            };
            if map.is_empty() {
                return Err(question_error(
                    qid,
                    "choice question needs at least one option in criteria",
                ));
            }
            if map.len() > crate::family::letter_branches::MAX_CHOICE_OPTIONS {
                return Err(question_error(
                    qid,
                    format!(
                        "choice question has {} options, maximum is {}",
                        map.len(),
                        crate::family::letter_branches::MAX_CHOICE_OPTIONS
                    ),
                ));
            }
            let options = map
                .iter()
                .map(|(name, entry)| {
                    let text = entry_text(Some(entry));
                    if text.is_empty() {
                        name.clone()
                    } else {
                        format!("{name}: {text}")
                    }
                })
                .collect();
            Ok(Prepared {
                qid: qid.to_string(),
                task: "choice",
                instructions: instructions.to_string(),
                options,
                keys: map.keys().cloned().collect(),
                legend: None,
                family: Family::LetterBranches,
            })
        }
        "score" => {
            let items = match criteria {
                Some(Value::Array(items)) => items,
                _ => {
                    return Err(question_error(
                        qid,
                        "criteria is required: a list of level descriptions, lowest first",
                    ));
                }
            };
            if items.is_empty() {
                return Err(question_error(
                    qid,
                    "score question needs at least one level in criteria",
                ));
            }
            if items.len() > crate::family::letter_branches::MAX_SCORE_LEVELS {
                return Err(question_error(
                    qid,
                    format!(
                        "score question has {} levels, maximum is {}",
                        items.len(),
                        crate::family::letter_branches::MAX_SCORE_LEVELS
                    ),
                ));
            }
            let options = items
                .iter()
                .enumerate()
                .map(|(i, entry)| {
                    let text = entry_text(Some(entry));
                    if text.is_empty() {
                        format!("level {i}")
                    } else {
                        format!("level {i}: {text}")
                    }
                })
                .collect();
            let keys = (0..items.len()).map(|i| i.to_string()).collect();
            Ok(Prepared {
                qid: qid.to_string(),
                task: "score",
                instructions: instructions.to_string(),
                options,
                keys,
                legend: Some(items.clone()),
                family: Family::LetterBranches,
            })
        }
        "noul" => {
            let sides = match criteria {
                None | Some(Value::Null) => None,
                Some(Value::Object(map)) => {
                    let side = |key: &str| -> Option<String> {
                        match map.get(key) {
                            None | Some(Value::Null) => None,
                            Some(entry) => Some(entry_text(Some(entry))),
                        }
                    };
                    Some((side("true"), side("false")))
                }
                Some(other) => {
                    return Err(question_error(
                        qid,
                        format!(
                            "criteria must be an object with optional \"true\"/\"false\" entries (got {other})"
                        ),
                    ));
                }
            };
            let (t, f) = sides.unwrap_or((None, None));
            let options = vec![
                format!(
                    "yes: {}",
                    t.unwrap_or_else(|| {
                        crate::family::letter_branches::NOUL_DEFAULT_TRUE.to_string()
                    })
                ),
                format!(
                    "no: {}",
                    f.unwrap_or_else(|| {
                        crate::family::letter_branches::NOUL_DEFAULT_FALSE.to_string()
                    })
                ),
            ];
            Ok(Prepared {
                qid: qid.to_string(),
                task: "noul",
                instructions: instructions.to_string(),
                options,
                keys: Vec::new(),
                legend: None,
                family: Family::LetterBranches,
            })
        }
        other => Err(question_error(
            qid,
            format!("unknown type {other:?} (expected \"choice\", \"score\", or \"noul\")"),
        )),
    }
}

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
            let options = keys
                .iter()
                .zip(&descriptions)
                .map(|(key, desc)| match (family, desc) {
                    (Family::MarkerHead, Some(text)) => format!("{key}: {text}"),
                    (Family::IndependentOptions, Some(text)) => text.trim().to_string(),
                    (Family::MarkerHead, None) => key.clone(),
                    (Family::IndependentOptions, None) => key.trim().to_string(),
                    (Family::LetterReadout, _) | (Family::LetterBranches, _) => {
                        unreachable!("dispatched above")
                    }
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
                legend: Some(levels.into_iter().map(Value::String).collect()),
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
                Family::LetterReadout | Family::LetterBranches => {
                    unreachable!("dispatched above")
                }
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

fn letter_choice_criteria(
    qid: &str,
    criteria: Option<&Value>,
) -> Result<(Vec<String>, Vec<String>), ApiError> {
    let (keys, options): (Vec<String>, Vec<String>) = match criteria {
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

fn independent_noul_pair(criteria: Option<&Value>) -> Result<Vec<String>, String> {
    use crate::family::independent_options::NOUL_DEFAULT_PAIR;
    let sides: Option<(Option<String>, Option<String>)> = match criteria {
        None | Some(Value::Null) => None,
        Some(Value::Object(map)) => {
            let side = |key: &str| -> Result<Option<String>, String> {
                match map.get(key) {
                    None | Some(Value::Null) => Ok(None),
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
                    legend.insert(i.to_string(), text.clone());
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
                Family::MarkerHead | Family::LetterReadout | Family::LetterBranches => None,
                Family::IndependentOptions => Some(c.probabilities.first().copied().unwrap_or(0.0)),
            },
        },
    }
}
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
    let assets = match &req.assets {
        Some(map) if !map.is_empty() => parse_assets(map)?,
        _ => Vec::new(),
    };
    let (media, state_value) = if assets.is_empty() {
        (Vec::new(), req.state.clone())
    } else {
        if !pool.media_capable() {
            return Err(ApiError::InvalidRequest(
                "this model does not accept image input (no processor in the package)".to_string(),
            ));
        }
        if refs_in_questions(&req.questions, &assets) {
            return Err(ApiError::InvalidRequest(
                "asset references belong in the state; this model's contract places \
                 images in the state only"
                    .to_string(),
            ));
        }
        let mut state_value = req.state.clone();
        let media = resolve_assets(
            &mut state_value,
            &assets,
            crate::family::letter_branches::VISION_BLOCK,
            crate::family::letter_branches::VISION_BLOCK_VIDEO,
        )?;
        (media, state_value)
    };
    let state = match family {
        Family::LetterReadout => crate::family::letter_readout::serialize_state(&state_value),
        Family::LetterBranches => match &state_value {
            Value::String(s) => s.clone(),
            other => py_json_dumps_indent2(other, 0),
        },
        _ => serialize_state(&state_value),
    };
    let prepared: Vec<Prepared> = req
        .questions
        .iter()
        .map(|(qid, raw)| prepare(qid, raw, family))
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
    let input_tokens =
        match family {
            Family::LetterReadout | Family::LetterBranches if media.is_empty() => {
                let rows: Vec<Vec<u32>> = prepared
                    .iter()
                    .flat_map(|p| pool.build_row_ids(p.task, &p.instructions, &state, &p.options))
                    .collect();
                crate::family::letter_readout::unique_tokens(
                    &rows.iter().map(|r| r.as_slice()).collect::<Vec<_>>(),
                )
            }
            Family::LetterBranches => {
                let n = results.len() as u64;
                let sum: u64 = results.iter().map(|(_, tokens)| *tokens).sum();
                let first = &prepared[0];
                let row_len =
                    pool.build_row_ids(first.task, &first.instructions, &state, &first.options)[0]
                        .len() as u64;
                let prefix = results[0].1 - (row_len - pool.state_prefix_len(&state) as u64);
                sum - (n - 1) * prefix
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
            output_tokens: match family {
                Family::LetterBranches => req.questions.len() as u64,
                _ => 0,
            },
        },
    }))
}
