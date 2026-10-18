//! The marker-head family recipe: task-prefixed head, shared-attention option
//! markers, band-table temperature calibration.
//!
//! Every rule is re-typed from this family's official Python implementation
//! and pinned by the golden vectors (tokenization and calibration parity);
//! authoritative facts
//! live in each golden file's `meta` block. Per-model data (special-token
//! names, budgets, graph name, temperature tables) comes from model.json via
//! the generic loaders; only the recipe lives here.

use crate::tokenize::{BuiltSequence, TokenizerBundle};

/// Per-option token cap: one mask token + at most 48 text tokens (rl_common line 60).
const OPT_TEXT_CAP: usize = 48;
/// If the option block leaves less head budget than this, every option is shrunk
/// evenly and the head shrinks to the remainder (rl_common lines 62-66).
const MIN_HEAD_BUDGET: usize = 16;
/// Options may never shrink below this many tokens (rl_common line 63).
const MIN_OPT_TOKENS: usize = 4;
/// The fixed noul pair (rl_common.render_options): `p[1] == P(true)` depends
/// on this exact wording. The bare descriptions back the /v1/systemone
/// adapter's one-sided Jev criteria fallback; PAIR[i] is
/// `"{false|true}: {DESCRIPTIONS[i]}"`.
pub const NOUL_DEFAULT_PAIR: [&str; 2] = [
    "false: no, the statement does not hold",
    "true: yes, the statement holds",
];
pub const NOUL_DEFAULT_DESCRIPTIONS: [&str; 2] = [
    "no, the statement does not hold",
    "yes, the statement holds",
];

/// Option texts in marker order, re-typed from rl_common.render_options:
/// choice passes options through (crit values are None); score renders
/// "level %d: %s"; noul renders the two fixed sentences so `p[1] == P(true)`.
pub fn render_options(task: &str, options: &[String]) -> Vec<String> {
    match task {
        "choice" => options.to_vec(),
        "score" => options
            .iter()
            .enumerate()
            .map(|(i, text)| format!("level {i}: {text}"))
            .collect(),
        // A caller-supplied 2-item pair (Jev criteria via /v1/systemone)
        // replaces the default sentences; the native protocol passes none.
        _ => {
            if options.len() == 2 {
                options.to_vec()
            } else {
                NOUL_DEFAULT_PAIR.map(|s| s.to_string()).to_vec()
            }
        }
    }
}

/// Build the request sequence: `[CLS]` head `[SEP]` (`[MASK]` opt)k `[SEP]` state `[SEP]`,
/// with the full model-layer truncation semantics (48-cap, 192 shrink, 512 room).
/// `options` carries the option/level texts (choice/score); noul uses its
/// fixed pair unless the caller supplies two custom sentences (Jev criteria).
pub fn build_sequence(
    bundle: &TokenizerBundle,
    task: &str,
    question: &str,
    state: &str,
    options: &[String],
) -> BuiltSequence {
    let layout = &bundle.layout;
    let ins = question.replace(&layout.mask_literal, " ");
    let head_full = bundle.encode_plain(&format!("{task} question: {ins}"));

    // Uncapped option encodings: `[MASK]` + " " + text (mask included in each).
    let opt_full: Vec<Vec<u32>> = render_options(task, options)
        .iter()
        .map(|opt| {
            let mut ids = vec![layout.mask_id];
            ids.extend(
                bundle.encode_plain(&format!(" {}", opt.replace(&layout.mask_literal, " "))),
            );
            ids
        })
        .collect();
    // Uncapped rendered length: `[CLS]` head `[SEP]` opts `[SEP]` state [SEP].
    let mut full_len =
        1 + head_full.len() + 1 + opt_full.iter().map(|o| o.len()).sum::<usize>() + 1;

    // Per-option cap: the mask token is kept and at most OPT_TEXT_CAP text tokens
    // follow it (rl_common line 60 caps the text, then prepends the mask -- the
    // cap never eats the mask).
    let mut opt_ids: Vec<Vec<u32>> = opt_full
        .iter()
        .map(|o| o[..(1 + OPT_TEXT_CAP).min(o.len())].to_vec())
        .collect();
    let mut opt_budget = layout
        .head_max_len
        .saturating_sub(opt_ids.iter().map(|o| o.len()).sum::<usize>());
    if opt_budget < MIN_HEAD_BUDGET {
        let per =
            ((layout.head_max_len - MIN_HEAD_BUDGET) / opt_ids.len().max(1)).max(MIN_OPT_TOKENS);
        opt_ids = opt_ids
            .iter()
            .map(|o| o[..per.min(o.len())].to_vec())
            .collect();
        opt_budget = layout
            .head_max_len
            .saturating_sub(opt_ids.iter().map(|o| o.len()).sum::<usize>());
    }
    let head_ids = &head_full[..head_full.len().min(opt_budget.max(8))];

    let mut ids = Vec::with_capacity(layout.max_len);
    ids.push(layout.cls_id);
    ids.extend_from_slice(head_ids);
    ids.push(layout.sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for opt in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(opt);
    }
    ids.push(layout.sep_id);

    let state_full = bundle.encode_plain(&state.replace(&layout.mask_literal, " "));
    full_len += state_full.len() + 1;
    // API path default: truncate_right (keep the head of the state).
    let room = layout.max_len.saturating_sub(ids.len() + 1);
    ids.extend_from_slice(&state_full[..state_full.len().min(room)]);
    ids.push(layout.sep_id);
    ids.truncate(layout.max_len);

    BuiltSequence {
        markers: markers
            .into_iter()
            .filter(|m| *m < layout.max_len)
            .collect(),
        state_tokens: state_full.len(),
        ids,
        full_len,
        nopts: Vec::new(),
    }
}

/// Band rule re-typed from rl_common.temp_bucket: k<=2 -> "2", k<=5 -> "3-5",
/// k<=10 -> "6-10", else "11+".
pub fn band_of(k: usize) -> &'static str {
    if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    }
}

/// `temperature_by_options["task:band"]` with the `temperature[qtype]` fallback
/// (rl_agent_api.py line 62).
pub fn lookup_temperature(table: &crate::calibration::BandTable, task: &str, k: usize) -> f64 {
    let key = format!("{task}:{}", band_of(k));
    let qtype = crate::calibration::qtype_of(task).unwrap_or(0);
    table
        .temperature_by_options
        .get(&key)
        .copied()
        .unwrap_or(table.temperature[qtype])
}

/// Jev-style confidence: 1 - normalized entropy of the FULL-PRECISION answer
/// distribution (the reference `confidenceFromProbs`, ported exactly: the
/// 1e-12 clamp and the k < 2 pin are part of the port). Served by
/// /v1/systemone; the native protocol keeps it off its wire shape.
pub fn confidence_from_probs(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let entropy: f64 = p.iter().map(|&x| -x * x.max(1e-12).ln()).sum();
    1.0 - entropy / (k as f64).ln()
}

/// The full official calibration: z = logits / T; p = softmax(z); round(p, 4).
pub fn calibrate(
    task: &str,
    logits: &[f64],
    cfg: &crate::calibration::CalibrationConfig,
) -> crate::calibration::Calibrated {
    use crate::calibration::{BandTable, Calibrated, argmax, round4, softmax};
    let table = match cfg {
        crate::calibration::CalibrationConfig::Bands(table) => table,
        // The loader only builds the band schema for this family.
        _ => unreachable!("the band schema is only constructed for this family"),
    };
    let BandTable { .. } = table;
    let t = lookup_temperature(table, task, logits.len());
    let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
    let p = softmax(&scaled);
    let probabilities: Vec<f64> = p.iter().map(|v| round4(*v)).collect();
    let confidence = round4(confidence_from_probs(&p));
    match task {
        "choice" => Calibrated {
            probabilities,
            choice: Some(argmax(logits)),
            probability: None,
            score: None,
            confidence,
            x_p_max: None,
            certainty: None,
            level_fit: None,
            fit_mass: None,
        },
        "noul" => Calibrated {
            probabilities,
            choice: None,
            probability: Some(round4(p[1])),
            score: None,
            confidence,
            x_p_max: None,
            certainty: None,
            level_fit: None,
            fit_mass: None,
        },
        _ => Calibrated {
            probabilities,
            choice: None,
            probability: None,
            score: Some(round4(
                p.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>(),
            )),
            confidence,
            x_p_max: None,
            certainty: None,
            level_fit: None,
            fit_mass: None,
        },
    }
}

/// The noul option pair for /v1/systemone Jev criteria. No criteria (or an
/// empty object) yields no options, letting build_sequence render its default
/// pair; the Jev criteria `{"true": ..., "false": ...}` yield the two
/// sentences, each side falling back to its default when absent or empty (JS
/// truthiness: an empty string takes the default).
pub fn noul_pair(criteria: Option<&serde_json::Value>) -> Result<Vec<String>, String> {
    use serde_json::Value;
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
