use crate::tokenize::{BuiltSequence, TokenizerBundle};

const OPT_TEXT_CAP: usize = 48;

const MIN_HEAD_BUDGET: usize = 16;

const MIN_OPT_TOKENS: usize = 4;

pub const NOUL_DEFAULT_PAIR: [&str; 2] = [
    "false: no, the statement does not hold",
    "true: yes, the statement holds",
];
pub const NOUL_DEFAULT_DESCRIPTIONS: [&str; 2] = [
    "no, the statement does not hold",
    "yes, the statement holds",
];

pub fn render_options(task: &str, options: &[String]) -> Vec<String> {
    match task {
        "choice" => options.to_vec(),
        "score" => options
            .iter()
            .enumerate()
            .map(|(i, text)| format!("level {i}: {text}"))
            .collect(),
        _ => {
            if options.len() == 2 {
                options.to_vec()
            } else {
                NOUL_DEFAULT_PAIR.map(|s| s.to_string()).to_vec()
            }
        }
    }
}

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
    let mut full_len =
        1 + head_full.len() + 1 + opt_full.iter().map(|o| o.len()).sum::<usize>() + 1;
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
        media: Vec::new(),
        text: String::new(),
        prefix_len: 0,
        nopts: Vec::new(),
    }
}

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

pub fn lookup_temperature(table: &crate::calibration::BandTable, task: &str, k: usize) -> f64 {
    let key = format!("{task}:{}", band_of(k));
    let qtype = crate::calibration::qtype_of(task).unwrap_or(0);
    table
        .temperature_by_options
        .get(&key)
        .copied()
        .unwrap_or(table.temperature[qtype])
}

pub fn confidence_from_probs(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let entropy: f64 = p.iter().map(|&x| -x * x.max(1e-12).ln()).sum();
    1.0 - entropy / (k as f64).ln()
}

pub fn calibrate(
    task: &str,
    logits: &[f64],
    cfg: &crate::calibration::CalibrationConfig,
) -> crate::calibration::Calibrated {
    use crate::calibration::{BandTable, Calibrated, argmax, round4, softmax};
    let table = match cfg {
        crate::calibration::CalibrationConfig::Bands(table) => table,
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

pub fn noul_pair(criteria: Option<&serde_json::Value>) -> Result<Vec<String>, String> {
    use serde_json::Value;
    let (t, f) = match criteria {
        None | Some(Value::Null) => return Ok(Vec::new()),
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
