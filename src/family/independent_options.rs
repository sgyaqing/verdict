use crate::calibration::{CalibCtx, Calibrated, CalibrationConfig, argmax, round4, softmax};
use crate::tokenize::{BuiltSequence, TokenizerBundle};

pub const NOUL_DEFAULT_PAIR: [&str; 2] = ["Yes, condition holds true.", "No, condition is false."];

fn neutralise(text: &str, specials: [&str; 2]) -> String {
    let mut out = text.to_string();
    for sp in specials {
        if sp.len() >= 2 && out.contains(sp) {
            out = out.replace(sp, &format!("{}\u{200d}{}", &sp[..1], &sp[1..]));
        }
    }
    out
}

fn split_digits(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 2);
    let mut prev_digit = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            if prev_digit {
                out.push(' ');
            }
            prev_digit = true;
        } else {
            prev_digit = false;
        }
        out.push(ch);
    }
    out
}

pub fn render_options(task: &str, options: &[String]) -> Vec<String> {
    match task {
        "choice" | "score" => options.to_vec(),
        _ => {
            if options.len() == 2 {
                options.to_vec()
            } else {
                NOUL_DEFAULT_PAIR.map(|s| s.to_string()).to_vec()
            }
        }
    }
}

fn fit_state(bundle: &TokenizerBundle, state: &str, reserve: usize) -> (String, usize) {
    let ids = bundle.encode_plain(state);
    let limit = bundle.layout.max_len.saturating_sub(reserve).max(16);
    if ids.len() <= limit {
        return (state.to_string(), ids.len());
    }
    let head = (limit as f64 * 0.6) as usize;
    let tail = limit - head - 2;
    let head_text = bundle
        .tok
        .decode(&ids[..head], true)
        .expect("tokenizer decode cannot fail on valid ids");
    let tail_text = bundle
        .tok
        .decode(&ids[ids.len() - tail..], true)
        .expect("tokenizer decode cannot fail on valid ids");
    let fitted = format!("{head_text} ... {tail_text}");
    let fitted_len = bundle.encode_plain(&fitted).len();
    (fitted, fitted_len)
}

pub fn build_sequence(
    bundle: &TokenizerBundle,
    task: &str,
    question: &str,
    state: &str,
    options: &[String],
) -> BuiltSequence {
    let layout = &bundle.layout;
    let specials = [layout.mask_literal.as_str(), layout.sep_literal.as_str()];
    let question_n = neutralise(question, specials);
    let descs: Vec<String> = render_options(task, options)
        .iter()
        .map(|opt| neutralise(opt, specials).trim().to_string())
        .collect();
    let opts_packed = descs
        .iter()
        .map(|d| format!("{} {}", layout.mask_literal, d))
        .collect::<Vec<_>>()
        .join(" ");
    let reserve_prefix = if question_n.is_empty() {
        String::new()
    } else {
        format!("{} ", question_n.trim())
    };
    let reserve_packed = format!("{}{} {}", reserve_prefix, layout.sep_literal, opts_packed);
    let reserve = 2 + bundle.encode_plain(&reserve_packed).len() + 8;
    let (state_fitted, state_tokens) = fit_state(bundle, state, reserve);
    let state_n = neutralise(&state_fitted, specials);
    let prefix = if question_n.is_empty() {
        state_n.trim().to_string()
    } else {
        format!("{} {}", question_n, state_n).trim().to_string()
    };
    let mut packed = format!(
        "{} {} {}",
        prefix,
        layout.sep_literal,
        descs
            .iter()
            .map(|d| format!("{} {}", layout.mask_literal, d))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if bundle.digit_split {
        packed = split_digits(&packed);
    }
    let body = bundle.encode_plain(&packed);
    let mut ids = Vec::with_capacity(body.len() + 2);
    ids.push(layout.cls_id);
    ids.extend(body);
    ids.push(layout.sep_id);
    let markers: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter(|(_, id)| **id == layout.mask_id)
        .map(|(i, _)| i)
        .collect();
    BuiltSequence {
        full_len: ids.len(),
        state_tokens,
        ids,
        markers,
        media: Vec::new(),
        text: String::new(),
        prefix_len: 0,
        nopts: Vec::new(),
    }
}

pub fn noul_decide(p: f64, band: &crate::calibration::NoulBand) -> f64 {
    let p = p.clamp(0.0, 1.0);
    let out = if p >= 0.5 {
        band.edge + band.slope * (p - 0.5)
    } else {
        (1.0 - band.edge) - band.slope * (0.5 - p)
    };
    out.clamp(0.0, 1.0)
}

pub fn temperature_of(
    logits: &[f64],
    state_tokens: usize,
    map: &crate::calibration::FormulaMap,
) -> f64 {
    let n = logits.len().max(1);
    let entropy = if n > 1 {
        let p = softmax(logits);
        -p.iter().map(|&x| x * x.max(1e-12).ln()).sum::<f64>() / (n as f64).ln()
    } else {
        0.0
    };
    let raw = map.bias
        + map.entropy * entropy
        + map.log_tokens * ((state_tokens.max(1) as f64).log10() / 4.0)
        + map.n_options * (logits.len() as f64 / 8.0);
    raw.clamp(map.lo, map.hi)
}

pub fn confidence_from_probs(p: &[f64]) -> f64 {
    let n = p.len();
    if n <= 1 {
        return 1.0;
    }
    let conf =
        (n as f64 * p.iter().copied().fold(f64::NEG_INFINITY, f64::max) - 1.0) / (n as f64 - 1.0);
    round3(conf.clamp(0.0, 1.0))
}
pub fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}
pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

pub fn calibrate(
    task: &str,
    logits: &[f64],
    ctx: &CalibCtx,
    cfg: &CalibrationConfig,
) -> Calibrated {
    let CalibrationConfig::Formula { map, noul_band, .. } = cfg else {
        unreachable!("the formula schema is only constructed for this family")
    };
    let t = temperature_of(logits, ctx.state_tokens, map);
    let scaled: Vec<f64> = logits.iter().map(|l| l / t.max(1e-4)).collect();
    let p = softmax(&scaled);
    let probabilities: Vec<f64> = p.iter().map(|v| round4(*v)).collect();
    let confidence = confidence_from_probs(&p);
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
            probability: Some(round4(noul_decide(probabilities[0], noul_band))),
            choice: None,
            score: None,
            probabilities,
            confidence,
            x_p_max: None,
            certainty: None,
            level_fit: None,
            fit_mass: None,
        },
        _ => Calibrated {
            score: Some(round2(
                p.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>(),
            )),
            choice: None,
            probability: None,
            probabilities,
            confidence,
            x_p_max: None,
            certainty: None,
            level_fit: None,
            fit_mass: None,
        },
    }
}
