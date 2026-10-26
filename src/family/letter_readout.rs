use crate::calibration::{CalibCtx, Calibrated, CalibrationConfig, argmax, round4, softmax};
use crate::tokenize::{BuiltSequence, TokenizerBundle};

pub const NARROW: usize = 10;

pub const MAX_OPTIONS: usize = 255;

pub const LETTERS: [&str; NARROW] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

pub const NOUL_DEFAULT_PAIR: [&str; 2] = ["no", "yes"];

pub const NOUL_WITHOUT_INSTRUCTIONS: &str = "Which answer fits the context?";

const ISOLATED: &str = "\nProposed answer: ";
const ISOLATED_TAIL: &str = "\nDoes the proposed answer fit?";

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

pub fn strip_level_number(text: &str) -> &str {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let ws = |i: &mut usize| {
        while *i < n && chars[*i].is_whitespace() {
            *i += 1;
        }
    };
    ws(&mut i);
    if i < n && chars[i] == '-' {
        i += 1;
    }
    let digits_start = i;
    while i < n && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return text;
    }
    ws(&mut i);
    if i >= n || chars[i] != ':' {
        return text;
    }
    i += 1;
    ws(&mut i);
    let byte = if i < n {
        text.char_indices()
            .nth(i)
            .map(|(b, _)| b)
            .unwrap_or(text.len())
    } else {
        text.len()
    };
    &text[byte..]
}

fn build_row(
    bundle: &TokenizerBundle,
    question: &str,
    state: &str,
    options: &[String],
) -> BuiltSequence {
    let ctx = bundle.encode_plain(&format!("Context:\n{state}"));
    let mut ids: Vec<u32> = ctx.into_iter().take(bundle.layout.max_len).collect();
    let nopts = options.len();
    if nopts <= NARROW {
        let mut block = format!("\n\nQuestion: {question}\nOptions:");
        for (j, opt) in options.iter().enumerate() {
            block.push_str(&format!("\n({}) {}", LETTERS[j], opt));
        }
        block.push_str("\nAnswer: (");
        ids.extend(bundle.encode_plain(&block));
    } else {
        ids.extend(bundle.encode_plain(&format!("\n\nQuestion: {question}\nOptions:")));
        let open = bundle.encode_plain("\n(");
        for (j, opt) in options.iter().enumerate() {
            ids.extend_from_slice(&open);
            ids.push(bundle.letter_table[j]);
            ids.extend(bundle.encode_plain(&format!(") {opt}")));
        }
        ids.extend(bundle.encode_plain("\nAnswer: ("));
    }
    let slot = ids.len() - 1;
    let full_len = ids.len();
    BuiltSequence {
        ids,
        markers: vec![slot],
        full_len,
        state_tokens: 0,
        media: Vec::new(),
        text: String::new(),
        prefix_len: 0,
        nopts: vec![nopts],
    }
}

pub fn build_rows(
    bundle: &TokenizerBundle,
    task: &str,
    question: &str,
    state: &str,
    options: &[String],
) -> Vec<BuiltSequence> {
    let options = render_options(task, options);
    if task != "score" {
        return vec![build_row(bundle, question, state, &options)];
    }
    options
        .iter()
        .map(|level| {
            let q = format!(
                "{question}{ISOLATED}{}{ISOLATED_TAIL}",
                strip_level_number(level)
            );
            build_row(bundle, &q, state, &NOUL_DEFAULT_PAIR.map(|s| s.to_string()))
        })
        .collect()
}

pub fn unique_tokens(rows: &[&[u32]]) -> u64 {
    if rows.len() < 2 {
        return rows.iter().map(|r| r.len() as u64).sum();
    }
    let short = rows.iter().map(|r| r.len()).min().unwrap();
    let mut lcp = 0;
    'outer: while lcp < short {
        let v = rows[0][lcp];
        for r in rows {
            if r[lcp] != v {
                break 'outer;
            }
        }
        lcp += 1;
    }
    lcp as u64
        + rows
            .iter()
            .map(|r| r.len() as u64 - lcp as u64)
            .sum::<u64>()
}

pub fn serialize_state(state: &serde_json::Value) -> String {
    use serde_json::Value;
    match state {
        Value::String(s) => s.clone(),
        other => crate::systemone::py_json_dumps(&annotate_indices(other)),
    }
}

fn annotate_indices(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::{Map, Value, json};
    match v {
        Value::Array(items) if items.len() >= 8 => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(i, item)| match annotate_indices(item) {
                    Value::Object(inner) => {
                        let mut out = Map::new();
                        out.insert("_index".to_string(), json!(i));
                        for (key, value) in inner {
                            out.insert(key, value);
                        }
                        Value::Object(out)
                    }
                    other => json!({"_index": i, "value": other}),
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(annotate_indices).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), annotate_indices(value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

pub fn certainty(p: &[f64]) -> f64 {
    if p.len() <= 1 {
        return 1.0;
    }
    let h: f64 = p
        .iter()
        .map(|&x| if x > 0.0 { -x * x.ln() } else { 0.0 })
        .sum();
    (1.0 - h / (p.len() as f64).ln()).max(0.0)
}
fn clip01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn normalised(p: &[f64]) -> Vec<f64> {
    let tot: f64 = p.iter().sum();
    if tot == 0.0 {
        return vec![1.0 / p.len() as f64; p.len()];
    }
    p.iter().map(|x| x / tot).collect()
}

pub fn choice_confidence(p: &[f64]) -> f64 {
    let n = p.len();
    if n <= 1 {
        return 1.0;
    }
    let p = normalised(p);
    let p_max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    clip01((n as f64 * p_max - 1.0) / (n as f64 - 1.0))
}

pub fn score_confidence(p: &[f64]) -> f64 {
    let n = p.len();
    if n <= 1 {
        return 1.0;
    }
    let p = normalised(p);
    let k = argmax(&p);
    let spread: f64 = p
        .iter()
        .enumerate()
        .map(|(i, x)| x * (i as f64 - k as f64).abs())
        .sum();
    let uniform: f64 = (0..n)
        .map(|i| (i as f64 - (n as f64 - 1.0) / 2.0).abs())
        .sum::<f64>()
        / n as f64;
    clip01(1.0 - spread / uniform)
}

fn renormalised(p: &[f64]) -> Vec<f64> {
    let s = p.iter().sum::<f64>();
    if s == 0.0 {
        return p.to_vec();
    }
    p.iter().map(|x| x / s).collect()
}

pub fn calibrate(
    task: &str,
    logits: &[f64],
    _ctx: &CalibCtx,
    cfg: &CalibrationConfig,
) -> Calibrated {
    let CalibrationConfig::ByType(table) = cfg else {
        unreachable!("the by-type schema is only constructed for this family")
    };
    let t = table.temperature_for(task);
    match task {
        "score" => {
            assert!(
                !logits.is_empty() && logits.len().is_multiple_of(2),
                "letter score rows are [no, yes] pairs"
            );
            let pairs: Vec<[f64; 2]> = logits.as_chunks::<2>().0.to_vec();
            let fit: Vec<f64> = pairs
                .iter()
                .map(|[no, yes]| {
                    let p = softmax(&[*no / t, *yes / t]);
                    p[1]
                })
                .collect();
            let tot: f64 = fit.iter().sum();
            let mass = if tot == 0.0 { 1e-9 } else { tot };
            let p = renormalised(&fit.iter().map(|x| x / mass).collect::<Vec<_>>());
            let j = argmax(&p);
            let probabilities: Vec<f64> = p.iter().map(|v| round4(*v)).collect();
            Calibrated {
                score: Some(round2(
                    p.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>(),
                )),
                confidence: round4(score_confidence(&p)),
                x_p_max: Some(round4(p[j])),
                certainty: Some(round4(certainty(&p))),
                level_fit: Some(fit.iter().map(|v| round4(*v)).collect()),
                fit_mass: Some(round4(mass)),
                probabilities,
                choice: None,
                probability: None,
            }
        }
        "noul" => {
            let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
            let p = softmax(&scaled);
            Calibrated {
                probability: Some(round4(p[1])),
                probabilities: p.iter().map(|v| round4(*v)).collect(),
                confidence: round4(choice_confidence(&p)),
                choice: None,
                score: None,
                x_p_max: None,
                certainty: None,
                level_fit: None,
                fit_mass: None,
            }
        }
        _ => {
            let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
            let p = renormalised(&softmax(&scaled));
            let j = argmax(&p);
            Calibrated {
                probabilities: p.iter().map(|v| round4(*v)).collect(),
                choice: Some(j),
                confidence: round4(choice_confidence(&p)),
                x_p_max: Some(round4(p[j])),
                certainty: Some(round4(certainty(&p))),
                probability: None,
                score: None,
                level_fit: None,
                fit_mass: None,
            }
        }
    }
}
pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
