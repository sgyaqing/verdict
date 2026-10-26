use crate::calibration::{CalibCtx, Calibrated, CalibrationConfig, argmax, softmax};
use std::collections::HashMap;

pub const SYSTEM_PROMPT: &str = "Apply the question to the state. Choose exactly one of the \
     listed options. Respond with only its uppercase letter, with no explanation or reasoning.";

pub const CHAT_PREFIX: &str = "<|im_start|>system\n";
pub const CHAT_MID: &str = "<|im_end|>\n<|im_start|>user\n";
pub const CHAT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub const VISION_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";

pub const VISION_BLOCK_VIDEO: &str = "<|vision_start|><|video_pad|><|vision_end|>";

pub const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

pub const NOT_ONE_TOKEN: [&str; 25] = [
    "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ",
    "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ",
];

pub const MAX_SLOTS: usize = 256;

pub const MAX_CHOICE_OPTIONS: usize = 255;

pub const MAX_SCORE_LEVELS: usize = 10;

pub fn slot_labels() -> Vec<String> {
    let mut out: Vec<String> = LETTERS.chars().map(|c| c.to_string()).collect();
    'outer: for a in LETTERS.chars() {
        for b in LETTERS.chars() {
            if out.len() >= MAX_SLOTS {
                break 'outer;
            }
            let pair = format!("{a}{b}");
            if !NOT_ONE_TOKEN.contains(&pair.as_str()) {
                out.push(pair);
            }
        }
    }
    out
}

pub fn default_instructions(kind: &str) -> &'static str {
    match kind {
        "noul" => "Is the statement true, or is the answer to the question yes?",
        "score" => "Which level describes the state?",
        _ => "Which option applies to the state?",
    }
}

pub const NOUL_DEFAULT_TRUE: &str = "the statement is true / the answer is yes";
pub const NOUL_DEFAULT_FALSE: &str = "the statement is false / the answer is no";

fn options_block(options: &[String]) -> String {
    let labels = slot_labels();
    options
        .iter()
        .enumerate()
        .map(|(i, opt)| format!("{}. {}", labels[i], opt))
        .collect::<Vec<_>>()
        .join("\n")
}

fn answer_line(n: usize) -> String {
    let labels = slot_labels();
    let word = if n <= 26 { "letter" } else { "label" };
    format!("Answer with one {word}: {}.", labels[..n].join(", "))
}

pub fn render_suffix(kind: &str, instructions: &str, options: &[String]) -> String {
    let header = if kind == "score" {
        "Rate the state:"
    } else {
        "Question:"
    };
    let instructions = if instructions.is_empty() {
        default_instructions(kind)
    } else {
        instructions
    };
    format!(
        "{header} {instructions}\n\nOptions:\n{}\n\n{}",
        options_block(options),
        answer_line(options.len())
    )
}

fn build_row(
    bundle: &crate::tokenize::TokenizerBundle,
    kind: &str,
    instructions: &str,
    state: &str,
    options: &[String],
    media: &[crate::tokenize::MediaItem],
) -> crate::tokenize::BuiltSequence {
    let text = format!(
        "{CHAT_PREFIX}{SYSTEM_PROMPT}{CHAT_MID}<state>\n{state}\n</state>\n\n{}{CHAT_SUFFIX}",
        render_suffix(kind, instructions, options)
    );
    let ids = bundle.encode_plain(&text);
    let slot = ids.len() - 1;
    let full_len = ids.len();
    let nopts = options.len();
    let prefix_len = bundle
        .encode_plain(&format!(
            "{CHAT_PREFIX}{SYSTEM_PROMPT}{CHAT_MID}<state>\n{state}\n</state>\n\n"
        ))
        .len();
    crate::tokenize::BuiltSequence {
        ids,
        markers: vec![slot],
        full_len,
        state_tokens: 0,
        nopts: vec![nopts],
        media: media.to_vec(),
        text,
        prefix_len,
    }
}

pub fn render_options(task: &str, options: &[String]) -> Vec<String> {
    if task == "noul" && options.is_empty() {
        return vec![
            format!("yes: {NOUL_DEFAULT_TRUE}"),
            format!("no: {NOUL_DEFAULT_FALSE}"),
        ];
    }
    options.to_vec()
}

pub fn build_rows(
    bundle: &crate::tokenize::TokenizerBundle,
    task: &str,
    question: &str,
    state: &str,
    options: &[String],
    media: &[crate::tokenize::MediaItem],
) -> Vec<crate::tokenize::BuiltSequence> {
    let options = render_options(task, options);
    vec![build_row(bundle, task, question, state, &options, media)]
}

pub fn prefix_token_len(bundle: &crate::tokenize::TokenizerBundle, state: &str) -> usize {
    bundle
        .encode_plain(&format!(
            "{CHAT_PREFIX}{SYSTEM_PROMPT}{CHAT_MID}<state>\n{state}\n</state>\n\n"
        ))
        .len()
}

pub fn temperature(table: &HashMap<String, f64>, kind: &str, k: usize) -> f64 {
    table
        .get(&format!("{kind}:{k}"))
        .or_else(|| table.get(kind))
        .copied()
        .unwrap_or(1.0)
}

fn normalized(p: &[f64]) -> Vec<f64> {
    let total: f64 = p.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / p.len() as f64; p.len()];
    }
    p.iter().map(|x| x / total).collect()
}

pub fn choice_confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let p = normalized(p);
    let uniform = 1.0 / p.len() as f64;
    let p_max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (p_max - uniform) / (1.0 - uniform)
}

pub fn score_confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let p = normalized(p);
    let mode = argmax(&p);
    let distance: f64 = p
        .iter()
        .enumerate()
        .map(|(i, x)| x * (i as f64 - mode as f64).abs())
        .sum();
    let center = (p.len() - 1) as f64 / 2.0;
    let uniform_mad: f64 =
        (0..p.len()).map(|i| (i as f64 - center).abs()).sum::<f64>() / p.len() as f64;
    (1.0 - distance / uniform_mad).max(0.0)
}

pub fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

pub fn calibrate(
    task: &str,
    logits: &[f64],
    _ctx: &CalibCtx,
    cfg: &CalibrationConfig,
) -> Calibrated {
    let CalibrationConfig::BranchTable(table) = cfg else {
        unreachable!("the branch-table schema is only constructed for this family")
    };
    let t = temperature(table, task, logits.len());
    let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
    let p = normalized(&softmax(&scaled));
    match task {
        "noul" => Calibrated {
            probability: Some(round6(p[0].clamp(0.0, 1.0))),
            probabilities: p.iter().map(|v| round6(*v)).collect(),
            confidence: round6(choice_confidence(&p)),
            choice: None,
            score: None,
            x_p_max: None,
            certainty: None,
            level_fit: None,
            fit_mass: None,
        },
        "score" => Calibrated {
            score: Some(round6(
                p.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>(),
            )),
            probabilities: p.iter().map(|v| round6(*v)).collect(),
            confidence: round6(score_confidence(&p)),
            choice: None,
            probability: None,
            x_p_max: None,
            certainty: None,
            level_fit: None,
            fit_mass: None,
        },
        _ => {
            let j = argmax(&p);
            Calibrated {
                probabilities: p.iter().map(|v| round6(*v)).collect(),
                choice: Some(j),
                confidence: round6(choice_confidence(&p)),
                score: None,
                probability: None,
                x_p_max: None,
                certainty: None,
                level_fit: None,
                fit_mass: None,
            }
        }
    }
}
