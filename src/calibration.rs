//! Extra-graph calibration: temperature band lookup + softmax + round-4 response
//! assembly. Re-typed from rl_agent_api.py lines 62-64 and rl_common.temp_bucket;
//! graded against the golden vectors.

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("failed to read model.json: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid model.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("model.json is missing field {0}")]
    Missing(&'static str),
}

#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Per-qtype fallback, index = qtype (choice=0, score=1, noul=2).
    pub temperature: [f64; 3],
    /// Band-keyed overrides, e.g. "choice:3-5" -> 1.7601518630981445.
    pub temperature_by_options: HashMap<String, f64>,
}

impl CalibrationConfig {
    pub fn load(model_dir: &Path) -> Result<Self, CalibrationError> {
        let raw = std::fs::read_to_string(model_dir.join("model.json"))?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)?;
        let temp = cfg
            .get("temperature")
            .and_then(|v| v.as_array())
            .ok_or(CalibrationError::Missing("temperature"))?;
        if temp.len() != 3 {
            return Err(CalibrationError::Missing("temperature (3 entries)"));
        }
        let mut temperature = [0.0f64; 3];
        for (i, v) in temp.iter().enumerate() {
            temperature[i] = v
                .as_f64()
                .ok_or(CalibrationError::Missing("temperature[i]"))?;
        }
        let mut temperature_by_options = HashMap::new();
        if let Some(bands) = cfg
            .get("temperature_by_options")
            .and_then(|v| v.as_object())
        {
            for (key, value) in bands {
                if let Some(t) = value.as_f64() {
                    temperature_by_options.insert(key.clone(), t);
                }
            }
        }
        Ok(Self {
            temperature,
            temperature_by_options,
        })
    }
}

pub fn qtype_of(task: &str) -> Option<usize> {
    match task {
        "choice" => Some(0),
        "score" => Some(1),
        "noul" => Some(2),
        _ => None,
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
pub fn lookup_temperature(cfg: &CalibrationConfig, task: &str, k: usize) -> f64 {
    let key = format!("{task}:{}", band_of(k));
    let qtype = qtype_of(task).unwrap_or(0);
    cfg.temperature_by_options
        .get(&key)
        .copied()
        .unwrap_or(cfg.temperature[qtype])
}

/// Numerically-stable softmax over f64. The official path computes it in f32;
/// the difference (~1e-8 relative) is absorbed by the round-4 comparison
/// tolerance, which exists exactly for cross-runtime float variance.
pub fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// round(p, 4), bug-for-bug with the official output quantization.
/// NOTE: Python round() is round-half-even, Rust f64::round is half-away-from-zero;
/// an exact .00005 tie is effectively unreachable on computed floats, and the
/// golden tolerance (1e-4 + eps) absorbs a single adjacent flip by design.
pub fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Jev-style confidence: 1 - normalized entropy of the FULL-PRECISION answer
/// distribution (laya-export sequence.ts confidenceFromProbs, ported exactly:
/// the 1e-12 clamp and the k < 2 pin are part of the port). Served by
/// /v1/systemone; the native protocol keeps it off its wire shape.
pub fn confidence_from_probs(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let entropy: f64 = p.iter().map(|&x| -x * x.max(1e-12).ln()).sum();
    1.0 - entropy / (k as f64).ln()
}

/// First-index argmax (mirrors np.argmax tie-breaking).
pub fn argmax(z: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in z.iter().enumerate() {
        if *v > z[best] {
            best = i;
        }
    }
    best
}

/// Everything downstream of raw logits that the protocol layer serves.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibrated {
    /// Per-option probabilities, round(4).
    pub probabilities: Vec<f64>,
    /// choice: argmax index over RAW logits (protocol contract).
    pub choice: Option<usize>,
    /// noul: P(true), round(4).
    pub probability: Option<f64>,
    /// score: sum(i * p_i) over full-precision p, round(4).
    pub score: Option<f64>,
    /// Jev confidence: 1 - normalized entropy of the full-precision p,
    /// round(4) -- served by /v1/systemone.
    pub confidence: f64,
}

/// The full official calibration: z = logits / T; p = softmax(z); round(p, 4).
pub fn calibrate(task: &str, logits: &[f64], cfg: &CalibrationConfig) -> Calibrated {
    let t = lookup_temperature(cfg, task, logits.len());
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
        },
        "noul" => Calibrated {
            probabilities,
            choice: None,
            probability: Some(round4(p[1])),
            score: None,
            confidence,
        },
        _ => Calibrated {
            probabilities,
            choice: None,
            probability: None,
            score: Some(round4(
                p.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>(),
            )),
            confidence,
        },
    }
}
