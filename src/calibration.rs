//! Extra-graph calibration: temperature + softmax + response assembly,
//! dispatched to the model's family recipe. Both recipes live under
//! `crate::family`; this module owns the manifest schemas and the shared
//! numeric helpers. Graded by the golden calibration tests (tests/golden_l2.rs).

use std::collections::HashMap;
use std::path::Path;

use crate::family::Family;

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("failed to read model.json: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid model.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("model.json is missing field {0}")]
    Missing(String),
    #[error("invalid model.json: {0}")]
    Family(String),
}

/// The band-table schema: `temperature[qtype]` fallback +
/// `temperature_by_options["task:band"]` overrides.
#[derive(Debug, Clone)]
pub struct BandTable {
    /// Per-qtype fallback, index = qtype (choice=0, score=1, noul=2).
    pub temperature: [f64; 3],
    /// Band-keyed overrides, e.g. "choice:3-5" -> 1.7601518630981445.
    pub temperature_by_options: HashMap<String, f64>,
}

/// The formula schema: a bounded linear temperature map over request
/// features (the independent-options family's shipped calibration), plus the noul commit
/// band and the zero-shot debias prior.
#[derive(Debug, Clone)]
pub struct FormulaMap {
    pub bias: f64,
    pub entropy: f64,
    pub log_tokens: f64,
    pub n_options: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Piecewise-affine noul commit rule: p >= 0.5 -> edge + slope*(p-0.5),
/// else (1-edge) - slope*(0.5-p). edge=0.5, slope=1 is the identity.
#[derive(Debug, Clone, Copy)]
pub struct NoulBand {
    pub edge: f64,
    pub slope: f64,
}

/// Zero-shot noul debiasing prior: correction = a*(l_true - l_false) + b
/// applied to the true-option logit before calibration.
#[derive(Debug, Clone, Copy)]
pub struct NoulPrior {
    pub a: f64,
    pub b: f64,
}

/// The by-type schema: one temperature per answer type (the letter-readout
/// family), falling back to the scalar for a type the manifest omits.
#[derive(Debug, Clone)]
pub struct ByTypeTable {
    pub temperature: f64,
    /// Keys are answer types ("choice"/"noul"/"score"); unknown keys fail
    /// loudly at load, like the official validator.
    pub temperature_by_type: HashMap<String, f64>,
}

impl ByTypeTable {
    pub fn temperature_for(&self, task: &str) -> f64 {
        self.temperature_by_type
            .get(task)
            .copied()
            .unwrap_or(self.temperature)
    }
}

/// One of the calibration schemas, selected by the manifest's family.
#[derive(Debug, Clone)]
pub enum CalibrationConfig {
    Bands(BandTable),
    Formula {
        map: FormulaMap,
        noul_band: NoulBand,
        /// Absent in the manifest falls back to the official pre-fit
        /// behavior (a=0.7, b=0), never to "no debiasing".
        noul_prior: NoulPrior,
    },
    ByType(ByTypeTable),
}

impl CalibrationConfig {
    pub fn family(&self) -> Family {
        match self {
            CalibrationConfig::Bands(_) => Family::MarkerHead,
            CalibrationConfig::Formula { .. } => Family::IndependentOptions,
            CalibrationConfig::ByType(_) => Family::LetterReadout,
        }
    }

    /// The zero-shot noul debias prior; None on the schemas whose families
    /// have no debiasing forward.
    pub fn noul_prior(&self) -> Option<NoulPrior> {
        match self {
            CalibrationConfig::Bands(_) | CalibrationConfig::ByType(_) => None,
            CalibrationConfig::Formula { noul_prior, .. } => Some(*noul_prior),
        }
    }

    pub fn load(model_dir: &Path) -> Result<Self, CalibrationError> {
        let raw = std::fs::read_to_string(model_dir.join("model.json"))?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)?;
        let family = Family::from_manifest(&cfg).map_err(CalibrationError::Family)?;
        match family {
            Family::MarkerHead => Ok(CalibrationConfig::Bands(load_bands(&cfg)?)),
            Family::IndependentOptions => load_formula(&cfg),
            Family::LetterReadout => Ok(CalibrationConfig::ByType(load_by_type(&cfg)?)),
        }
    }
}

fn load_bands(cfg: &serde_json::Value) -> Result<BandTable, CalibrationError> {
    let temp = cfg
        .get("temperature")
        .and_then(|v| v.as_array())
        .ok_or(CalibrationError::Missing("temperature".to_string()))?;
    if temp.len() != 3 {
        return Err(CalibrationError::Missing(
            "temperature (3 entries)".to_string(),
        ));
    }
    let mut temperature = [0.0f64; 3];
    for (i, v) in temp.iter().enumerate() {
        temperature[i] = v
            .as_f64()
            .ok_or(CalibrationError::Missing("temperature[i]".to_string()))?;
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
    Ok(BandTable {
        temperature,
        temperature_by_options,
    })
}

fn load_by_type(cfg: &serde_json::Value) -> Result<ByTypeTable, CalibrationError> {
    let temperature = cfg
        .get("temperature")
        .and_then(|v| v.as_f64())
        .ok_or(CalibrationError::Missing("temperature".to_string()))?;
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(CalibrationError::Family(
            "temperature must be a finite number > 0".into(),
        ));
    }
    let mut temperature_by_type = HashMap::new();
    if let Some(map) = cfg.get("temperature_by_type").and_then(|v| v.as_object()) {
        for (key, value) in map {
            if !matches!(key.as_str(), "choice" | "noul" | "score") {
                return Err(CalibrationError::Family(format!(
                    "temperature_by_type has the unknown key {key:?} (the keys are \
                     \"choice\", \"noul\" and \"score\")"
                )));
            }
            let t = value.as_f64().ok_or_else(|| {
                CalibrationError::Family(format!("temperature_by_type[{key:?}] is not a number"))
            })?;
            if !(t.is_finite() && t > 0.0) {
                return Err(CalibrationError::Family(format!(
                    "temperature_by_type[{key:?}] must be a finite number > 0"
                )));
            }
            temperature_by_type.insert(key.clone(), t);
        }
    }
    Ok(ByTypeTable {
        temperature,
        temperature_by_type,
    })
}

fn load_formula(cfg: &serde_json::Value) -> Result<CalibrationConfig, CalibrationError> {
    let field = |name: &str| -> Result<f64, CalibrationError> {
        cfg.get("calibration_formula")
            .and_then(|v| v.get(name))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| CalibrationError::Missing(format!("calibration_formula.{name}")))
    };
    let map = FormulaMap {
        bias: field("bias")?,
        entropy: field("entropy")?,
        log_tokens: field("log_tokens")?,
        n_options: field("n_options")?,
        lo: field("lo")?,
        hi: field("hi")?,
    };
    // No silent defaults here, unlike calibration_formula right above: a
    // package that loses or mistypes one of these must fail at load rather
    // than keep serving with a different calibration. Same `field` contract.
    let sub = |group: &'static str, key: &'static str| -> Result<f64, CalibrationError> {
        cfg.get(group)
            .and_then(|v| v.get(key))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| CalibrationError::Missing(format!("{group}.{key}")))
    };
    let noul_band = NoulBand {
        edge: sub("noul_band", "edge")?,
        slope: sub("noul_band", "slope")?,
    };
    let noul_prior = NoulPrior {
        a: sub("noul_prior", "a")?,
        b: sub("noul_prior", "b")?,
    };
    Ok(CalibrationConfig::Formula {
        map,
        noul_band,
        noul_prior,
    })
}

/// Request-scoped features a temperature recipe may read.
pub struct CalibCtx {
    /// Token count of the (fitted) state text, no special tokens -- the
    /// formula schema's log_tokens feature input.
    pub state_tokens: usize,
}

pub fn qtype_of(task: &str) -> Option<usize> {
    match task {
        "choice" => Some(0),
        "score" => Some(1),
        "noul" => Some(2),
        _ => None,
    }
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

/// round(p, 4), matching the official output quantization.
/// NOTE: Python round() is round-half-even, Rust f64::round is half-away-from-zero;
/// an exact .00005 tie is effectively unreachable on computed floats, and the
/// golden tolerance (1e-4 + eps) absorbs a single adjacent flip by design.
pub fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
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
    /// noul: committed P(true), round(4).
    pub probability: Option<f64>,
    /// score: expectation over the full-precision distribution, rounded per
    /// the family recipe.
    pub score: Option<f64>,
    /// Confidence served on the compat surface.
    pub confidence: f64,
    /// The letter-readout family's official answer extras (None elsewhere):
    /// the largest probability, the 1-minus-entropy certainty, and the
    /// isolated-score recombination fields.
    pub x_p_max: Option<f64>,
    pub certainty: Option<f64>,
    pub level_fit: Option<Vec<f64>>,
    pub fit_mass: Option<f64>,
}

/// The full official calibration, dispatched to the model's family recipe.
pub fn calibrate(
    task: &str,
    logits: &[f64],
    ctx: &CalibCtx,
    cfg: &CalibrationConfig,
) -> Calibrated {
    match cfg.family() {
        Family::MarkerHead => crate::family::marker_head::calibrate(task, logits, cfg),
        Family::IndependentOptions => {
            crate::family::independent_options::calibrate(task, logits, ctx, cfg)
        }
        Family::LetterReadout => crate::family::letter_readout::calibrate(task, logits, ctx, cfg),
    }
}
