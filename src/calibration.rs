use crate::family::Family;
use std::collections::HashMap;
use std::path::Path;
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

#[derive(Debug, Clone)]
pub struct BandTable {
    pub temperature: [f64; 3],

    pub temperature_by_options: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct FormulaMap {
    pub bias: f64,
    pub entropy: f64,
    pub log_tokens: f64,
    pub n_options: f64,
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NoulBand {
    pub edge: f64,
    pub slope: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NoulPrior {
    pub a: f64,
    pub b: f64,
}

#[derive(Debug, Clone)]
pub struct ByTypeTable {
    pub temperature: f64,

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

#[derive(Debug, Clone)]
pub enum CalibrationConfig {
    Bands(BandTable),
    Formula {
        map: FormulaMap,
        noul_band: NoulBand,

        noul_prior: NoulPrior,
    },
    ByType(ByTypeTable),

    BranchTable(HashMap<String, f64>),
}
impl CalibrationConfig {
    pub fn family(&self) -> Family {
        match self {
            CalibrationConfig::Bands(_) => Family::MarkerHead,
            CalibrationConfig::Formula { .. } => Family::IndependentOptions,
            CalibrationConfig::ByType(_) => Family::LetterReadout,
            CalibrationConfig::BranchTable(_) => Family::LetterBranches,
        }
    }

    pub fn noul_prior(&self) -> Option<NoulPrior> {
        match self {
            CalibrationConfig::Bands(_)
            | CalibrationConfig::ByType(_)
            | CalibrationConfig::BranchTable(_) => None,
            CalibrationConfig::Formula { noul_prior, .. } => Some(*noul_prior),
        }
    }
    pub fn load(model_dir: &Path) -> Result<Self, CalibrationError> {
        let cfg: serde_json::Value =
            crate::manifest::read_manifest(model_dir).map_err(|err| match err {
                crate::manifest::ManifestError::Io { source, .. } => CalibrationError::Io(source),
                crate::manifest::ManifestError::Parse(source) => CalibrationError::Parse(source),
                crate::manifest::ManifestError::Invalid(msg) => CalibrationError::Family(msg),
            })?;
        let family = Family::from_manifest(&cfg).map_err(CalibrationError::Family)?;
        match family {
            Family::MarkerHead => Ok(CalibrationConfig::Bands(load_bands(&cfg)?)),
            Family::IndependentOptions => load_formula(&cfg),
            Family::LetterReadout => Ok(CalibrationConfig::ByType(load_by_type(&cfg)?)),
            Family::LetterBranches => Ok(CalibrationConfig::BranchTable(load_branch_table(&cfg)?)),
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

fn load_branch_table(cfg: &serde_json::Value) -> Result<HashMap<String, f64>, CalibrationError> {
    let mut table = HashMap::new();
    let Some(entries) = cfg.get("temperatures").and_then(|v| v.as_object()) else {
        return Ok(table);
    };
    for (key, value) in entries {
        let kind = key.split(':').next().unwrap_or_default();
        if !matches!(kind, "choice" | "noul" | "score") {
            return Err(CalibrationError::Family(format!(
                "temperatures has the unknown key {key:?} (keys are \"choice\", \
                 \"noul\", \"score\" or \"<kind>:<k>\")"
            )));
        }
        let t = value.as_f64().ok_or_else(|| {
            CalibrationError::Family(format!("temperatures[{key:?}] is not a number"))
        })?;
        if !(t.is_finite() && t > 0.0) {
            return Err(CalibrationError::Family(format!(
                "temperatures[{key:?}] must be a finite number > 0"
            )));
        }
        table.insert(key.clone(), t);
    }
    Ok(table)
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

pub struct CalibCtx {
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

pub fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

pub fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

pub fn argmax(z: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in z.iter().enumerate() {
        if *v > z[best] {
            best = i;
        }
    }
    best
}

#[derive(Debug, Clone, PartialEq)]
pub struct Calibrated {
    pub probabilities: Vec<f64>,

    pub choice: Option<usize>,

    pub probability: Option<f64>,

    pub score: Option<f64>,

    pub confidence: f64,

    pub x_p_max: Option<f64>,
    pub certainty: Option<f64>,
    pub level_fit: Option<Vec<f64>>,
    pub fit_mass: Option<f64>,
}

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
        Family::LetterBranches => crate::family::letter_branches::calibrate(task, logits, ctx, cfg),
    }
}
