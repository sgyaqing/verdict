//! Replay engine: maps golden token sequences to golden raw logits so CI grades
//! the calibration and protocol layers without model weights.
//! The real engine (OrtEngine) sits behind the same seam.

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    #[error("failed to read golden file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid golden file {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(serde::Deserialize)]
struct GoldenDoc {
    cases: Vec<GoldenCase>,
}

#[derive(serde::Deserialize)]
struct GoldenCase {
    /// The letter-readout schema stores one entry per engine row (a score
    /// request is one row per isolated level); the singular fields remain
    /// the marker families' schema.
    #[serde(default)]
    rows: Option<Vec<GoldenRow>>,
    expected_token_ids: Option<Vec<u32>>,
    expected_logits: Option<Vec<f64>>,
    /// Zero-shot noul cases carry the context-free debias pass the pipeline
    /// also submits; the replay table must answer it or replay errors.
    null_forward: Option<NullCase>,
    /// The criteria-composed noul pair (letter family): the sequence the
    /// systemone adapter renders from Jev criteria, which no native case
    /// produces -- the replay table answers it for the mapping tests.
    criteria_forward: Option<NullCase>,
}

#[derive(serde::Deserialize)]
struct GoldenRow {
    token_ids: Vec<u32>,
    logits: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct NullCase {
    token_ids: Vec<u32>,
    logits: Vec<f64>,
}

pub struct ReplayEngine {
    table: HashMap<Vec<u32>, Vec<f64>>,
}

impl ReplayEngine {
    /// Load the three golden files (`choice.json`, `score.json`,
    /// `noul.json`) from a directory.
    pub fn from_golden_dir(golden_dir: &Path) -> Result<Self, ReplayError> {
        let mut table = HashMap::new();
        for task in ["choice", "score", "noul"] {
            let path = golden_dir.join(format!("{task}.json"));
            let raw = std::fs::read_to_string(&path).map_err(|source| ReplayError::Io {
                path: path.display().to_string(),
                source,
            })?;
            let doc: GoldenDoc =
                serde_json::from_str(&raw).map_err(|source| ReplayError::Parse {
                    path: path.display().to_string(),
                    source,
                })?;
            for case in doc.cases {
                // Duplicate sequences would mean duplicate corpus inputs; the
                // later entry wins, and identical inputs imply identical logits.
                if let Some(null) = case.null_forward {
                    table.insert(null.token_ids, null.logits);
                }
                if let Some(crit) = case.criteria_forward {
                    table.insert(crit.token_ids, crit.logits);
                }
                match case.rows {
                    Some(rows) => {
                        for row in rows {
                            table.insert(row.token_ids, row.logits);
                        }
                    }
                    None => {
                        table.insert(
                            case.expected_token_ids.expect("singular schema fields"),
                            case.expected_logits.expect("singular schema fields"),
                        );
                    }
                }
            }
        }
        Ok(Self { table })
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Raw logits for an exact token-sequence match.
    pub fn infer<'a>(&'a self, ids: &[u32]) -> Option<&'a [f64]> {
        self.table.get(ids).map(|v| v.as_slice())
    }
}
