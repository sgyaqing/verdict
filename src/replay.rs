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
    expected_token_ids: Vec<u32>,
    expected_logits: Vec<f64>,
}

pub struct ReplayEngine {
    table: HashMap<Vec<u32>, Vec<f64>>,
}

impl ReplayEngine {
    /// Load all three golden files from a golden directory.
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
                table.insert(case.expected_token_ids, case.expected_logits);
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
