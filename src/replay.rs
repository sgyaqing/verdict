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
    #[serde(default)]
    rows: Option<Vec<GoldenRow>>,
    expected_token_ids: Option<Vec<u32>>,
    expected_logits: Option<Vec<f64>>,

    null_forward: Option<NullCase>,

    criteria_forward: Option<NullCase>,
}
#[derive(serde::Deserialize)]
struct GoldenRow {
    token_ids: Vec<u32>,
    logits: Vec<f64>,

    #[serde(default)]
    expanded_len: Option<u32>,
}
#[derive(serde::Deserialize)]
struct NullCase {
    token_ids: Vec<u32>,
    logits: Vec<f64>,
}
pub struct ReplayEngine {
    table: HashMap<Vec<u32>, Vec<f64>>,

    expanded: HashMap<Vec<u32>, u32>,
}
impl ReplayEngine {
    pub fn from_golden_dir(golden_dir: &Path) -> Result<Self, ReplayError> {
        let mut table = HashMap::new();
        let mut expanded = HashMap::new();
        let mut load = |path: std::path::PathBuf| -> Result<(), ReplayError> {
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
                if let Some(null) = case.null_forward {
                    table.insert(null.token_ids, null.logits);
                }
                if let Some(crit) = case.criteria_forward {
                    table.insert(crit.token_ids, crit.logits);
                }
                match case.rows {
                    Some(rows) => {
                        for row in rows {
                            if let Some(len) = row.expanded_len {
                                expanded.insert(row.token_ids.clone(), len);
                            }
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
            Ok(())
        };
        for task in ["choice", "score", "noul"] {
            load(golden_dir.join(format!("{task}.json")))?;
        }
        let media = golden_dir.join("media.json");
        if media.exists() {
            load(media)?;
        }
        Ok(Self { table, expanded })
    }
    pub fn len(&self) -> usize {
        self.table.len()
    }
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn infer<'a>(&'a self, ids: &[u32]) -> Option<&'a [f64]> {
        self.table.get(ids).map(|v| v.as_slice())
    }

    pub fn media_capable(&self) -> bool {
        !self.expanded.is_empty()
    }

    pub fn expanded_len(&self, ids: &[u32]) -> Option<u32> {
        self.expanded.get(ids).copied()
    }
}
