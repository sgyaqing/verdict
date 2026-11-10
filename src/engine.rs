use crate::replay::ReplayEngine;
use crate::scheduler::{BatchRunner, CollatedBatch};
use ndarray::{Array1, Array2, Array4};
use ort::session::Session;
use std::hash::{DefaultHasher, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EngineError(pub String);

#[derive(Debug, Clone, PartialEq)]
pub struct EngineRow {
    pub logits: Vec<f64>,
    pub expanded_len: Option<u32>,
}

fn plain_rows(rows: Vec<Vec<f64>>) -> Vec<EngineRow> {
    rows.into_iter()
        .map(|logits| EngineRow {
            logits,
            expanded_len: None,
        })
        .collect()
}
pub trait Engine: Send + Sync {
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<EngineRow>, EngineError>;

    fn media_capable(&self) -> bool {
        false
    }

    fn shape_buckets(&self) -> Vec<usize> {
        Vec::new()
    }

    fn health(&self) -> bool;
}

pub fn engine_runner(engine: Arc<dyn Engine>) -> BatchRunner {
    Arc::new(move |batch| engine.infer(batch).map_err(|err| err.0))
}

pub struct StubEngine {
    latency: Duration,
}
impl StubEngine {
    pub fn new(latency: Duration) -> Self {
        Self { latency }
    }
}

fn stub_score(seed: u64, j: usize) -> f64 {
    let h = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    (h >> 33) as f64 / u32::MAX as f64 * 6.0 - 3.0
}
impl Engine for StubEngine {
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<EngineRow>, EngineError> {
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency);
        }
        let mut rows = Vec::with_capacity(batch.b);
        for (i, meta) in batch.rows.iter().enumerate() {
            let real = &batch.input_ids[i * batch.l..i * batch.l + meta.real_len];
            let mut hasher = DefaultHasher::new();
            for id in real {
                hasher.write_i64(*id);
            }
            let seed = hasher.finish();
            let mut row: Vec<f64> = (0..meta.real_k).map(|j| stub_score(seed, j)).collect();
            row.resize(meta.real_k.max(batch.k), -1e4);
            rows.push(row);
        }
        Ok(plain_rows(rows))
    }
    fn health(&self) -> bool {
        true
    }
}

pub fn resolve_dylib(model_dir: &Path) -> Option<PathBuf> {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return None;
    }
    find_dylib(&model_dir.join("lib")).or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(find_dylib))
            .flatten()
    })
}
fn find_dylib(dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            let prefix = if cfg!(target_os = "windows") {
                "onnxruntime"
            } else {
                "libonnxruntime"
            };
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default();
            name.starts_with(prefix)
                && (ext == "dylib" || ext == "so" || (cfg!(target_os = "windows") && ext == "dll"))
        })
        .collect();
    candidates.sort_by_key(|p| p.file_name().map(|n| n.len()).unwrap_or(usize::MAX));
    candidates.into_iter().next()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}
impl DeviceKind {
    pub fn name(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Cuda => "cuda",
        }
    }
}

pub struct OrtEngine {
    sessions: Vec<(usize, Mutex<Session>)>,
    #[allow(dead_code)]
    model_path: PathBuf,

    dialect: crate::family::Family,

    sliding_window: usize,
}
impl OrtEngine {
    pub fn new(model_dir: &Path) -> Result<Self, EngineError> {
        Self::with_device(model_dir, DeviceKind::Cpu)
    }
    pub fn with_device(model_dir: &Path, device: DeviceKind) -> Result<Self, EngineError> {
        Self::with_device_and_threads(model_dir, device, 0)
    }

    pub fn with_device_and_threads(
        model_dir: &Path,
        device: DeviceKind,
        intra_threads: usize,
    ) -> Result<Self, EngineError> {
        let manifest = crate::manifest::read_manifest(model_dir)
            .map_err(|err| EngineError(err.to_string()))?;
        let mut fixed: Vec<(usize, String)> = match manifest.get("fixed_shapes") {
            Some(v) => {
                let obj = v.as_object().ok_or_else(|| {
                    EngineError(
                        "model.json 'fixed_shapes' must map padded lengths to ONNX file names"
                            .into(),
                    )
                })?;
                let out: Vec<(usize, String)> = obj
                    .iter()
                    .map(|(len, file)| {
                        let len: usize = len.parse().map_err(|_| {
                            EngineError(format!(
                                "fixed_shapes key {len:?} is not an integer length"
                            ))
                        })?;
                        let file = file.as_str().ok_or_else(|| {
                            EngineError(format!("fixed_shapes[{len}] must be an ONNX file name"))
                        })?;
                        Ok((len, file.to_owned()))
                    })
                    .collect::<Result<_, EngineError>>()?;
                out
            }
            None => Vec::new(),
        };
        fixed.sort();
        let graph = manifest
            .get("graph")
            .and_then(|g| g.as_str())
            .map(str::to_owned);
        if fixed.is_empty() && graph.is_none() {
            return Err(EngineError(
                "model.json has neither 'graph' nor 'fixed_shapes' (an ONNX file name is required)"
                    .into(),
            ));
        }
        let family = crate::family::Family::from_manifest(&manifest).map_err(EngineError)?;
        let sliding_window = manifest.get("sliding_window").and_then(|v| v.as_u64());
        let sliding_window = match (family, sliding_window) {
            (crate::family::Family::IndependentOptions, Some(v)) => v as usize,
            (crate::family::Family::IndependentOptions, None) => {
                return Err(EngineError(
                    "model.json is missing the 'sliding_window' field (the \
                     independent-options family needs it to build attention masks)"
                        .into(),
                ));
            }
            (_, Some(v)) => v as usize,
            (_, None) => 0,
        };
        let build_session = |file: &str| -> Result<Session, EngineError> {
            let builder = Session::builder().map_err(|err| EngineError(err.to_string()))?;
            let builder = if intra_threads > 0 {
                builder
                    .with_intra_threads(intra_threads)
                    .map_err(|err| EngineError(err.to_string()))?
            } else {
                builder
            };
            let mut builder = match device {
                DeviceKind::Cpu => builder,
                DeviceKind::Cuda => builder
                    .with_execution_providers([ort::execution_providers::CUDA::default()
                        .with_tf32(false)
                        .with_arena_extend_strategy(
                            ort::execution_providers::ArenaExtendStrategy::SameAsRequested,
                        )
                        .build()
                        .error_on_failure()])
                    .map_err(|err| {
                        EngineError(format!(
                            "failed to register the CUDA EP: {err} -- the CUDA provider could \
                             not be loaded; the CUDA runtime libraries on LD_LIBRARY_PATH must \
                             match this onnxruntime build's CUDA version"
                        ))
                    })?,
            };
            let model_path = model_dir.join(file);
            if !model_path.exists() {
                return Err(EngineError(format!(
                    "model file not found: {} (the real engine requires the ONNX graph; \
                     StubEngine is available for tokenization/calibration-only runs)",
                    model_path.display()
                )));
            }
            builder.commit_from_file(&model_path).map_err(|err| {
                let msg = err.to_string();
                if msg.contains("driver version is insufficient")
                    || msg.contains("CUDA failure 35")
                {
                    EngineError(format!(
                        "failed to load {}: the NVIDIA driver is too old for this build's CUDA \
                         stack ({msg}) -- upgrade the host driver or use the CPU image",
                        model_path.display()
                    ))
                } else {
                    EngineError(format!(
                        "failed to load {}: {err} (is ORT_DYLIB_PATH set / a dylib beside the model?)",
                        model_path.display()
                    ))
                }
            })
        };
        let (sessions, model_path) = if fixed.is_empty() {
            let g = graph.expect("presence checked above");
            let session = build_session(&g)?;
            (vec![(0, Mutex::new(session))], model_dir.join(&g))
        } else {
            let mut sessions = Vec::with_capacity(fixed.len());
            for (len, file) in &fixed {
                sessions.push((*len, Mutex::new(build_session(file)?)));
            }
            let first = model_dir.join(&fixed[0].1);
            (sessions, first)
        };
        Ok(Self {
            sessions,
            model_path,
            dialect: family,
            sliding_window,
        })
    }
}
impl Engine for OrtEngine {
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<EngineRow>, EngineError> {
        let (b, l, k) = (batch.b, batch.l, batch.k);
        if b == 0 {
            return Ok(Vec::new());
        }
        let shape_err = |what: &str| {
            EngineError(format!(
                "internal tensor shape mismatch building {what} ({b}x{l}/{k})"
            ))
        };
        let von_tensors = match self.dialect {
            crate::family::Family::IndependentOptions => Some(
                independent_options_tensors(&batch, b, l, k, self.sliding_window)
                    .map_err(EngineError)?,
            ),
            crate::family::Family::MarkerHead => None,
            crate::family::Family::LetterReadout | crate::family::Family::LetterBranches => None,
        };
        let letter_mask = matches!(
            self.dialect,
            crate::family::Family::LetterReadout | crate::family::Family::LetterBranches
        )
        .then(|| letter_causal_mask(&batch, b, l));
        let ids =
            Array2::from_shape_vec((b, l), batch.input_ids).map_err(|_| shape_err("input_ids"))?;
        let marker_pos = Array2::from_shape_vec((b, k), batch.marker_pos)
            .map_err(|_| shape_err("marker_pos"))?;
        let marker_mask = Array2::from_shape_vec((b, k), batch.marker_mask)
            .map_err(|_| shape_err("marker_mask"))?;
        let mut session = if self.sessions.len() == 1 {
            self.sessions[0]
                .1
                .lock()
                .expect("ort session mutex poisoned")
        } else {
            self.sessions
                .iter()
                .find(|(len, _)| *len == l)
                .ok_or_else(|| {
                    let served: Vec<String> =
                        self.sessions.iter().map(|(n, _)| n.to_string()).collect();
                    EngineError(format!(
                        "padded length {l} matches no fixed shape (served: {}) -- \
                         the request exceeds the largest served shape",
                        served.join("/")
                    ))
                })?
                .1
                .lock()
                .expect("ort session mutex poisoned")
        };
        let outputs = match self.dialect {
            crate::family::Family::MarkerHead => {
                let attention = Array2::from_shape_vec((b, l), batch.attention_mask)
                    .map_err(|_| shape_err("attention_mask"))?;
                let qtype =
                    Array1::from_shape_vec(b, batch.qtype).map_err(|_| shape_err("qtype"))?;
                let inputs: Vec<(&str, ort::session::SessionInputValue)> = vec![
                    (
                        "input_ids",
                        ort::value::Tensor::from_array(ids)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "attention_mask",
                        ort::value::Tensor::from_array(attention)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "marker_pos",
                        ort::value::Tensor::from_array(marker_pos)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "marker_mask",
                        ort::value::Tensor::from_array(marker_mask)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "qtype",
                        ort::value::Tensor::from_array(qtype)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                ];
                session
                    .run(inputs)
                    .map_err(|e| EngineError(format!("inference failed: {e}")))?
            }
            crate::family::Family::IndependentOptions => {
                let (position, full, sliding) = von_tensors.expect("built above for this dialect");
                let inputs: Vec<(&str, ort::session::SessionInputValue)> = vec![
                    (
                        "input_ids",
                        ort::value::Tensor::from_array(ids)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "full_mask",
                        ort::value::Tensor::from_array(full)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "sliding_mask",
                        ort::value::Tensor::from_array(sliding)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "position_ids",
                        ort::value::Tensor::from_array(position)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "marker_pos",
                        ort::value::Tensor::from_array(marker_pos)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "marker_mask",
                        ort::value::Tensor::from_array(marker_mask)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                ];
                session
                    .run(inputs)
                    .map_err(|e| EngineError(format!("inference failed: {e}")))?
            }
            crate::family::Family::LetterReadout | crate::family::Family::LetterBranches => {
                if k != 1 || batch.nopts.len() != b {
                    return Err(shape_err("letter slots (exactly one slot per row)"));
                }
                let slot_idx = marker_pos
                    .into_shape_with_order([b])
                    .map_err(|_| shape_err("slot_idx"))?;
                let slot_batch = Array1::from_iter((0..b).map(|i| i as i64));
                let causal_mask = letter_mask.expect("built above for this dialect");
                let nopts =
                    Array1::from_shape_vec(b, batch.nopts).map_err(|_| shape_err("nopts"))?;
                let inputs: Vec<(&str, ort::session::SessionInputValue)> = vec![
                    (
                        "input_ids",
                        ort::value::Tensor::from_array(ids)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "causal_mask",
                        ort::value::Tensor::from_array(causal_mask)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "slot_idx",
                        ort::value::Tensor::from_array(slot_idx)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "slot_batch",
                        ort::value::Tensor::from_array(slot_batch)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                    (
                        "nopts",
                        ort::value::Tensor::from_array(nopts)
                            .map_err(|e| EngineError(e.to_string()))?
                            .into(),
                    ),
                ];
                let outputs = session
                    .run(inputs)
                    .map_err(|e| EngineError(format!("inference failed: {e}")))?;
                let (_, logits) = outputs["logits"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| EngineError(format!("failed to read logits output: {e}")))?;
                const LETTER_WIDTH: usize = 255;
                if logits.len() != b * LETTER_WIDTH {
                    return Err(EngineError(format!(
                        "letter graph returned {} values for {} slots of width {LETTER_WIDTH}",
                        logits.len(),
                        b
                    )));
                }
                return Ok(plain_rows(
                    (0..b)
                        .map(|i| {
                            logits[i * LETTER_WIDTH..(i + 1) * LETTER_WIDTH]
                                .iter()
                                .map(|v| *v as f64)
                                .collect::<Vec<f64>>()
                        })
                        .collect::<Vec<_>>(),
                ));
            }
        };
        let (_, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| EngineError(format!("failed to read logits output: {e}")))?;
        if logits.len() != b * k {
            return Err(EngineError(format!(
                "graph returned {} logit slots for a {b}x{k} batch",
                logits.len()
            )));
        }
        Ok(plain_rows(
            (0..b)
                .map(|i| {
                    logits[i * k..(i + 1) * k]
                        .iter()
                        .map(|v| *v as f64)
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<_>>(),
        ))
    }
    fn health(&self) -> bool {
        true
    }
}

type VonTensors = (Array2<i64>, Array4<f32>, Array4<f32>);

fn letter_causal_mask(batch: &CollatedBatch, b: usize, l: usize) -> Array4<f32> {
    let mut mask = Array4::<f32>::from_elem((b, 1, l, l), f32::MIN);
    for row in 0..b {
        let last_key = batch.rows[row].real_len.saturating_sub(1);
        for q in 0..l {
            for key in 0..=q.min(last_key) {
                mask[[row, 0, q, key]] = 0.0;
            }
        }
    }
    mask
}

fn independent_options_tensors(
    batch: &CollatedBatch,
    b: usize,
    l: usize,
    k: usize,
    sliding_window: usize,
) -> Result<VonTensors, String> {
    let mut position = Array2::<i64>::from_shape_fn((b, l), |(_row, col)| col as i64);
    let mut full = Array4::<f32>::zeros((b, 1, l, l));
    let mut sliding = Array4::<f32>::zeros((b, 1, l, l));
    const NEG: f32 = f32::MIN;
    for row in 0..b {
        let real_len = batch.rows[row].real_len;
        let markers: Vec<usize> = batch.marker_pos[row * k..row * k + batch.rows[row].real_k]
            .iter()
            .map(|&m| m as usize)
            .collect();
        if markers.is_empty() || markers[0] >= real_len {
            return Err(format!(
                "row {row}: no marker positions inside the real ids"
            ));
        }
        let prefix_len = markers[0];
        let mut option_id = vec![-1i64; l];
        for (j, &start) in markers.iter().enumerate() {
            let end = markers
                .get(j + 1)
                .copied()
                .unwrap_or(real_len - 1)
                .min(real_len);
            for pos in start..end {
                option_id[pos] = j as i64;
                position[[row, pos]] = (prefix_len + (pos - start)) as i64;
            }
        }
        for q in 0..l {
            let oq = option_id[q];
            for key in 0..l {
                let ok = option_id[key];
                let base = (if oq == -1 {
                    ok == -1
                } else {
                    ok == -1 || ok == oq
                }) && key < real_len;
                let eye = q == key;
                if !base && !eye {
                    full[[row, 0, q, key]] = NEG;
                }
                if sliding_window > 0 {
                    let local =
                        (position[[row, q]] - position[[row, key]]).abs() <= sliding_window as i64;
                    if !(eye || base && local) {
                        sliding[[row, 0, q, key]] = NEG;
                    }
                }
            }
        }
    }
    Ok((position, full, sliding))
}

pub struct ReplayEngineAdapter {
    replay: Arc<ReplayEngine>,
    latency: Duration,
}
impl ReplayEngineAdapter {
    pub fn new(replay: Arc<ReplayEngine>, latency: Duration) -> Self {
        Self { replay, latency }
    }
}
impl Engine for ReplayEngineAdapter {
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<EngineRow>, EngineError> {
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency);
        }
        let mut rows = Vec::with_capacity(batch.b);
        for (i, meta) in batch.rows.iter().enumerate() {
            let real_ids: Vec<u32> = batch.input_ids[i * batch.l..i * batch.l + meta.real_len]
                .iter()
                .map(|id| *id as u32)
                .collect();
            let logits = self.replay.infer(&real_ids).ok_or_else(|| {
                EngineError(format!(
                    "replay table miss for row {i}: sequence not in golden corpus"
                ))
            })?;
            let mut row = logits.to_vec();
            row.resize(meta.real_k.max(batch.k), -1e4);
            rows.push(EngineRow {
                logits: row,
                expanded_len: self.replay.expanded_len(&real_ids),
            });
        }
        Ok(rows)
    }
    fn media_capable(&self) -> bool {
        self.replay.media_capable()
    }
    fn health(&self) -> bool {
        !self.replay.is_empty()
    }
}
