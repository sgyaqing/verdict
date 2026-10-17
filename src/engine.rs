//! Engine abstraction: the seam behind the scheduler; OrtEngine (CPU + CUDA
//! execution providers) is the real implementation, StubEngine the test fake.
//!
//! `infer` is deliberately SYNCHRONOUS (a deviation from the usual
//! `#[async_trait] async fn infer` sketch): inference is blocking CPU/GPU work
//! that the scheduler already runs inside `spawn_blocking` (invariant 5), a
//! sync method keeps the trait dyn-safe without the async-trait dependency,
//! and `health()` mirrors TEI's `Infer::health` / vLLM's real liveness probe.

use std::hash::{DefaultHasher, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ndarray::{Array1, Array2};
use ort::session::Session;

use crate::replay::ReplayEngine;
use crate::scheduler::{BatchRunner, CollatedBatch};

/// Engine failure; the scheduler fans it out to every responder in the batch.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EngineError(pub String);

pub trait Engine: Send + Sync {
    /// Blocking batch inference. Contract: one row per input row, in batch
    /// order, each of length `batch.k` (graph-style, -1e4 at masked slots);
    /// the scheduler trims rows to each request's real_k.
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<Vec<f64>>, EngineError>;

    /// Liveness probe surfaced through /health.
    fn health(&self) -> bool;
}

/// Adapt any Engine into the scheduler's BatchRunner.
pub fn engine_runner(engine: Arc<dyn Engine>) -> BatchRunner {
    Arc::new(move |batch| engine.infer(batch).map_err(|err| err.0))
}

// ---------------------------------------------------------------- StubEngine

/// Deterministic fake engine: scores are a pure function of the row's token
/// ids (std DefaultHasher is documented deterministic), plus an optional
/// simulated latency. Production wires OrtEngine behind the same
/// `Arc<dyn Engine>` seam.
pub struct StubEngine {
    latency: Duration,
}

impl StubEngine {
    pub fn new(latency: Duration) -> Self {
        Self { latency }
    }
}

/// Deterministic pseudo-score in [-3, 3] from a row seed and marker index.
/// All integer math is wrapping: the second constant is ~1.38e19, so a plain
/// `j * CONST` overflows u64 for j >= 2.
fn stub_score(seed: u64, j: usize) -> f64 {
    let h = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((j as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    (h >> 33) as f64 / u32::MAX as f64 * 6.0 - 3.0
}

impl Engine for StubEngine {
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<Vec<f64>>, EngineError> {
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency); // runs on the blocking pool
        }
        let mut rows = Vec::with_capacity(batch.b);
        for (i, meta) in batch.rows.iter().enumerate() {
            let real = &batch.input_ids[i * batch.l..i * batch.l + meta.real_len];
            let mut hasher = DefaultHasher::new();
            for id in real {
                hasher.write_i64(*id);
            }
            let seed = hasher.finish();
            // Graph-style row: real scores then -1e4 sentinels up to batch.k.
            let mut row: Vec<f64> = (0..meta.real_k).map(|j| stub_score(seed, j)).collect();
            row.resize(batch.k, -1e4);
            rows.push(row);
        }
        Ok(rows)
    }

    fn health(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------- OrtEngine

/// Resolve the onnxruntime shared library for the load-dynamic strategy
/// (ort rc.13 has no programmatic loader; it reads ORT_DYLIB_PATH lazily on
/// the first API call). Priority: an already-set ORT_DYLIB_PATH (the caller's
/// explicit choice wins) > MODEL_DIR/lib/ > the executable's directory.
/// Returns the path to set, or None when the env var is already set.
///
/// The caller must apply it with `set_var` BEFORE the first ort call and,
/// because set_var races concurrent getenv, before spawning threads.
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
    // Prefer the shortest name: plain libonnxruntime.dylib over versioned copies.
    candidates.sort_by_key(|p| p.file_name().map(|n| n.len()).unwrap_or(usize::MAX));
    candidates.into_iter().next()
}

/// Execution device for the ort engine (engine-level mirror of the CLI flag).
/// `Cuda` registers the CUDA EP but does not guarantee it is *used*: ORT
/// places graph nodes at session load and silently keeps them on CPU when the
/// CUDA provider cannot initialize (no driver, no device, provider library
/// missing); `ORT_LOG=verbose` surfaces the placement ("All nodes placed on
/// [...]"). A real dual-EP check requires GPU hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

/// The real inference engine: onnxruntime, model loaded from
/// MODEL_DIR/`<graph from model.json>` (external weights resolve relative
/// to it). CPU EP is onnxruntime's default; `DeviceKind::Cuda` appends the
/// CUDA provider.
///
/// `Session::run` needs `&mut self`, so the session sits behind a Mutex; the
/// engine only ever runs inside `spawn_blocking` and the scheduler drives it
/// sequentially, so in practice the lock is uncontended (the overlap design
/// still serializes engine calls -- it overlaps COLLATION, not execution).
pub struct OrtEngine {
    session: Mutex<Session>,
    #[allow(dead_code)]
    model_path: PathBuf,
}

impl OrtEngine {
    pub fn new(model_dir: &Path) -> Result<Self, EngineError> {
        Self::with_device(model_dir, DeviceKind::Cpu)
    }

    pub fn with_device(model_dir: &Path, device: DeviceKind) -> Result<Self, EngineError> {
        Self::with_device_and_threads(model_dir, device, 0)
    }

    /// `intra_threads = 0` keeps ORT's default (one pool spanning all cores --
    /// right for a single session). A data-parallel pool passes cores/N per
    /// replica: N sessions each defaulting their own 96-thread pool measured
    /// as N-fold oversubscription scaling throughput DOWN with N; partition
    /// restores the concurrent-sessions shape.
    pub fn with_device_and_threads(
        model_dir: &Path,
        device: DeviceKind,
        intra_threads: usize,
    ) -> Result<Self, EngineError> {
        // The graph file name is model-family data read from the manifest --
        // never hardcode a family's file name.
        let manifest_path = model_dir.join("model.json");
        let manifest = std::fs::read_to_string(&manifest_path).map_err(|source| {
            EngineError(format!(
                "failed to read {}: {source}",
                manifest_path.display()
            ))
        })?;
        let graph = serde_json::from_str::<serde_json::Value>(&manifest)
            .ok()
            .and_then(|v| v.get("graph").and_then(|g| g.as_str()).map(str::to_owned))
            .ok_or_else(|| {
                EngineError("model.json is missing the 'graph' field (the ONNX file name)".into())
            })?;
        let model_path = model_dir.join(graph);
        if !model_path.exists() {
            return Err(EngineError(format!(
                "model file not found: {} (the real engine requires the ONNX graph; \
                 StubEngine is available for tokenization/calibration-only runs)",
                model_path.display()
            )));
        }
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
            // fp32 stays the compute dtype (golden discipline; fp16 would need
            // its own tolerance case). TF32 must be OFF: ORT's
            // CUDA EP enables it by default (`use_tf32 = 1`), which silently
            // rounds fp32 matmul inputs to a 10-bit mantissa on Ampere+ --
            // ~1e-3-scale drift, far beyond the 2e-4 logits tolerance.
            // Turing and older have no TF32 hardware and ignore the flag.
            DeviceKind::Cuda => builder
                .with_execution_providers([ort::execution_providers::CUDA::default()
                    .with_tf32(false)
                    .build()])
                .map_err(|err| EngineError(format!("failed to register the CUDA EP: {err}")))?,
        };
        let session = builder.commit_from_file(&model_path).map_err(|err| {
            EngineError(format!(
                "failed to load {}: {err} (is ORT_DYLIB_PATH set / a dylib beside the model?)",
                model_path.display()
            ))
        })?;
        Ok(Self {
            session: Mutex::new(session),
            model_path,
        })
    }
}

impl Engine for OrtEngine {
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<Vec<f64>>, EngineError> {
        let (b, l, k) = (batch.b, batch.l, batch.k);
        if b == 0 {
            return Ok(Vec::new());
        }
        let shape_err = |what: &str| {
            EngineError(format!(
                "internal tensor shape mismatch building {what} ({b}x{l}/{k})"
            ))
        };
        let ids =
            Array2::from_shape_vec((b, l), batch.input_ids).map_err(|_| shape_err("input_ids"))?;
        let attention = Array2::from_shape_vec((b, l), batch.attention_mask)
            .map_err(|_| shape_err("attention_mask"))?;
        let marker_pos = Array2::from_shape_vec((b, k), batch.marker_pos)
            .map_err(|_| shape_err("marker_pos"))?;
        let marker_mask = Array2::from_shape_vec((b, k), batch.marker_mask)
            .map_err(|_| shape_err("marker_mask"))?;
        let qtype = Array1::from_shape_vec(b, batch.qtype).map_err(|_| shape_err("qtype"))?;

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

        let mut session = self.session.lock().expect("ort session mutex poisoned");
        let outputs = session
            .run(inputs)
            .map_err(|e| EngineError(format!("inference failed: {e}")))?;
        // act_probs [B,2] is intentionally not read (semantics unpinned).
        let (_, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| EngineError(format!("failed to read logits output: {e}")))?;
        if logits.len() != b * k {
            return Err(EngineError(format!(
                "graph returned {} logit slots for a {b}x{k} batch",
                logits.len()
            )));
        }
        // Graph-style rows of length k (masked slots hold the -1e4 sentinel);
        // the scheduler trims to each request's real_k.
        Ok((0..b)
            .map(|i| {
                logits[i * k..(i + 1) * k]
                    .iter()
                    .map(|v| *v as f64)
                    .collect()
            })
            .collect())
    }

    fn health(&self) -> bool {
        // A constructed session is alive; a real probe (tiny forward) is future work.
        true
    }
}

// ------------------------------------------------------- ReplayEngine adapter

/// Wraps the golden ReplayEngine as an Engine: CI grades the full HTTP stack
/// without model weights. Rows are unpadded with
/// their real lengths before the lookup (golden keys are unpadded sequences).
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
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<Vec<f64>>, EngineError> {
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency);
        }
        let mut rows = Vec::with_capacity(batch.b);
        for (i, meta) in batch.rows.iter().enumerate() {
            // Collated ids are i64 (graph tensors); golden keys are u32.
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
            row.resize(batch.k, -1e4);
            rows.push(row);
        }
        Ok(rows)
    }

    fn health(&self) -> bool {
        !self.replay.is_empty()
    }
}
