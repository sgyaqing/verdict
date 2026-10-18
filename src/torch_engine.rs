//! Torch runtime backend: the Engine implementation for packages that serve
//! through the generic torch executor (manifest `"runtime": "torch"`) instead
//! of an ONNX graph. The executor is a `python3.12` child process speaking
//! one JSON object per stdio line (v0.2 protocol; not binary); this engine
//! owns its lifecycle. Row semantics stay on the Rust side: the child only
//! runs the forward pass and returns letter logits.
//!
//! Failure model (shared with the ORT path): a forward error or a dead child
//! becomes an EngineError that the scheduler fans out to the whole batch.
//! There is no auto-restart in v0.2: once the child dies, health() reports
//! dead and infer keeps returning the error, so the failure is loud at the
//! /health surface instead of silently restarting mid-flight.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;

use crate::engine::{Engine, EngineError};
use crate::scheduler::CollatedBatch;

const EXECUTOR_SCRIPT: &str = include_str!("../executor/torch_exec.py");

/// Startup handshake bound: cold python + torch import + a ~4 GB checkpoint
/// load + first-call kernel compilation measured far below this on a
/// single-GPU CUDA host; a slower machine surfaces as a startup failure, not
/// a hang.
const READY_TIMEOUT: Duration = Duration::from_secs(600);

/// The letter-readout family's label table width (official MAX_OPTIONS).
const LETTER_WIDTH: usize = 255;

/// i64 vector -> base64 little-endian bytes (the wire form for ids, masks
/// and slot indices; JSON text of these is most of the per-batch IPC cost).
fn pack_i64(vals: &[i64]) -> String {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    B64.encode(bytes)
}

/// base64 little-endian f32 blob -> the executor's letter-logit rows.
/// `expected_rows` must match the batch or the reply is not this batch's.
fn unpack_logits(b64: &str, expected_rows: usize) -> Result<Vec<Vec<f64>>, EngineError> {
    let bytes = B64
        .decode(b64)
        .map_err(|err| EngineError(format!("executor logits are not valid base64: {err}")))?;
    if bytes.len() % 4 != 0 || bytes.is_empty() {
        return Err(EngineError(format!(
            "executor logits blob has {} bytes (not a multiple of 4)",
            bytes.len()
        )));
    }
    let mut flat = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.as_chunks::<4>().0 {
        flat.push(f32::from_le_bytes(*chunk) as f64);
    }
    if flat.len() % LETTER_WIDTH != 0 {
        return Err(EngineError(format!(
            "executor returned {} logits, not a multiple of {LETTER_WIDTH}",
            flat.len()
        )));
    }
    let rows: Vec<Vec<f64>> = flat.chunks(LETTER_WIDTH).map(<[f64]>::to_vec).collect();
    if rows.len() != expected_rows {
        return Err(EngineError(format!(
            "executor returned {} rows for a batch of {expected_rows}",
            rows.len()
        )));
    }
    Ok(rows)
}

/// One engine replica's executor child: the process handle, its stdin, and
/// the receiving end of the stdout pump. The pump thread owns the stdout so
/// `infer` never races a second reader, and a dead child closes the channel
/// (recv errors immediately instead of hanging).
struct TorchChild {
    process: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<String>,
    dead: bool,
}

pub struct TorchEngine {
    child: Mutex<TorchChild>,
}

impl TorchEngine {
    /// Production constructor: the model dir holds the manifest plus the
    /// downloaded checkpoint files (the model_source two-level fetch puts
    /// them beside model.json).
    pub fn new(model_dir: &Path) -> Result<Self, EngineError> {
        Self::with_checkpoint(model_dir, model_dir)
    }

    /// Test/dev constructor: the manifest comes from `model_dir`, the
    /// checkpoint from `ckpt_dir` (a run against an official snapshot sets
    /// VERDICT_TORCH_CKPT to it).
    pub fn with_checkpoint(model_dir: &Path, ckpt_dir: &Path) -> Result<Self, EngineError> {
        let manifest_raw =
            std::fs::read_to_string(model_dir.join("model.json")).map_err(|err| {
                EngineError(format!(
                    "failed to read {}: {err}",
                    model_dir.join("model.json").display()
                ))
            })?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_raw)
            .map_err(|err| EngineError(format!("invalid model.json: {err}")))?;
        let family = crate::family::Family::from_manifest(&manifest).map_err(EngineError)?;
        if family != crate::family::Family::LetterReadout {
            return Err(EngineError(
                "the torch runtime currently supports the letter-readout family only".to_string(),
            ));
        }
        if !ckpt_dir.join("config.json").is_file()
            || std::fs::read_dir(ckpt_dir)
                .map_err(|err| EngineError(format!("read {}: {err}", ckpt_dir.display())))?
                .filter_map(|e| e.ok())
                .all(|e| e.path().extension().and_then(|x| x.to_str()) != Some("safetensors"))
        {
            return Err(EngineError(format!(
                "checkpoint files not found in {} (config.json + safetensors are \
                 expected; the model download fetches them from the \
                 checkpoint repo declared in the manifest)",
                ckpt_dir.display()
            )));
        }
        let ckpt = ckpt_dir.to_string_lossy().into_owned();
        Self::spawn(EXECUTOR_SCRIPT, &[ckpt])
    }

    /// Shared spawn + handshake (the production constructor passes the
    /// embedded executor; tests pass a stub script). The child inherits this
    /// process's stderr so python tracebacks land in the serve log.
    pub fn spawn(script: &str, args: &[String]) -> Result<Self, EngineError> {
        let mut process = Command::new("python3.12")
            .arg("-c")
            .arg(script)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|err| {
                EngineError(format!(
                    "failed to spawn python3.12 for the torch executor: {err} \
                     (needs python3.12 with torch + transformers installed)"
                ))
            })?;
        let stdin = process.stdin.take().expect("stdin was piped at spawn");
        let stdout = process.stdout.take().expect("stdout was piped at spawn");
        let (tx, rx) = mpsc::channel::<String>();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let first = match rx.recv_timeout(READY_TIMEOUT) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = process.kill();
                return Err(EngineError(format!(
                    "the torch executor did not become ready within {}s \
                     (see its stderr above)",
                    READY_TIMEOUT.as_secs()
                )));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = process.kill();
                return Err(EngineError(
                    "the torch executor exited during startup (see its stderr above)".to_string(),
                ));
            }
        };
        let ready: serde_json::Value = serde_json::from_str(&first).map_err(|err| {
            EngineError(format!(
                "the torch executor's first output line was not valid JSON ({err}): {first}"
            ))
        })?;
        if ready.get("ready").and_then(|v| v.as_bool()) != Some(true) {
            let _ = process.kill();
            return Err(EngineError(format!(
                "the torch executor failed to initialize: {first}"
            )));
        }
        Ok(Self {
            child: Mutex::new(TorchChild {
                process,
                stdin,
                rx,
                dead: false,
            }),
        })
    }
}

impl Engine for TorchEngine {
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<Vec<f64>>, EngineError> {
        let (b, k) = (batch.b, batch.k);
        if k != 1 || batch.nopts.len() != b {
            return Err(EngineError(format!(
                "letter batches need exactly one slot per row (got k={k}, nopts={})",
                batch.nopts.len()
            )));
        }
        let mut slot_idx = Vec::with_capacity(b);
        for i in 0..b {
            if !batch.marker_mask[i] {
                return Err(EngineError(format!("row {i} has no slot marker")));
            }
            slot_idx.push(batch.marker_pos[i]);
        }
        let slot_batch: Vec<i64> = (0..b).map(|i| i as i64).collect();
        let request = serde_json::json!({
            "input_ids_b64": pack_i64(&batch.input_ids),
            "attention_mask_b64": pack_i64(&batch.attention_mask),
            "slot_idx_b64": pack_i64(&slot_idx),
            "slot_batch_b64": pack_i64(&slot_batch),
        });

        let mut child = self.child.lock().expect("torch child mutex poisoned");
        if child.dead {
            return Err(EngineError(
                "the torch executor is dead; restart the server".to_string(),
            ));
        }
        if let Err(err) = writeln!(child.stdin, "{request}").and_then(|_| child.stdin.flush()) {
            child.dead = true;
            return Err(EngineError(format!(
                "failed to write to the torch executor (broken pipe): {err}"
            )));
        }
        let line = match child.rx.recv() {
            Ok(line) => line,
            Err(_) => {
                child.dead = true;
                return Err(EngineError(
                    "the torch executor died mid-forward (see its stderr above)".to_string(),
                ));
            }
        };
        let reply: serde_json::Value = serde_json::from_str(&line).map_err(|err| {
            EngineError(format!("executor reply is not valid JSON ({err}): {line}"))
        })?;
        if let Some(err) = reply.get("error").and_then(|v| v.as_str()) {
            return Err(EngineError(format!("the torch executor failed: {err}")));
        }
        let logits_b64 = reply
            .get("logits_b64")
            .and_then(|v| v.as_str())
            .ok_or_else(|| EngineError(format!("executor reply has no logits_b64: {line}")))?;
        let mut rows = unpack_logits(logits_b64, b)?;
        if let Some(n_rows) = reply.get("n_rows").and_then(|v| v.as_u64())
            && n_rows as usize != b
        {
            return Err(EngineError(format!(
                "executor reports {n_rows} rows for a batch of {b}"
            )));
        }
        for (i, row) in rows.iter_mut().enumerate() {
            // Same contract as the letter graph: slots beyond the row's own
            // option count carry the -1e4 sentinel, never a real letter logit.
            let nopts = batch.nopts[i] as usize;
            for v in row.iter_mut().skip(nopts) {
                *v = -1e4;
            }
        }
        Ok(rows)
    }

    fn health(&self) -> bool {
        let mut child = self.child.lock().expect("torch child mutex poisoned");
        if child.dead {
            return false;
        }
        match child.process.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) | Err(_) => {
                child.dead = true;
                false
            }
        }
    }
}

impl Drop for TorchEngine {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.process.kill();
            let _ = child.process.wait();
        }
    }
}
