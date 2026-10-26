use crate::engine::{Engine, EngineError};
use crate::scheduler::CollatedBatch;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
const EXECUTOR_SCRIPT: &str = include_str!("../executor/torch_exec.py");

pub fn executor_sha() -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in EXECUTOR_SCRIPT.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}
fn executor_script() -> Result<String, EngineError> {
    match std::env::var("VERDICT_EXECUTOR_PATH") {
        Ok(path) => std::fs::read_to_string(&path)
            .map_err(|err| EngineError(format!("read VERDICT_EXECUTOR_PATH {path}: {err}"))),
        Err(std::env::VarError::NotPresent) => Ok(EXECUTOR_SCRIPT.to_string()),
        Err(std::env::VarError::NotUnicode(_)) => Err(EngineError(
            "VERDICT_EXECUTOR_PATH is not valid UTF-8".to_string(),
        )),
    }
}

const READY_TIMEOUT: Duration = Duration::from_secs(600);

fn pack_i64(vals: &[i64]) -> String {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    B64.encode(bytes)
}
fn media_value(batch: &CollatedBatch) -> serde_json::Value {
    let frame = |f: &crate::tokenize::MediaFrame| {
        serde_json::json!({
            "format": f.format,
            "data_b64": B64.encode(&f.data),
        })
    };
    serde_json::json!(
        batch
            .media
            .iter()
            .map(|m| match m {
                crate::tokenize::MediaItem::Image(f) => serde_json::json!({
                    "kind": "image",
                    "format": f.format,
                    "data_b64": B64.encode(&f.data),
                }),
                crate::tokenize::MediaItem::Video { frames, fps } => {
                    serde_json::json!({
                        "kind": "video",
                        "fps": fps,
                        "frames": frames.iter().map(&frame).collect::<Vec<_>>(),
                    })
                }
            })
            .collect::<Vec<_>>()
    )
}

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
    if flat.len() % expected_rows != 0 {
        return Err(EngineError(format!(
            "executor returned {} logits for {expected_rows} rows (not divisible)",
            flat.len()
        )));
    }
    let width = flat.len() / expected_rows;
    let rows: Vec<Vec<f64>> = flat.chunks(width).map(<[f64]>::to_vec).collect();
    if rows.len() != expected_rows {
        return Err(EngineError(format!(
            "executor returned {} rows for a batch of {expected_rows}",
            rows.len()
        )));
    }
    Ok(rows)
}
fn finish_rows(
    mut rows: Vec<Vec<f64>>,
    batch: &CollatedBatch,
    expanded: Vec<Option<u32>>,
) -> Vec<crate::engine::EngineRow> {
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter_mut().enumerate() {
        let nopts = batch.nopts[i] as usize;
        for v in row.iter_mut().skip(nopts) {
            *v = -1e4;
        }
        out.push(crate::engine::EngineRow {
            logits: std::mem::take(row),
            expanded_len: expanded[i],
        });
    }
    out
}

struct TorchChild {
    process: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<String>,
    dead: bool,
}
pub struct TorchEngine {
    child: Mutex<TorchChild>,
    media_capable: bool,

    exit_on_death: bool,
}

fn spawn_child_ready(script: &str, args: &[String]) -> Result<(TorchChild, bool), EngineError> {
    let mut process = spawn_child(script, args)?;
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
    let ready = wait_ready(&mut process, &rx)?;
    let media_capable = ready
        .get("multimodal")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Ok((
        TorchChild {
            process,
            stdin,
            rx,
            dead: false,
        },
        media_capable,
    ))
}
fn spawn_child(script: &str, args: &[String]) -> Result<Child, EngineError> {
    let mut cmd = Command::new("python3.12");
    cmd.arg("-c").arg(script).args(args);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| {
            EngineError(format!(
                "failed to spawn python3.12 for the torch executor: {err} \
                 (needs python3.12 with torch + transformers installed)"
            ))
        })
}
fn wait_ready(
    process: &mut Child,
    rx: &mpsc::Receiver<String>,
) -> Result<serde_json::Value, EngineError> {
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
    Ok(ready)
}
impl TorchEngine {
    pub fn new(model_dir: &Path) -> Result<Self, EngineError> {
        Self::with_device(model_dir, model_dir, "cuda")
    }

    pub fn with_checkpoint(model_dir: &Path, ckpt_dir: &Path) -> Result<Self, EngineError> {
        Self::with_device(model_dir, ckpt_dir, "cuda")
    }

    pub fn with_device(
        model_dir: &Path,
        ckpt_dir: &Path,
        device: &str,
    ) -> Result<Self, EngineError> {
        let manifest = crate::manifest::read_manifest(model_dir)
            .map_err(|err| EngineError(err.to_string()))?;
        let family = crate::family::Family::from_manifest(&manifest).map_err(EngineError)?;
        if !matches!(
            family,
            crate::family::Family::LetterReadout | crate::family::Family::LetterBranches
        ) {
            return Err(EngineError(
                "the torch runtime supports the letter families only (letter-readout, \
                 letter-branches)"
                    .to_string(),
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
        let manifest_arg = serde_json::to_string(&manifest)
            .map_err(|err| EngineError(format!("serialize manifest for the executor: {err}")))?;
        let script = executor_script()?;
        Self::spawn_with_policy(&script, &[device.to_string(), ckpt, manifest_arg], true)
    }

    pub fn spawn(script: &str, args: &[String]) -> Result<Self, EngineError> {
        Self::spawn_with_policy(script, args, false)
    }
    fn spawn_with_policy(
        script: &str,
        args: &[String],
        exit_on_death: bool,
    ) -> Result<Self, EngineError> {
        let (child, media_capable) = spawn_child_ready(script, args)?;
        Ok(Self {
            child: Mutex::new(child),
            media_capable,
            exit_on_death,
        })
    }

    fn on_child_death(&self, why: &str) {
        eprintln!("[verdict] ERROR {why}");
        if self.exit_on_death {
            eprintln!(
                "[verdict] ERROR exiting: the torch executor is gone and this server cannot \
                 serve without it"
            );
            std::process::exit(1);
        }
    }
}
impl TorchEngine {
    pub fn loaded_media(&self) -> bool {
        self.media_capable
    }
}
fn infer_json(
    child: &mut TorchChild,
    batch: &CollatedBatch,
    slot_batch: &[i64],
    slot_idx: &[i64],
    has_media: bool,
) -> Result<Vec<crate::engine::EngineRow>, EngineError> {
    let b = batch.b;
    let mut request = serde_json::json!({
        "input_ids_b64": pack_i64(&batch.input_ids),
        "attention_mask_b64": pack_i64(&batch.attention_mask),
        "slot_idx_b64": pack_i64(slot_idx),
        "slot_batch_b64": pack_i64(slot_batch),
    });
    if has_media {
        request["media"] = media_value(batch);
        request["media_counts"] = serde_json::json!(batch.media_counts);
        request["row_texts"] = serde_json::json!(batch.row_texts);
    }
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
    let reply: serde_json::Value = serde_json::from_str(&line)
        .map_err(|err| EngineError(format!("executor reply is not valid JSON ({err}): {line}")))?;
    if let Some(err) = reply.get("error").and_then(|v| v.as_str()) {
        return Err(EngineError(format!("the torch executor failed: {err}")));
    }
    if reply.get("eager").and_then(|v| v.as_bool()) == Some(true) {
        metrics::counter!("verdict_torch_eager_requests_total").increment(1);
    }
    let logits_b64 = reply
        .get("logits_b64")
        .and_then(|v| v.as_str())
        .ok_or_else(|| EngineError(format!("executor reply has no logits_b64: {line}")))?;
    let rows = unpack_logits(logits_b64, b)?;
    if let Some(n_rows) = reply.get("n_rows").and_then(|v| v.as_u64())
        && n_rows as usize != b
    {
        return Err(EngineError(format!(
            "executor reports {n_rows} rows for a batch of {b}"
        )));
    }
    let expanded: Vec<Option<u32>> = if has_media {
        let lens = reply
            .get("expanded_lens")
            .and_then(|v| v.as_array())
            .ok_or_else(|| EngineError("media batch reply has no expanded_lens".to_string()))?;
        lens.iter().map(|v| v.as_u64().map(|n| n as u32)).collect()
    } else {
        vec![None; b]
    };
    Ok(finish_rows(rows, batch, expanded))
}
impl Engine for TorchEngine {
    fn media_capable(&self) -> bool {
        self.media_capable
    }
    fn infer(&self, batch: CollatedBatch) -> Result<Vec<crate::engine::EngineRow>, EngineError> {
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
        let has_media = !batch.media.is_empty();
        let mut child = self.child.lock().expect("torch child mutex poisoned");
        if child.dead {
            self.on_child_death("the torch executor is dead");
            return Err(EngineError(
                "the torch executor is dead; the server is shutting down".to_string(),
            ));
        }
        match infer_json(&mut child, &batch, &slot_batch, &slot_idx, has_media) {
            Err(err) if child.dead => {
                self.on_child_death(&format!("a request killed the torch executor: {err}"));
                Err(err)
            }
            other => other,
        }
    }
    fn health(&self) -> bool {
        let mut child = self.child.lock().expect("torch child mutex poisoned");
        if child.dead {
            self.on_child_death("the torch executor is dead");
            return false;
        }
        match child.process.try_wait() {
            Ok(None) => true,
            Ok(Some(status)) => {
                child.dead = true;
                self.on_child_death(&format!("the torch executor exited ({status})"));
                false
            }
            Err(err) => {
                child.dead = true;
                self.on_child_death(&format!("cannot observe the torch executor: {err}"));
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
