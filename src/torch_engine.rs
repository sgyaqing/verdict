use crate::engine::{Engine, EngineError};
use crate::scheduler::CollatedBatch;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::mpsc;
use std::thread;
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

type Waiters = VecDeque<tokio::sync::oneshot::Sender<String>>;

struct TorchChild {
    process: Mutex<Child>,
    write_side: Mutex<(ChildStdin, Waiters)>,
    dead: OnceLock<String>,
}

impl TorchChild {
    fn is_dead(&self) -> bool {
        self.dead.get().is_some()
    }

    fn die(&self, why: &str) {
        if self.dead.set(why.to_string()).is_ok()
            && let Ok(mut side) = self.write_side.lock()
        {
            for waiter in side.1.drain(..) {
                let _ = waiter.send(String::new());
            }
        }
    }
}
pub struct TorchEngine {
    child: Mutex<std::sync::Arc<TorchChild>>,
    media_capable: bool,
    shape_buckets: Vec<usize>,
    graph_slot: usize,

    exit_on_death: bool,
}

fn spawn_child_ready(
    script: &str,
    args: &[String],
) -> Result<
    (
        std::sync::Arc<TorchChild>,
        bool,
        Vec<usize>,
        serde_json::Value,
    ),
    EngineError,
> {
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
    let shape_buckets = ready
        .get("l_buckets")
        .and_then(|v| v.as_array())
        .map(|xs| {
            xs.iter()
                .filter_map(|x| x.as_u64())
                .map(|n| n as usize)
                .collect()
        })
        .unwrap_or_default();
    let child = std::sync::Arc::new(TorchChild {
        process: Mutex::new(process),
        write_side: Mutex::new((stdin, Waiters::new())),
        dead: OnceLock::new(),
    });
    thread::spawn({
        let child = std::sync::Arc::clone(&child);
        move || route_replies(child, rx)
    });
    Ok((child, media_capable, shape_buckets, ready))
}

fn route_replies(child: std::sync::Arc<TorchChild>, rx: mpsc::Receiver<String>) {
    for line in rx {
        let waiter = match child.write_side.lock() {
            Ok(mut side) => side.1.pop_front(),
            Err(_) => return,
        };
        if let Some(waiter) = waiter {
            let _ = waiter.send(line);
        }
    }
    child.die("the torch executor closed its stdout");
}
fn spawn_child(script: &str, args: &[String]) -> Result<Child, EngineError> {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in script.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let dir = std::env::temp_dir();
    let path = dir.join(format!("verdict-executor-{hash:016x}.py"));
    if !path.exists() {
        let tmp = dir.join(format!(
            "verdict-executor-{hash:016x}.{}.tmp",
            std::process::id()
        ));
        std::fs::write(&tmp, script)
            .and_then(|_| std::fs::rename(&tmp, &path))
            .map_err(|err| EngineError(format!("materialize the torch executor: {err}")))?;
    }
    let mut cmd = Command::new("python3.12");
    cmd.arg(&path).args(args);
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
    let first = match rx.recv() {
        Ok(line) => line,
        Err(_) => {
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
        let (child, media_capable, shape_buckets, ready) = spawn_child_ready(script, args)?;
        let graph_slot = crate::graph_stats::register();
        crate::graph_stats::set_ready(graph_slot, &ready);
        Ok(Self {
            child: Mutex::new(child),
            media_capable,
            shape_buckets,
            graph_slot,
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
    child: &std::sync::Arc<TorchChild>,
    batch: &CollatedBatch,
    slot_batch: &[i64],
    slot_idx: &[i64],
    has_media: bool,
    slot: usize,
) -> Result<Vec<crate::engine::EngineRow>, EngineError> {
    let b = batch.b;
    let t0 = std::time::Instant::now();
    let build_ms;
    let prefix_len: Vec<i64> = batch.rows.iter().map(|r| r.prefix_len as i64).collect();
    let mut request = serde_json::json!({
        "input_ids_b64": pack_i64(&batch.input_ids),
        "attention_mask_b64": pack_i64(&batch.attention_mask),
        "slot_idx_b64": pack_i64(slot_idx),
        "slot_batch_b64": pack_i64(slot_batch),
        "prefix_len_b64": pack_i64(&prefix_len),
    });
    if has_media {
        request["media"] = media_value(batch);
        request["media_counts"] = serde_json::json!(batch.media_counts);
        request["row_texts"] = serde_json::json!(batch.row_texts);
    }
    let reply_rx = {
        let mut side = child.write_side.lock().expect("torch child mutex poisoned");
        if child.is_dead() {
            return Err(EngineError(
                "the torch executor is dead; restart the server".to_string(),
            ));
        }
        let (tx, reply_rx) = tokio::sync::oneshot::channel();
        side.1.push_back(tx);
        let stdin = &mut side.0;
        if let Err(err) = writeln!(stdin, "{request}").and_then(|_| stdin.flush()) {
            side.1.pop_back();
            drop(side);
            child.die(&format!(
                "failed to write to the torch executor (broken pipe): {err}"
            ));
            return Err(EngineError(format!(
                "failed to write to the torch executor (broken pipe): {err}"
            )));
        }
        build_ms = t0.elapsed().as_secs_f64() * 1e3;
        reply_rx
    };
    let line = match reply_rx.blocking_recv() {
        Ok(line) if !line.is_empty() => line,
        _ => {
            return Err(EngineError(
                "the torch executor died mid-forward (see its stderr above)".to_string(),
            ));
        }
    };
    let reply_ms = t0.elapsed().as_secs_f64() * 1e3;
    let reply: serde_json::Value = serde_json::from_str(&line)
        .map_err(|err| EngineError(format!("executor reply is not valid JSON ({err}): {line}")))?;
    if std::env::var_os("VERDICT_BATCH_PROFILE").is_some() {
        eprintln!(
            "[rust-prof] b={b} build={build_ms:.2} reply={reply_ms:.2} parse={:.2}",
            t0.elapsed().as_secs_f64() * 1e3 - reply_ms
        );
    }
    if let Some(err) = reply.get("error").and_then(|v| v.as_str()) {
        return Err(EngineError(format!("the torch executor failed: {err}")));
    }
    crate::graph_stats::record(slot, &reply);
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
    fn shape_buckets(&self) -> Vec<usize> {
        self.shape_buckets.clone()
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
        let child = self
            .child
            .lock()
            .expect("torch child mutex poisoned")
            .clone();
        if child.is_dead() {
            self.on_child_death("the torch executor is dead");
            return Err(EngineError(
                "the torch executor is dead; the server is shutting down".to_string(),
            ));
        }
        let outcome = infer_json(
            &child,
            &batch,
            &slot_batch,
            &slot_idx,
            has_media,
            self.graph_slot,
        );
        if let Err(err) = &outcome
            && child.is_dead()
        {
            self.on_child_death(&format!("a request killed the torch executor: {err}"));
        }
        outcome
    }
    fn health(&self) -> bool {
        let child = self
            .child
            .lock()
            .expect("torch child mutex poisoned")
            .clone();
        if child.is_dead() {
            self.on_child_death("the torch executor is dead");
            return false;
        }
        let status = child
            .process
            .lock()
            .expect("torch child mutex poisoned")
            .try_wait();
        match status {
            Ok(None) => true,
            Ok(Some(status)) => {
                child.die(&format!("the torch executor exited ({status})"));
                self.on_child_death(&format!("the torch executor exited ({status})"));
                false
            }
            Err(err) => {
                child.die(&format!("cannot observe the torch executor: {err}"));
                self.on_child_death(&format!("cannot observe the torch executor: {err}"));
                false
            }
        }
    }
}
impl Drop for TorchEngine {
    fn drop(&mut self) {
        if let Ok(child) = self.child.lock()
            && let Ok(mut proc) = child.process.lock()
        {
            let _ = proc.kill();
            let _ = proc.wait();
        }
    }
}
