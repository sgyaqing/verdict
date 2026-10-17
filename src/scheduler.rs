//! Dynamic batching scheduler: bounded queue + window batching +
//! mixed-task flow + ragged-K collate + padded-token budget admission.
//!
//! Reference patterns:
//! - TEI `core/src/queue.rs` `queue_blocking_task`: padded-cost admission
//!   `max(max_length, entry_len) * (n + 1) > budget` -> the entry stays queued
//!   (never dropped); entries whose responder was dropped (client gone) are
//!   skipped at batch time.
//! - TEI `core/src/infer.rs` `backend_task`: a batch error propagates to EVERY
//!   responder in that batch; the head of a batch is always admissible so the
//!   budget can never starve a request.
//! - vLLM `v1/core/sched/scheduler.py` `schedule`: token budget decremented
//!   across admissions each step.
//!
//! Five invariants (each verified
//! under mutation):
//! 1. every submitted request eventually receives a response (success or error)
//! 2. responses map back to their own request, never crossed
//! 3. a full queue rejects immediately instead of dropping or blocking
//! 4. shutdown drains: queued requests are processed, none lost
//! 5. inference runs on the blocking pool; the async reactor never stalls

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::timeout;

use crate::tokenize::{BuiltSequence, Layout};

/// Raw (uncalibrated) per-option logits; calibration happens after the scheduler.
pub type RawLogits = Vec<f64>;

#[derive(Debug, Clone, thiserror::Error)]
pub enum SchedulerError {
    #[error("queue is full, try again later")]
    QueueFull,
    #[error("scheduler is shutting down")]
    ShuttingDown,
    #[error("engine error: {0}")]
    Engine(String),
    #[error("invalid scheduler configuration: {0}")]
    Config(String),
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Bounded intake capacity; a submit beyond it is rejected (invariant 3).
    pub queue_cap: usize,
    /// Requests per batch at most.
    pub max_batch_size: usize,
    /// After the first arrival, wait up to this long for the batch to fill.
    pub batch_window: Duration,
    /// Padded-token budget per batch: `batch_max_len * n_requests` (the vLLM
    /// `max_num_batched_tokens` semantics for a padded model).
    pub token_budget: usize,
}

impl SchedulerConfig {
    /// Defaults derived the way the CLI wires them: window from `batch_window_ms`,
    /// budget from `max_batch_size * max_len` so every single (at most max_len)
    /// request always fits a fresh batch.
    pub fn new(
        queue_cap: usize,
        max_batch_size: usize,
        batch_window_ms: u64,
        max_len: usize,
    ) -> Self {
        Self {
            queue_cap,
            max_batch_size,
            batch_window: Duration::from_millis(batch_window_ms),
            token_budget: max_batch_size * max_len,
        }
    }

    pub fn validate(&self, max_len: usize) -> Result<(), String> {
        if self.queue_cap == 0 {
            return Err("queue_cap must be positive".to_string());
        }
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be positive".to_string());
        }
        // Otherwise a max-length request could never be admitted -> starvation.
        if self.token_budget < max_len {
            return Err(format!(
                "token_budget ({}) must be >= max_len ({}) so a single request always fits",
                self.token_budget, max_len
            ));
        }
        Ok(())
    }
}

/// One admitted request flowing through the pipeline.
pub struct ScheduledRequest {
    pub task: String,
    pub built: BuiltSequence,
    pub respond: oneshot::Sender<Result<RawLogits, SchedulerError>>,
    /// vLLM-aligned priority: smaller value = admitted earlier (default 0).
    pub priority: i64,
    /// Global arrival sequence assigned at submit; the ordering tiebreak
    /// (vLLM compares arrival_time then request_id; seq is both in one).
    pub seq: u64,
}

/// Exactly the ONNX graph's input tensors, flat row-major. `l` is the batch max
/// REAL length (pad-to-batch-max); `k` is the batch max marker count. The graph
/// writes -1e4 into masked marker slots; rows are trimmed to their real_k before
/// responses leave, so sentinels never reach a caller.
#[derive(Debug, Clone, PartialEq)]
pub struct CollatedBatch {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub marker_pos: Vec<i64>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<i64>,
    pub b: usize,
    pub l: usize,
    pub k: usize,
    /// Per-row real shapes (post-padding truth) for result splitting.
    pub rows: Vec<RowMeta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMeta {
    pub real_len: usize,
    pub real_k: usize,
}

/// Blocking batch inference. Runs ONLY inside `spawn_blocking` (invariant 5).
/// Contract: returns one row per input row, in batch order, each of length `k`
/// (the graph emits -1e4 at masked slots, which the scheduler trims).
pub type BatchRunner = Arc<dyn Fn(CollatedBatch) -> Result<Vec<Vec<f64>>, String> + Send + Sync>;

/// Right-pad every row to the batch max length, pad marker slots to the batch
/// max marker count (marker_pos 0 / marker_mask false), one qtype per row.
pub fn collate(items: &[ScheduledRequest], layout: &Layout) -> CollatedBatch {
    let b = items.len();
    let l = items.iter().map(|r| r.built.ids.len()).max().unwrap_or(0);
    let k = items
        .iter()
        .map(|r| r.built.markers.len())
        .max()
        .unwrap_or(0);
    let mut input_ids = Vec::with_capacity(b * l);
    let mut attention_mask = Vec::with_capacity(b * l);
    let mut marker_pos = vec![0i64; b * k];
    let mut marker_mask = vec![false; b * k];
    let mut qtype = Vec::with_capacity(b);
    let mut rows = Vec::with_capacity(b);
    for (i, req) in items.iter().enumerate() {
        let real_len = req.built.ids.len();
        input_ids.extend(req.built.ids.iter().map(|id| *id as i64));
        input_ids.extend(std::iter::repeat_n(layout.pad_id as i64, l - real_len));
        attention_mask.extend(std::iter::repeat_n(1i64, real_len));
        attention_mask.extend(std::iter::repeat_n(0i64, l - real_len));
        for (j, marker) in req.built.markers.iter().enumerate() {
            marker_pos[i * k + j] = *marker as i64;
            marker_mask[i * k + j] = true;
        }
        qtype.push(crate::calibration::qtype_of(&req.task).unwrap_or(0) as i64);
        rows.push(RowMeta {
            real_len,
            real_k: req.built.markers.len(),
        });
    }
    CollatedBatch {
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        b,
        l,
        k,
        rows,
    }
}

/// Batch admission policy seam; FCFS is the default, `priority` optional.
pub trait SchedulingPolicy: Send + Sync + 'static {
    /// Move the requests admitted into the next batch out of `pending`;
    /// leftovers stay queued. Must admit at least one open request whenever
    /// `pending` holds one (otherwise the loop would stall).
    fn admit(
        &self,
        pending: &mut VecDeque<ScheduledRequest>,
        max_batch_size: usize,
        token_budget: usize,
    ) -> Vec<ScheduledRequest>;
}

/// Front-of-queue admission scan shared by the policies: admit while the
/// padded cost `max_l * n` stays within the token budget. Entries whose
/// responder is already closed (client disconnected) are skipped, not
/// counted. The FIRST entry is always admitted (no stall on a single
/// over-budget request).
fn admit_in_front_order(
    pending: &mut VecDeque<ScheduledRequest>,
    max_batch_size: usize,
    token_budget: usize,
) -> Vec<ScheduledRequest> {
    let mut batch: Vec<ScheduledRequest> = Vec::new();
    let mut max_l = 0usize;
    while let Some(req) = pending.pop_front() {
        if req.respond.is_closed() {
            continue; // nobody is waiting; skip instead of wasting a batch slot
        }
        let candidate_l = max_l.max(req.built.ids.len());
        let cost_fits = batch.is_empty() || candidate_l * (batch.len() + 1) <= token_budget;
        if batch.len() < max_batch_size && cost_fits {
            max_l = candidate_l;
            batch.push(req);
        } else {
            pending.push_front(req); // stays queued for the next batch, never dropped
            break;
        }
    }
    batch
}

/// First-come-first-served (default).
pub struct Fcfs;

impl SchedulingPolicy for Fcfs {
    fn admit(
        &self,
        pending: &mut VecDeque<ScheduledRequest>,
        max_batch_size: usize,
        token_budget: usize,
    ) -> Vec<ScheduledRequest> {
        admit_in_front_order(pending, max_batch_size, token_budget)
    }
}

/// Priority policy (vLLM `policy=priority` semantics, request_queue.py +
/// request.py: smaller `priority` first, earlier arrival as tiebreak).
/// Sorting is O(n log n) per window, n bounded by queue_cap.
pub struct Priority;

impl SchedulingPolicy for Priority {
    fn admit(
        &self,
        pending: &mut VecDeque<ScheduledRequest>,
        max_batch_size: usize,
        token_budget: usize,
    ) -> Vec<ScheduledRequest> {
        // seq is unique, so (priority, seq) is a total order.
        pending
            .make_contiguous()
            .sort_by_key(|r| (r.priority, r.seq));
        admit_in_front_order(pending, max_batch_size, token_budget)
    }
}

pub struct SchedulerHandle {
    tx: mpsc::Sender<ScheduledRequest>,
    next_seq: AtomicU64,
    join_batching: JoinHandle<()>,
    join_backend: JoinHandle<()>,
}

impl SchedulerHandle {
    /// Enqueue without ever blocking or dropping: a full queue returns
    /// `QueueFull` immediately (the protocol layer maps it to 503).
    pub fn submit(
        &self,
        task: &str,
        built: BuiltSequence,
        priority: i64,
    ) -> Result<oneshot::Receiver<Result<RawLogits, SchedulerError>>, SchedulerError> {
        let (respond, rx) = oneshot::channel();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.tx
            .try_send(ScheduledRequest {
                task: task.to_string(),
                built,
                respond,
                priority,
                seq,
            })
            .map_err(|err| match err {
                TrySendError::Full(_) => SchedulerError::QueueFull,
                TrySendError::Closed(_) => SchedulerError::ShuttingDown,
            })?;
        Ok(rx)
    }

    /// Graceful shutdown: close intake, let both tasks drain everything queued
    /// (intake -> pending -> prefetch channel -> engine), wait for exit
    /// (invariant 4).
    pub async fn shutdown(self) {
        drop(self.tx);
        let _ = self.join_batching.await;
        let _ = self.join_backend.await;
    }
}

pub fn spawn_scheduler(
    cfg: SchedulerConfig,
    layout: Layout,
    policy: Arc<dyn SchedulingPolicy>,
    runner: BatchRunner,
    replica: usize,
) -> Result<SchedulerHandle, SchedulerError> {
    cfg.validate(layout.max_len)
        .map_err(SchedulerError::Config)?;
    let (tx, rx) = mpsc::channel::<ScheduledRequest>(cfg.queue_cap);
    // Prefetch depth 1 (TEI structure): batch ASSEMBLY overlaps inference;
    // engine calls stay strictly sequential (one session behind one mutex) --
    // this overlaps collation, never execution.
    let (job_tx, job_rx) = mpsc::channel::<Job>(1);
    let join_batching = tokio::spawn(batching_task(cfg, layout, policy, rx, job_tx, replica));
    let join_backend = tokio::spawn(backend_task(runner, job_rx));
    Ok(SchedulerHandle {
        next_seq: AtomicU64::new(0),
        tx,
        join_batching,
        join_backend,
    })
}

/// A fully collated batch on its way to the backend task.
struct Job {
    items: Vec<ScheduledRequest>,
    batch: CollatedBatch,
}

/// Stage 1: intake, window batching, admission policy, collation.
/// Reserve-first: the send permit is acquired BEFORE pulling work, so a full
/// prefetch slot parks this task at `reserve()` -- natural backpressure with
/// the documented admission bound `queue_cap + 2 * max_batch_size` (one
/// forming batch holding a permit, one executing or prefetched).
async fn batching_task(
    cfg: SchedulerConfig,
    layout: Layout,
    policy: Arc<dyn SchedulingPolicy>,
    mut rx: mpsc::Receiver<ScheduledRequest>,
    job_tx: mpsc::Sender<Job>,
    replica: usize,
) {
    let mut pending: VecDeque<ScheduledRequest> = VecDeque::new();
    let mut open = true; // intake still has live senders
    let mut permit = None;
    loop {
        let held = match permit.take() {
            Some(held) => held,
            None => match job_tx.reserve().await {
                Ok(held) => held,
                // Backend gone: it only exits after this task's sender drops,
                // so reaching here means the backend panicked.
                Err(_) => break,
            },
        };

        // Wait for the first arrival (skip the wait when leftovers exist).
        if open && pending.is_empty() {
            match rx.recv().await {
                Some(req) => pending.push_back(req),
                None => open = false,
            }
        }
        // Window fill: gather more arrivals until the batch is full or the
        // window elapses. Buffered arrivals return instantly, so the window
        // only costs real waiting when intake runs dry.
        if open && pending.len() < cfg.max_batch_size {
            let deadline = tokio::time::Instant::now() + cfg.batch_window;
            while pending.len() < cfg.max_batch_size {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match timeout(remaining, rx.recv()).await {
                    Ok(Some(req)) => pending.push_back(req),
                    Ok(None) => {
                        open = false;
                        break;
                    }
                    Err(_) => break, // window elapsed
                }
            }
        }
        if pending.is_empty() {
            if open {
                permit = Some(held);
                continue;
            }
            break; // intake closed and drained: exit, dropping job_tx
        }
        crate::metrics::record_queue_depth(pending.len(), replica);
        let items = policy.admit(&mut pending, cfg.max_batch_size, cfg.token_budget);
        if items.is_empty() {
            // Every pending responder was closed (skipped by the policy).
            permit = Some(held);
            continue;
        }
        // Collation happens HERE, overlapping whatever the backend executes.
        let batch = collate(&items, &layout);
        crate::metrics::record_batch(batch.b, replica);
        held.send(Job { items, batch }); // never blocks: the permit is held
    }
}

/// Stage 2 (TEI `backend_task` pattern): run the engine on the blocking pool
/// and deliver every responder its own trimmed result. Any failure resolves
/// the WHOLE batch with errors (invariant 1's error path).
async fn backend_task(runner: BatchRunner, mut job_rx: mpsc::Receiver<Job>) {
    while let Some(Job { items, batch }) = job_rx.recv().await {
        let runner = Arc::clone(&runner);
        let result = tokio::task::spawn_blocking(move || runner(batch)).await;
        match result {
            Ok(Ok(rows)) if rows.len() == items.len() => {
                for (req, row) in items.into_iter().zip(rows) {
                    let real_k = req.built.markers.len();
                    if row.len() < real_k {
                        let _ = req.respond.send(Err(SchedulerError::Engine(format!(
                            "engine returned {} markers for a request with {real_k}",
                            row.len()
                        ))));
                    } else {
                        // Trim -1e4 sentinel slots: callers only ever see their own options.
                        let _ = req.respond.send(Ok(row[..real_k].to_vec()));
                    }
                }
            }
            Ok(Ok(rows)) => {
                let err = SchedulerError::Engine(format!(
                    "engine returned {} rows for a batch of {}",
                    rows.len(),
                    items.len()
                ));
                for req in items {
                    let _ = req.respond.send(Err(err.clone()));
                }
            }
            Ok(Err(err)) => {
                let err = SchedulerError::Engine(err);
                for req in items {
                    let _ = req.respond.send(Err(err.clone()));
                }
            }
            Err(join_err) => {
                let err = SchedulerError::Engine(format!("engine runner panicked: {join_err}"));
                for req in items {
                    let _ = req.respond.send(Err(err.clone()));
                }
            }
        }
    }
}
