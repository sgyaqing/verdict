use crate::tokenize::{BuiltSequence, Layout};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::timeout;

const PROBE_QUANTUM: Duration = Duration::from_micros(500);

const ADMIT_POOL_FACTOR: usize = 4;

pub type RawLogits = crate::engine::EngineRow;
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
    pub queue_cap: usize,

    pub max_batch_size: usize,

    pub batch_window: Duration,

    pub token_budget: usize,
}
impl SchedulerConfig {
    pub fn new(
        queue_cap: usize,
        max_batch_size: usize,
        batch_window_ms: u64,
        token_budget: usize,
    ) -> Self {
        Self {
            queue_cap,
            max_batch_size,
            batch_window: Duration::from_millis(batch_window_ms),
            token_budget,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.queue_cap == 0 {
            return Err("queue_cap must be positive".to_string());
        }
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be positive".to_string());
        }
        if self.token_budget == 0 {
            return Err("token_budget must be positive".to_string());
        }
        Ok(())
    }
}

pub struct ScheduledRequest {
    pub task: String,
    pub built: BuiltSequence,
    pub respond: oneshot::Sender<Result<RawLogits, SchedulerError>>,

    pub priority: i64,

    pub seq: u64,

    pub request: u64,

    pub skips: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollatedBatch {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub marker_pos: Vec<i64>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<i64>,

    pub nopts: Vec<i64>,

    pub media: Vec<crate::tokenize::MediaItem>,
    pub media_counts: Vec<i64>,
    pub row_texts: Vec<String>,
    pub b: usize,
    pub l: usize,
    pub k: usize,

    pub rows: Vec<RowMeta>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMeta {
    pub real_len: usize,
    pub real_k: usize,
    pub prefix_len: usize,
}

pub type BatchRunner =
    Arc<dyn Fn(CollatedBatch) -> Result<Vec<crate::engine::EngineRow>, String> + Send + Sync>;

pub fn collate(items: &[ScheduledRequest], layout: &Layout) -> CollatedBatch {
    let b = items.len();
    let mut l = items.iter().map(|r| r.built.ids.len()).max().unwrap_or(0);
    if !layout.fixed_classes.is_empty() {
        if let Some(&class) = layout.fixed_classes.iter().find(|&&c| c >= l) {
            l = class;
        }
    } else if layout.seq_align > 1 {
        l = l.div_ceil(layout.seq_align) * layout.seq_align;
    }
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
    let mut nopts = Vec::with_capacity(b);
    let mut rows = Vec::with_capacity(b);
    let mut media = Vec::new();
    let mut media_counts = Vec::with_capacity(b);
    let mut row_texts = Vec::with_capacity(b);
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
        let real_k = req.built.real_k();
        if !req.built.nopts.is_empty() {
            nopts.push(real_k as i64);
        }
        media.extend(req.built.media.iter().cloned());
        media_counts.push(req.built.media.len() as i64);
        row_texts.push(req.built.text.clone());
        rows.push(RowMeta {
            real_len,
            real_k,
            prefix_len: req.built.prefix_len,
        });
    }
    CollatedBatch {
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        nopts,
        media,
        media_counts,
        row_texts,
        b,
        l,
        k,
        rows,
    }
}

pub trait SchedulingPolicy: Send + Sync + 'static {
    fn admit(
        &self,
        pending: &mut VecDeque<ScheduledRequest>,
        max_batch_size: usize,
        token_budget: usize,
    ) -> Vec<ScheduledRequest>;
}

fn admit_in_front_order(
    pending: &mut VecDeque<ScheduledRequest>,
    max_batch_size: usize,
    token_budget: usize,
) -> Vec<ScheduledRequest> {
    let mut batch: Vec<ScheduledRequest> = Vec::new();
    let mut max_l = 0usize;
    while let Some(req) = pending.pop_front() {
        if req.respond.is_closed() {
            continue;
        }
        let candidate_l = max_l.max(req.built.ids.len());
        let cost_fits = batch.is_empty() || candidate_l * (batch.len() + 1) <= token_budget;
        if batch.len() < max_batch_size && cost_fits {
            max_l = candidate_l;
            batch.push(req);
        } else {
            pending.push_front(req);
            break;
        }
    }
    batch
}

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

pub struct Priority;
impl SchedulingPolicy for Priority {
    fn admit(
        &self,
        pending: &mut VecDeque<ScheduledRequest>,
        max_batch_size: usize,
        token_budget: usize,
    ) -> Vec<ScheduledRequest> {
        pending
            .make_contiguous()
            .sort_by_key(|r| (r.priority, r.seq));
        admit_in_front_order(pending, max_batch_size, token_budget)
    }
}

fn length_class(buckets: &[usize], len: usize) -> Option<usize> {
    buckets.iter().copied().find(|&b| len <= b)
}

pub struct ShapeAware {
    buckets: Vec<usize>,
    starve_after: u32,
}

impl ShapeAware {
    pub fn new(buckets: Vec<usize>) -> Self {
        Self {
            buckets,
            starve_after: 8,
        }
    }
}
impl SchedulingPolicy for ShapeAware {
    fn admit(
        &self,
        pending: &mut VecDeque<ScheduledRequest>,
        max_batch_size: usize,
        token_budget: usize,
    ) -> Vec<ScheduledRequest> {
        if self.buckets.is_empty() {
            return admit_in_front_order(pending, max_batch_size, token_budget);
        }
        let mut items: Vec<ScheduledRequest> = pending
            .drain(..)
            .filter(|r| !r.respond.is_closed())
            .collect();
        if items.is_empty() {
            return Vec::new();
        }
        let anchor = items
            .iter()
            .position(|r| r.skips >= self.starve_after)
            .unwrap_or(0);
        let head = items.remove(anchor);
        let class = length_class(&self.buckets, head.built.ids.len());
        let request = head.request;
        let span = class.unwrap_or(head.built.ids.len());
        let cap_by_tokens = (token_budget / span.max(1)).max(1);
        let rows_cap = max_batch_size.min(cap_by_tokens).max(1);
        let mut chosen = Vec::with_capacity(rows_cap);
        chosen.push(head);
        let mut rest: Vec<ScheduledRequest> = Vec::with_capacity(items.len());
        for req in items.drain(..) {
            let fits = req.request == request && chosen.len() < rows_cap;
            if fits {
                chosen.push(req);
            } else {
                rest.push(req);
            }
        }
        let mut skipped: Vec<ScheduledRequest> = Vec::with_capacity(rest.len());
        for mut req in rest.drain(..) {
            let fits = length_class(&self.buckets, req.built.ids.len()) == class
                && chosen.len() < rows_cap;
            if fits {
                chosen.push(req);
            } else {
                req.skips = req.skips.saturating_add(1);
                skipped.push(req);
            }
        }
        pending.extend(skipped);
        chosen
    }
}
pub struct SchedulerHandle {
    tx: mpsc::Sender<ScheduledRequest>,
    next_seq: AtomicU64,
    join_batching: JoinHandle<()>,
    join_backend: JoinHandle<()>,
}
impl SchedulerHandle {
    pub fn submit(
        &self,
        task: &str,
        built: BuiltSequence,
        priority: i64,
    ) -> Result<oneshot::Receiver<Result<RawLogits, SchedulerError>>, SchedulerError> {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        self.submit_in(seq, task, built, priority)
    }

    pub fn submit_in(
        &self,
        request: u64,
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
                request,
                skips: 0,
            })
            .map_err(|err| match err {
                TrySendError::Full(_) => SchedulerError::QueueFull,
                TrySendError::Closed(_) => SchedulerError::ShuttingDown,
            })?;
        Ok(rx)
    }

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
    cfg.validate().map_err(SchedulerError::Config)?;
    let (tx, rx) = mpsc::channel::<ScheduledRequest>(cfg.queue_cap);
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

struct Job {
    items: Vec<ScheduledRequest>,
    batch: CollatedBatch,
}

async fn batching_task(
    cfg: SchedulerConfig,
    layout: Layout,
    policy: Arc<dyn SchedulingPolicy>,
    mut rx: mpsc::Receiver<ScheduledRequest>,
    job_tx: mpsc::Sender<Job>,
    replica: usize,
) {
    let mut pending: VecDeque<ScheduledRequest> = VecDeque::new();
    let mut open = true;
    let mut permit = None;
    loop {
        let mut waited_for_slot;
        let held = match permit.take() {
            Some(held) => {
                waited_for_slot = false;
                held
            }
            None => {
                let before = job_tx.capacity();
                match job_tx.reserve().await {
                    Ok(held) => {
                        waited_for_slot = before == 0;
                        held
                    }
                    Err(_) => break,
                }
            }
        };
        waited_for_slot = waited_for_slot && !pending.is_empty();
        if open && pending.is_empty() {
            match rx.recv().await {
                Some(req) => pending.push_back(req),
                None => open = false,
            }
        }
        let fill = cfg.max_batch_size.saturating_mul(ADMIT_POOL_FACTOR);
        if open && pending.len() < fill && !waited_for_slot {
            let deadline = tokio::time::Instant::now() + cfg.batch_window;
            while pending.len() < fill {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                let quantum = remaining.min(PROBE_QUANTUM);
                match timeout(quantum, rx.recv()).await {
                    Ok(Some(req)) => pending.push_back(req),
                    Ok(None) => {
                        open = false;
                        break;
                    }
                    Err(_) => break,
                }
            }
        }
        if pending.is_empty() {
            if open {
                permit = Some(held);
                continue;
            }
            break;
        }
        crate::metrics::record_queue_depth(pending.len(), replica);
        let items = policy.admit(&mut pending, cfg.max_batch_size, cfg.token_budget);
        if items.is_empty() {
            permit = Some(held);
            continue;
        }
        let batch = collate(&items, &layout);
        crate::metrics::record_batch(batch.b, replica);
        held.send(Job { items, batch });
    }
}

type InflightBatch = (
    Vec<ScheduledRequest>,
    tokio::task::JoinHandle<Result<Vec<crate::engine::EngineRow>, String>>,
);

async fn backend_task(runner: BatchRunner, mut job_rx: mpsc::Receiver<Job>) {
    let depth = usize::from(
        std::env::var("VERDICT_PIPELINE")
            .map(|v| v != "0")
            .unwrap_or(true),
    );
    let mut inflight: Vec<InflightBatch> = Vec::new();
    let mut alive = true;
    while alive || !inflight.is_empty() {
        while alive && inflight.len() < depth.max(1) {
            match job_rx.recv().await {
                Some(Job { items, batch }) => {
                    let runner = Arc::clone(&runner);
                    let n = items.len();
                    let spawned = tokio::task::spawn_blocking(move || {
                        runner(batch).map(|mut rows| {
                            rows.truncate(n);
                            rows
                        })
                    });
                    inflight.push((items, spawned));
                }
                None => {
                    alive = false;
                    break;
                }
            }
        }
        if inflight.is_empty() {
            continue;
        }
        let (items, handle) = inflight.remove(0);
        let result = handle.await;
        match result {
            Ok(Ok(rows)) if rows.len() == items.len() => {
                for (req, row) in items.into_iter().zip(rows) {
                    let real_k = req.built.real_k();
                    if row.logits.len() < real_k {
                        let _ = req.respond.send(Err(SchedulerError::Engine(format!(
                            "engine returned {} markers for a request with {real_k}",
                            row.logits.len()
                        ))));
                    } else {
                        let _ = req.respond.send(Ok(RawLogits {
                            logits: row.logits[..real_k].to_vec(),
                            expanded_len: row.expanded_len,
                        }));
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
