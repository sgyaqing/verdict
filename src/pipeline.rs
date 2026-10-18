//! Pipeline facade: the ONLY object the HTTP handlers talk to. Owns
//! tokenize -> admission check -> scheduler -> await -> calibration ->
//! response assembly, with per-request tracing spans for the metrics plane.
//!
//! [`Pool`] multiplexes N independent pipelines for `--data-parallel`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::calibration::{CalibCtx, Calibrated, CalibrationConfig, calibrate};
use crate::engine::Engine;
use crate::protocol::{
    ApiError, ChoiceRequest, ChoiceResponse, NoulRequest, NoulResponse, ScoreRequest,
    ScoreResponse, Usage,
};
use crate::scheduler::{SchedulerError, SchedulerHandle};
use crate::tokenize::TokenizerBundle;

pub struct Pipeline {
    bundle: TokenizerBundle,
    calibration: CalibrationConfig,
    scheduler: SchedulerHandle,
    max_input_tokens: usize,
    engine: Arc<dyn Engine>,
    served_model: String,
}

impl Pipeline {
    pub fn new(
        bundle: TokenizerBundle,
        calibration: CalibrationConfig,
        scheduler: SchedulerHandle,
        max_input_tokens: usize,
        engine: Arc<dyn Engine>,
        served_model: String,
    ) -> Self {
        Self {
            bundle,
            calibration,
            scheduler,
            max_input_tokens,
            engine,
            served_model,
        }
    }

    /// Model id advertised on the compat surface: the id identifies the
    /// served instance, never a hardcoded family name.
    pub fn served_model(&self) -> &str {
        &self.served_model
    }

    /// Liveness surfaced through /health: a real probe, not a static "ok".
    pub fn health(&self) -> bool {
        self.engine.health()
    }

    /// Which family recipe this pipeline serves (protocol adapters compose
    /// per-family request shapes).
    pub fn family(&self) -> crate::family::Family {
        self.bundle.family
    }

    #[tracing::instrument(skip_all, fields(task = "choice"))]
    pub async fn choice(&self, req: ChoiceRequest) -> Result<ChoiceResponse, ApiError> {
        let started = Instant::now();
        let (calibrated, input_tokens) = self
            .run_task(
                "choice",
                &req.question,
                &req.state,
                &req.options,
                req.priority(),
            )
            .await?;
        crate::metrics::record_decision("choice", started.elapsed().as_millis() as u64);
        Ok(ChoiceResponse {
            choice: calibrated.choice.unwrap_or(0),
            probabilities: calibrated.probabilities,
            usage: Usage { input_tokens },
            latency_ms: elapsed_ms(started),
        })
    }

    #[tracing::instrument(skip_all, fields(task = "score"))]
    pub async fn score(&self, req: ScoreRequest) -> Result<ScoreResponse, ApiError> {
        let started = Instant::now();
        let (calibrated, input_tokens) = self
            .run_task(
                "score",
                &req.question,
                &req.state,
                &req.levels,
                req.priority(),
            )
            .await?;
        crate::metrics::record_decision("score", started.elapsed().as_millis() as u64);
        Ok(ScoreResponse {
            score: calibrated.score.unwrap_or(0.0),
            probabilities: calibrated.probabilities,
            usage: Usage { input_tokens },
            latency_ms: elapsed_ms(started),
        })
    }

    #[tracing::instrument(skip_all, fields(task = "noul"))]
    pub async fn noul(&self, req: NoulRequest) -> Result<NoulResponse, ApiError> {
        let started = Instant::now();
        // noul options are server-rendered; none come from the caller.
        let (calibrated, input_tokens) = self
            .run_task("noul", &req.claim, &req.state, &[], req.priority())
            .await?;
        crate::metrics::record_decision("noul", started.elapsed().as_millis() as u64);
        Ok(NoulResponse {
            probability: calibrated.probability.unwrap_or(0.0),
            usage: Usage { input_tokens },
            latency_ms: elapsed_ms(started),
        })
    }

    /// The task seam every surface composes: tokenize -> protocol-layer token
    /// budget -> scheduler -> calibrated result. The native handlers wrap it
    /// per task; `/v1/systemone` drives it per question. Metrics stay with
    /// the callers (one decision = one record). Model-layer truncation
    /// (512/192) happens inside build_sequence; the 8192 check is on the
    /// UNCAPPED rendered length.
    pub async fn run_task(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        priority: i64,
    ) -> Result<(Calibrated, u64), ApiError> {
        match self.bundle.family {
            crate::family::Family::LetterReadout => {
                self.run_task_letter(task, question, state, options, priority)
                    .await
            }
            _ => {
                self.run_task_marker(task, question, state, options, priority)
                    .await
            }
        }
    }

    /// The letter-readout task path: a request renders ONE OR MORE rows (a
    /// score question runs one isolated yes/no row per level), every row is
    /// its own scheduler submission so the batcher packs them together, and
    /// usage counts the shared state prefix once (official unique_tokens --
    /// for a single-question request this is exactly the official number).
    async fn run_task_letter(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        priority: i64,
    ) -> Result<(Calibrated, u64), ApiError> {
        let rows = self.bundle.build_sequences(task, question, state, options);
        for row in &rows {
            if row.full_len > self.max_input_tokens {
                return Err(ApiError::InvalidRequest(format!(
                    "rendered prompt is {} tokens, exceeding --max-input-tokens {} \
                     (note: model-layer truncation at {} does not exempt a request from this budget)",
                    row.full_len, self.max_input_tokens, self.bundle.layout.max_len
                )));
            }
        }
        let input_tokens = crate::family::letter_readout::unique_tokens(
            &rows.iter().map(|r| r.ids.as_slice()).collect::<Vec<_>>(),
        );
        // Submit every row before awaiting: they land in the queue together
        // and the batcher forms one batch (subject to the token budget).
        let mut rxs = Vec::with_capacity(rows.len());
        for row in rows {
            let rx = self
                .scheduler
                .submit(task, row, priority)
                .map_err(map_scheduler_error)?;
            rxs.push(rx);
        }
        let mut logits = Vec::new();
        for rx in rxs {
            let row = rx
                .await
                .map_err(|_| ApiError::Internal("scheduler dropped the responder".into()))?
                .map_err(map_scheduler_error)?;
            logits.extend(row);
        }
        let ctx = CalibCtx { state_tokens: 0 };
        Ok((
            calibrate(task, &logits, &ctx, &self.calibration),
            input_tokens,
        ))
    }

    /// The marker-family task path (single row + the independent-options
    /// zero-shot noul debias pass).
    async fn run_task_marker(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        priority: i64,
    ) -> Result<(Calibrated, u64), ApiError> {
        let built = self.bundle.build_sequence(task, question, state, options);
        let mut input_tokens = built.full_len as u64;
        let state_tokens = built.state_tokens;
        if input_tokens > self.max_input_tokens as u64 {
            return Err(ApiError::InvalidRequest(format!(
                "rendered prompt is {input_tokens} tokens, exceeding --max-input-tokens {} \
                 (note: model-layer truncation at {} does not exempt a request from this budget)",
                self.max_input_tokens, self.bundle.layout.max_len
            )));
        }
        // Zero-shot noul on the independent-options family: the official
        // path runs a SECOND context-free forward and shifts the true-side
        // logit by a fitted prior before calibration (usage counts both
        // passes). Options empty == no caller criteria == zero-shot.
        let prior = self.calibration.noul_prior();
        let null_built = match (task, prior) {
            ("noul", Some(prior)) if options.is_empty() => Some((
                self.bundle.build_sequence(task, question, "", options),
                prior,
            )),
            _ => None,
        };
        let rx = self
            .scheduler
            .submit(task, built, priority)
            .map_err(map_scheduler_error)?;
        let null_rx = match &null_built {
            Some((null, _)) => Some(
                self.scheduler
                    .submit(task, null.clone(), priority)
                    .map_err(map_scheduler_error)?,
            ),
            None => None,
        };
        let logits = rx
            .await
            .map_err(|_| ApiError::Internal("scheduler dropped the responder".into()))?
            .map_err(map_scheduler_error)?;
        let logits = match (null_rx, null_built) {
            (Some(null_rx), Some((null, prior))) => {
                let null_logits = null_rx
                    .await
                    .map_err(|_| ApiError::Internal("scheduler dropped the responder".into()))?
                    .map_err(map_scheduler_error)?;
                input_tokens += null.full_len as u64;
                let bias = null_logits[0] - null_logits[1];
                vec![logits[0] - (prior.a * bias + prior.b), logits[1]]
            }
            _ => logits,
        };
        let ctx = CalibCtx { state_tokens };
        Ok((
            calibrate(task, &logits, &ctx, &self.calibration),
            input_tokens,
        ))
    }

    /// The engine rows a task request would render (no submission): the
    /// request-level usage seam for multi-question surfaces on the
    /// letter-readout family, whose usage counts the shared state prefix
    /// once across ALL questions of one request, not once per question.
    pub fn build_row_ids(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
    ) -> Vec<Vec<u32>> {
        self.bundle
            .build_sequences(task, question, state, options)
            .into_iter()
            .map(|row| row.ids)
            .collect()
    }

    /// Graceful shutdown: close intake, drain, join (called AFTER the HTTP
    /// side has stopped accepting and connections drained).
    pub async fn shutdown(self) {
        self.scheduler.shutdown().await;
    }
}

fn map_scheduler_error(err: SchedulerError) -> ApiError {
    match err {
        // Both map to "try later" semantics; the protocol has no distinct
        // draining kind (deliberate).
        SchedulerError::QueueFull | SchedulerError::ShuttingDown => ApiError::Overloaded,
        SchedulerError::Engine(_) => ApiError::Internal("inference failed".into()),
        SchedulerError::Config(message) => ApiError::Internal(message),
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

/// Data-parallel pool: N independent replicas, each a full [`Pipeline`] --
/// its own engine session, bounded queue, and batcher. On many-core x86
/// hosts, concurrent sessions measurably outperform one session executing
/// bigger batches; on GPUs and
/// constrained-core hosts the default N=1 remains the right shape.
///
/// Routing: round-robin with spill-over. A replica whose queue is full
/// rejects at submit time (invariant 3, immediate `QueueFull`) and the pool
/// takes the request to the next replica; only when EVERY replica refuses
/// does the caller see 503 -- no request is dropped while any replica has
/// room. Non-overload errors (validation, engine failures) are never
/// retried on another replica; a spill costs one extra tokenize (a full
/// queue answers before any inference work).
pub struct Pool {
    replicas: Vec<Arc<Pipeline>>,
    next: AtomicUsize,
}

impl Pool {
    /// Build a pool over `replicas`; panics on an empty vec (the server
    /// constructor guarantees at least one engine).
    pub fn new(replicas: Vec<Arc<Pipeline>>) -> Self {
        assert!(!replicas.is_empty(), "a pool needs at least one replica");
        Self {
            replicas,
            next: AtomicUsize::new(0),
        }
    }

    /// Route one request: try replicas in round-robin order, spilling on
    /// overload. `call` receives a fresh clone per attempt; the boxed future
    /// may borrow its replica (one allocation per attempt).
    async fn route<T, Req>(
        &self,
        req: Req,
        call: impl for<'a> Fn(
            &'a Pipeline,
            Req,
        ) -> std::pin::Pin<
            std::boxed::Box<dyn std::future::Future<Output = Result<T, ApiError>> + Send + 'a>,
        >,
    ) -> Result<T, ApiError>
    where
        Req: Clone,
    {
        let n = self.replicas.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for i in 0..n {
            let replica = &self.replicas[(start + i) % n];
            match call(replica, req.clone()).await {
                Ok(value) => return Ok(value),
                // Spill only on overload; every other error is the
                // request's own and must surface unchanged.
                Err(ApiError::Overloaded) => continue,
                Err(err) => return Err(err),
            }
        }
        Err(ApiError::Overloaded)
    }

    pub async fn choice(&self, req: ChoiceRequest) -> Result<ChoiceResponse, ApiError> {
        self.route(req, |replica, req| Box::pin(replica.choice(req)))
            .await
    }

    pub async fn score(&self, req: ScoreRequest) -> Result<ScoreResponse, ApiError> {
        self.route(req, |replica, req| Box::pin(replica.score(req)))
            .await
    }

    pub async fn noul(&self, req: NoulRequest) -> Result<NoulResponse, ApiError> {
        self.route(req, |replica, req| Box::pin(replica.noul(req)))
            .await
    }

    /// Per-question task submission for `/v1/systemone`: the same round-robin
    /// with spill-over as [`Pool::route`], inlined because borrowed arguments
    /// do not fit route's `Req: Clone` shape.
    pub async fn run_task(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        priority: i64,
    ) -> Result<(Calibrated, u64), ApiError> {
        let n = self.replicas.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for i in 0..n {
            let replica = &self.replicas[(start + i) % n];
            match replica
                .run_task(task, question, state, options, priority)
                .await
            {
                Ok(value) => return Ok(value),
                Err(ApiError::Overloaded) => continue, // spill like route
                Err(err) => return Err(err),
            }
        }
        Err(ApiError::Overloaded)
    }

    /// Model id advertised on the compat surface (identical across replicas).
    pub fn served_model(&self) -> &str {
        self.replicas[0].served_model()
    }

    /// The engine rows a task request would render, via the first replica's
    /// tokenizer (identical across replicas; no submission, no engine work).
    pub fn build_row_ids(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
    ) -> Vec<Vec<u32>> {
        self.replicas[0].build_row_ids(task, question, state, options)
    }

    /// Which family recipe the pool serves (identical across replicas).
    pub fn family(&self) -> crate::family::Family {
        self.replicas[0].family()
    }

    /// Liveness: healthy only when EVERY replica's engine is (a degraded
    /// replica means degraded capacity; no half-healthy 200s).
    pub fn health(&self) -> bool {
        self.replicas.iter().all(|r| r.health())
    }

    /// Graceful shutdown of every replica, in order (same pinned order as
    /// the single-pipeline server: called after HTTP has drained).
    pub async fn shutdown(self) {
        for replica in self.replicas {
            if let Ok(pipeline) = Arc::try_unwrap(replica) {
                pipeline.shutdown().await;
            }
        }
    }
}
