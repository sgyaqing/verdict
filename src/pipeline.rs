use crate::calibration::{CalibCtx, Calibrated, CalibrationConfig, calibrate};
use crate::engine::Engine;
use crate::protocol::{
    ApiError, ChoiceRequest, ChoiceResponse, NoulRequest, NoulResponse, ScoreRequest,
    ScoreResponse, Usage,
};
use crate::scheduler::{SchedulerError, SchedulerHandle};
use crate::tokenize::TokenizerBundle;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
pub struct Pipeline {
    bundle: TokenizerBundle,
    calibration: CalibrationConfig,
    scheduler: SchedulerHandle,
    max_input_tokens: usize,
    engine: Arc<dyn Engine>,
    served_model: String,
    next_request: AtomicU64,
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
            next_request: AtomicU64::new(0),
        }
    }

    fn next_request(&self) -> u64 {
        self.next_request.fetch_add(1, Ordering::Relaxed)
    }

    pub fn served_model(&self) -> &str {
        &self.served_model
    }

    pub fn health(&self) -> bool {
        self.engine.health()
    }

    fn engine_media_capable(&self) -> bool {
        self.engine.media_capable()
    }

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
                &[],
                req.priority(),
                self.next_request(),
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
                &[],
                req.priority(),
                self.next_request(),
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
        let (calibrated, input_tokens) = self
            .run_task(
                "noul",
                &req.claim,
                &req.state,
                &[],
                &[],
                req.priority(),
                self.next_request(),
            )
            .await?;
        crate::metrics::record_decision("noul", started.elapsed().as_millis() as u64);
        Ok(NoulResponse {
            probability: calibrated.probability.unwrap_or(0.0),
            usage: Usage { input_tokens },
            latency_ms: elapsed_ms(started),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn run_task(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        media: &[crate::tokenize::MediaItem],
        priority: i64,
        request: u64,
    ) -> Result<(Calibrated, u64), ApiError> {
        match self.bundle.family {
            crate::family::Family::LetterReadout | crate::family::Family::LetterBranches => {
                self.run_task_letter(task, question, state, options, media, priority, request)
                    .await
            }
            _ => {
                self.run_task_marker(task, question, state, options, priority, request)
                    .await
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_task_letter(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        media: &[crate::tokenize::MediaItem],
        priority: i64,
        request: u64,
    ) -> Result<(Calibrated, u64), ApiError> {
        let rows = self
            .bundle
            .build_sequences(task, question, state, options, media);
        for row in &rows {
            if row.full_len > self.max_input_tokens {
                return Err(ApiError::InvalidRequest(format!(
                    "rendered prompt is {} tokens, exceeding the input token budget {} \
                     (model max_len is {})",
                    row.full_len, self.max_input_tokens, self.bundle.layout.max_len
                )));
            }
        }
        let input_tokens = crate::family::letter_readout::unique_tokens(
            &rows.iter().map(|r| r.ids.as_slice()).collect::<Vec<_>>(),
        );
        let mut rxs = Vec::with_capacity(rows.len());
        for row in rows {
            let rx = self
                .scheduler
                .submit_in(request, task, row, priority)
                .map_err(map_scheduler_error)?;
            rxs.push(rx);
        }
        let mut logits = Vec::new();
        let mut expanded: Option<u32> = None;
        for rx in rxs {
            let row = rx
                .await
                .map_err(|_| ApiError::Internal("scheduler dropped the responder".into()))?
                .map_err(map_scheduler_error)?;
            logits.extend(row.logits);
            if row.expanded_len.is_some() {
                expanded = row.expanded_len;
            }
        }
        let ctx = CalibCtx { state_tokens: 0 };
        Ok((
            calibrate(task, &logits, &ctx, &self.calibration),
            expanded.map(|n| n as u64).unwrap_or(input_tokens),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_task_marker(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        priority: i64,
        request: u64,
    ) -> Result<(Calibrated, u64), ApiError> {
        let built = self.bundle.build_sequence(task, question, state, options);
        let mut input_tokens = built.full_len as u64;
        let state_tokens = built.state_tokens;
        if input_tokens > self.max_input_tokens as u64 {
            return Err(ApiError::InvalidRequest(format!(
                "rendered prompt is {input_tokens} tokens, exceeding the input token budget {} \
                 (model max_len is {})",
                self.max_input_tokens, self.bundle.layout.max_len
            )));
        }
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
            .submit_in(request, task, built, priority)
            .map_err(map_scheduler_error)?;
        let null_rx = match &null_built {
            Some((null, _)) => Some(
                self.scheduler
                    .submit_in(request, task, null.clone(), priority)
                    .map_err(map_scheduler_error)?,
            ),
            None => None,
        };
        let row = rx
            .await
            .map_err(|_| ApiError::Internal("scheduler dropped the responder".into()))?
            .map_err(map_scheduler_error)?;
        let logits = match (null_rx, null_built) {
            (Some(null_rx), Some((null, prior))) => {
                let null_row = null_rx
                    .await
                    .map_err(|_| ApiError::Internal("scheduler dropped the responder".into()))?
                    .map_err(map_scheduler_error)?;
                input_tokens += null.full_len as u64;
                let bias = null_row.logits[0] - null_row.logits[1];
                vec![row.logits[0] - (prior.a * bias + prior.b), row.logits[1]]
            }
            _ => row.logits,
        };
        let ctx = CalibCtx { state_tokens };
        Ok((
            calibrate(task, &logits, &ctx, &self.calibration),
            input_tokens,
        ))
    }

    pub fn build_row_ids(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
    ) -> Vec<Vec<u32>> {
        self.bundle
            .build_sequences(task, question, state, options, &[])
            .into_iter()
            .map(|row| row.ids)
            .collect()
    }

    pub async fn shutdown(self) {
        let Self {
            scheduler,
            bundle,
            calibration,
            engine,
            max_input_tokens: _,
            served_model: _,
            next_request: _,
        } = self;
        scheduler.shutdown().await;
        std::mem::forget((bundle, calibration, engine));
    }
}
fn map_scheduler_error(err: SchedulerError) -> ApiError {
    match err {
        SchedulerError::QueueFull | SchedulerError::ShuttingDown => ApiError::Overloaded,
        SchedulerError::Engine(message) => {
            tracing::error!(err = %message, "engine inference failed");
            ApiError::Internal("inference failed".into())
        }
        SchedulerError::Config(message) => {
            tracing::error!(err = %message, "scheduler configuration rejected");
            ApiError::Internal(message)
        }
    }
}
fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

pub struct Pool {
    replicas: Vec<Arc<Pipeline>>,
    next: AtomicUsize,
    next_request: AtomicU64,
}
impl Pool {
    pub fn new(replicas: Vec<Arc<Pipeline>>) -> Self {
        assert!(!replicas.is_empty(), "a pool needs at least one replica");
        Self {
            replicas,
            next: AtomicUsize::new(0),
            next_request: AtomicU64::new(0),
        }
    }

    pub fn next_request(&self) -> u64 {
        self.next_request.fetch_add(1, Ordering::Relaxed)
    }

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

    #[allow(clippy::too_many_arguments)]
    pub async fn run_task(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
        media: &[crate::tokenize::MediaItem],
        priority: i64,
        request: u64,
    ) -> Result<(Calibrated, u64), ApiError> {
        let n = self.replicas.len();
        let start = self.next.fetch_add(1, Ordering::Relaxed);
        for i in 0..n {
            let replica = &self.replicas[(start + i) % n];
            match replica
                .run_task(task, question, state, options, media, priority, request)
                .await
            {
                Ok(value) => return Ok(value),
                Err(ApiError::Overloaded) => continue,
                Err(err) => return Err(err),
            }
        }
        Err(ApiError::Overloaded)
    }

    pub fn served_model(&self) -> &str {
        self.replicas[0].served_model()
    }

    pub fn media_capable(&self) -> bool {
        self.replicas[0].engine_media_capable()
    }

    pub fn state_prefix_len(&self, state: &str) -> usize {
        match self.family() {
            crate::family::Family::LetterBranches => {
                crate::family::letter_branches::prefix_token_len(&self.replicas[0].bundle, state)
            }
            _ => 0,
        }
    }

    pub fn build_row_ids(
        &self,
        task: &str,
        question: &str,
        state: &str,
        options: &[String],
    ) -> Vec<Vec<u32>> {
        self.replicas[0].build_row_ids(task, question, state, options)
    }

    pub fn family(&self) -> crate::family::Family {
        self.replicas[0].family()
    }

    pub fn health(&self) -> bool {
        self.replicas.iter().all(|r| r.health())
    }

    pub async fn shutdown(self) {
        for replica in self.replicas {
            if let Ok(pipeline) = Arc::try_unwrap(replica) {
                pipeline.shutdown().await;
            }
        }
    }
}
