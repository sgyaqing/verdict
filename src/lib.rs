//! Verdict: a high-performance serving engine for System One decision models.
//!
//! Serves decision models the way vLLM serves LLMs: one binary, one command,
//! an HTTP API, and correctness pinned to golden vectors. The native protocol
//! exposes the three System One task primitives (`/v1/choice`, `/v1/score`,
//! `/v1/noul`); `/v1/systemone` speaks the category's own Jev shape (one
//! state, many typed questions); `/v1/decisions` speaks OpenAI's
//! cross-ecosystem Decisions dialect; an OpenAI-compatible layer
//! (`/v1/chat/completions`, `/v1/models`) rides on top for SDK compatibility.
//!
//! Boot the whole stack with [`run_server`]:
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = verdict::cli::parse_args(&[
//!     "verdict".to_string(), "serve".to_string(), "models/laya-onnx".to_string(),
//! ])?;
//! let engine = verdict::engine::StubEngine::new(std::time::Duration::ZERO);
//! let engine: std::sync::Arc<dyn verdict::engine::Engine> = std::sync::Arc::new(engine);
//! verdict::run_server(cfg, vec![engine]).await?;
//! # Ok(())
//! # }
//! ```

pub mod calibration;
pub mod cli;
pub mod compat;
pub mod decisions;
pub mod engine;
pub mod metrics;
pub mod model_source;
pub mod pipeline;
pub mod protocol;
pub mod replay;
pub mod scheduler;
pub mod systemone;
pub mod tokenize;

use std::net::SocketAddr;
use std::sync::Arc;

pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

/// Boot the whole stack and serve. The engines are constructed by the caller
/// in the single-threaded startup phase (blocking model load stays off async
/// contexts) and handed in behind the Engine seam -- one per data-parallel
/// replica, so `engines.len()` must equal `cfg.data_parallel`. Each replica
/// gets its own scheduler and pipeline; [`pipeline::Pool`] multiplexes them.
/// Shutdown order is pinned: stop accepting -> in-flight
/// connections drain (axum) -> scheduler intake closes -> queued requests
/// drain -> join (every replica).
pub async fn run_server(
    cfg: cli::ServeConfig,
    engines: Vec<Arc<dyn engine::Engine>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if engines.is_empty() || engines.len() != cfg.data_parallel {
        return Err(format!(
            "engine count ({}) must match --data-parallel ({}) and be at least 1",
            engines.len(),
            cfg.data_parallel,
        )
        .into());
    }
    metrics::install();
    let bundle = tokenize::TokenizerBundle::load(&cfg.model_dir)
        .map_err(|err| format!("model dir {}: {err}", cfg.model_dir.display()))?;
    let calibration = calibration::CalibrationConfig::load(&cfg.model_dir)
        .map_err(|err| format!("model dir {}: {err}", cfg.model_dir.display()))?;

    let policy: Arc<dyn scheduler::SchedulingPolicy> = match cfg.policy {
        cli::PolicyChoice::Fcfs => Arc::new(scheduler::Fcfs),
        cli::PolicyChoice::Priority => Arc::new(scheduler::Priority),
    };

    // Advertised model id: the flag wins, else the model dir's basename.
    let served_model = cfg.served_model_name.clone().unwrap_or_else(|| {
        cfg.model_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "verdict".to_string())
    });

    let mut replicas = Vec::with_capacity(engines.len());
    for (replica, engine) in engines.into_iter().enumerate() {
        let scheduler = scheduler::spawn_scheduler(
            scheduler::SchedulerConfig::new(
                cfg.queue_cap,
                cfg.max_batch_size,
                cfg.batch_window_ms,
                bundle.layout.max_len,
            ),
            bundle.layout.clone(),
            Arc::clone(&policy),
            engine::engine_runner(Arc::clone(&engine)),
            replica,
        )?;
        replicas.push(Arc::new(pipeline::Pipeline::new(
            bundle.clone(),
            calibration.clone(),
            scheduler,
            cfg.max_input_tokens,
            engine,
            served_model.clone(),
        )));
    }
    if replicas.len() > 1 {
        tracing::info!(
            replicas = replicas.len(),
            queue_cap_each = cfg.queue_cap,
            "data-parallel pool ready (requests spread round-robin with spill-over)"
        );
    }
    let pool = Arc::new(pipeline::Pool::new(replicas));
    let app = protocol::router(Arc::clone(&pool));
    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, model_dir = %cfg.model_dir.display(), "verdict listening (press Ctrl-C to stop)");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Connections have drained; the serve future is gone and so is the router.
    let pool = Arc::try_unwrap(pool)
        .map_err(|_| "the router still holds the pool after shutdown".to_string())?;
    pool.shutdown().await;
    tracing::info!("shutdown complete");
    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!(%err, "failed to install Ctrl-C handler");
    }
}
