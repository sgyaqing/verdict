pub mod calibration;
pub mod cli;
pub mod compat;
pub mod convert;
pub mod decisions;
pub mod engine;
pub mod family;
pub mod graph_stats;
pub mod manifest;
pub mod metrics;
pub mod model_source;
pub mod pipeline;
pub mod progress;
pub mod protocol;
pub mod replay;
pub mod scheduler;
pub mod systemone;
pub mod tokenize;
pub mod torch_engine;
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

pub fn init_tracing_stderr() {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

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
    let max_input_tokens = match cfg.max_input_tokens {
        Some(n) => {
            if n > bundle.layout.max_len {
                tracing::warn!(
                    "max-input-tokens {n} exceeds the model's max_len {}; longer \
                     requests are truncated",
                    bundle.layout.max_len
                );
            }
            n
        }
        None => usize::MAX,
    };
    let calibration = calibration::CalibrationConfig::load(&cfg.model_dir)
        .map_err(|err| format!("model dir {}: {err}", cfg.model_dir.display()))?;
    let policy: Arc<dyn scheduler::SchedulingPolicy> = match cfg.policy {
        cli::PolicyChoice::Shape => {
            Arc::new(scheduler::ShapeAware::new(engines[0].shape_buckets()))
        }
        cli::PolicyChoice::Fcfs => Arc::new(scheduler::Fcfs),
        cli::PolicyChoice::Priority => Arc::new(scheduler::Priority),
    };
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
                cfg.token_budget,
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
            max_input_tokens,
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
    let app = protocol::router(Arc::clone(&pool), cfg.max_body_mb * 1024 * 1024);
    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
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
