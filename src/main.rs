use ort::execution_providers::ExecutionProvider as _;
use std::process::ExitCode;
use std::sync::Arc;
use verdict::cli::{Device, EngineChoice};
use verdict::engine::{Engine, EngineError, OrtEngine, StubEngine, resolve_dylib};
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if verdict::cli::handled_special(&args) {
        return ExitCode::SUCCESS;
    }
    let mut cfg = match verdict::cli::parse_command(&args) {
        Ok(verdict::cli::Command::Serve(cfg)) => cfg,
        Ok(verdict::cli::Command::Download(dl)) => {
            verdict::init_tracing_stderr();
            if dl.progress_json {
                verdict::progress::install_json();
            }
            return match verdict::model_source::download_command(&dl.repo, &dl.dest) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::from(2)
                }
            };
        }
        Ok(verdict::cli::Command::Convert(cv)) => {
            verdict::init_tracing_stderr();
            if cv.progress_json {
                verdict::progress::install_json();
            }
            return match verdict::convert::convert_command(&cv.input, &cv.output) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::from(2)
                }
            };
        }
        Err(message) => {
            eprintln!("verdict: {message}");
            return ExitCode::from(2);
        }
    };
    verdict::init_tracing();
    let spec = cfg.model_dir.to_string_lossy().into_owned();
    match verdict::model_source::resolve_forced(&spec, cfg.force_download) {
        Ok(dir) => cfg.model_dir = dir,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(2);
        }
    }
    if cfg.served_model_name.is_none()
        && let Ok((base, Some(tag))) = verdict::model_source::split_spec(&spec)
    {
        let name = base.rsplit('/').next().unwrap_or(base);
        cfg.served_model_name = Some(format!("{name}:{tag}"));
    }
    let compile_cache = verdict::model_source::compile_cache_root(&cfg.model_dir, &spec);
    unsafe {
        std::env::set_var("VERDICT_GRAPH_ROWS_CAP", cfg.max_batch_size.to_string());
        std::env::set_var("VERDICT_GRAPH_MAX_LEN", cfg.graph_max_len.to_string());
        std::env::set_var(
            "VERDICT_GPU_MEMORY_UTILIZATION",
            (cfg.gpu_memory_utilization / cfg.data_parallel.max(1) as f64).to_string(),
        );
        std::env::set_var(
            "VERDICT_FORK_GRAPH_MAX_PREFIX",
            cfg.fork_graph_max_prefix.to_string(),
        );
        std::env::set_var(
            "VERDICT_FORK_GRAPH_MAX_WIDTH",
            cfg.fork_graph_max_width.to_string(),
        );
        std::env::set_var("VERDICT_COMPILE_CACHE", compile_cache);
        std::env::set_var(
            "VERDICT_EXECUTOR_SHA",
            verdict::torch_engine::executor_sha(),
        );
    }
    if let Some(path) = resolve_dylib(&cfg.model_dir) {
        tracing::info!(dylib = %path.display(), "resolved onnxruntime library");
        unsafe { std::env::set_var("ORT_DYLIB_PATH", &path) };
    }
    let intra_threads = if cfg.data_parallel > 1 {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = (cores / cfg.data_parallel).max(1);
        unsafe { std::env::set_var("OMP_NUM_THREADS", threads.to_string()) };
        tracing::info!(
            replicas = cfg.data_parallel,
            cores,
            threads_per_replica = threads,
            "partitioning the CPU thread budget across data-parallel replicas"
        );
        threads
    } else {
        0
    };
    let mut engines: Vec<Arc<dyn Engine>> = Vec::with_capacity(cfg.data_parallel);
    for replica in 0..cfg.data_parallel {
        match build_engine(&cfg, replica, intra_threads) {
            Ok(engine) => engines.push(engine),
            Err(err) => {
                tracing::error!(%err, "failed to initialize the inference engine");
                return ExitCode::FAILURE;
            }
        }
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("verdict: failed to start tokio runtime: {err}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(err) = runtime.block_on(verdict::run_server(cfg, engines)) {
        tracing::error!(%err, "server exited with error");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
fn build_engine(
    cfg: &verdict::cli::ServeConfig,
    replica: usize,
    intra_threads: usize,
) -> Result<Arc<dyn Engine>, String> {
    let manifest =
        verdict::manifest::read_manifest(&cfg.model_dir).map_err(|err| err.to_string())?;
    let runtime = match manifest.get("runtime").and_then(|v| v.as_str()) {
        Some(r @ ("ort" | "torch")) => r,
        Some(other) => {
            return Err(format!(
                "model.json declares unknown runtime {other:?} (this build knows: \"ort\", \"torch\")"
            ));
        }
        None => {
            return Err(
                "model.json is missing the 'runtime' field (\"ort\" or \"torch\")".to_string(),
            );
        }
    };
    if runtime == "torch" {
        if cfg.engine == EngineChoice::Stub {
            if replica == 0 {
                tracing::warn!("engine stub: pseudo-scores, not real inference");
            }
            return Ok(Arc::new(StubEngine::new(std::time::Duration::ZERO)));
        }
        let device = match cfg.device {
            Device::Cpu => {
                if replica == 0 {
                    tracing::info!(
                        "runtime \"torch\" on cpu: serving through the python torch \
                         executor (fp32, no graphs)"
                    );
                }
                "cpu"
            }
            _ => {
                if replica == 0 {
                    tracing::info!(
                        "runtime \"torch\": serving through the python torch executor \
                         (bf16, CUDA)"
                    );
                }
                "cuda"
            }
        };
        let engine =
            verdict::torch_engine::TorchEngine::with_device(&cfg.model_dir, &cfg.model_dir, device)
                .map_err(|verdict::engine::EngineError(msg)| msg)?;
        return Ok(Arc::new(engine));
    }
    match (cfg.engine, cfg.device) {
        (EngineChoice::Stub, _) => {
            if replica == 0 {
                tracing::warn!("engine stub: pseudo-scores, not real inference");
            }
            Ok(Arc::new(StubEngine::new(std::time::Duration::ZERO)))
        }
        (EngineChoice::Ort, device) => {
            let kind = match device {
                Device::Cuda => verdict::engine::DeviceKind::Cuda,
                Device::Cpu => verdict::engine::DeviceKind::Cpu,
                Device::Auto => {
                    let cuda_registered = ort::execution_providers::CUDA::default()
                        .is_available()
                        .unwrap_or(false);
                    if cuda_registered {
                        tracing::info!("device auto: cuda");
                        verdict::engine::DeviceKind::Cuda
                    } else {
                        tracing::info!("device auto: cpu");
                        verdict::engine::DeviceKind::Cpu
                    }
                }
            };
            if replica == 0
                && cfg.data_parallel == 1
                && kind == verdict::engine::DeviceKind::Cpu
                && std::thread::available_parallelism().is_ok_and(|n| n.get() >= 48)
            {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(0);
                let suggested = (cores / 24).max(2);
                tracing::info!("note: --data-parallel ~{suggested} on this host");
            }
            let load_started = std::time::Instant::now();
            tracing::info!(
                replica,
                model = %cfg.model_dir.display(),
                "loading the model graph"
            );
            let engine =
                match OrtEngine::with_device_and_threads(&cfg.model_dir, kind, intra_threads) {
                    Ok(engine) => engine,
                    Err(EngineError(msg))
                        if device == Device::Auto && kind == verdict::engine::DeviceKind::Cuda =>
                    {
                        tracing::warn!("device auto: cuda failed ({msg}); serving on cpu");
                        OrtEngine::with_device_and_threads(
                            &cfg.model_dir,
                            verdict::engine::DeviceKind::Cpu,
                            intra_threads,
                        )
                        .map_err(|EngineError(msg)| msg)?
                    }
                    Err(EngineError(msg)) => return Err(msg),
                };
            tracing::info!(
                replica,
                secs = load_started.elapsed().as_secs(),
                "model ready"
            );
            Ok(Arc::new(engine))
        }
    }
}
