use std::process::ExitCode;
use std::sync::Arc;

use ort::execution_providers::ExecutionProvider as _;
use verdict::cli::{Device, EngineChoice};
use verdict::engine::{Engine, EngineError, OrtEngine, StubEngine, resolve_dylib};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if verdict::cli::handled_special(&args) {
        return ExitCode::SUCCESS;
    }
    let mut cfg = match verdict::cli::parse_args(&args) {
        Ok(cfg) => cfg,
        Err(message) => {
            eprintln!("verdict: {message}");
            return ExitCode::from(2);
        }
    };
    verdict::init_tracing();

    // Model spec -> local dir, downloading HF repo ids into the cache
    // (blocking startup work, same single-threaded-phase rule as the engine).
    match verdict::model_source::resolve_forced(
        &cfg.model_dir.to_string_lossy(),
        cfg.force_download,
    ) {
        Ok(dir) => cfg.model_dir = dir,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::from(2);
        }
    }

    // Resolve the onnxruntime dylib and set ORT_DYLIB_PATH BEFORE anything
    // touches ort and before the tokio runtime: set_var must not race
    // concurrent getenv, so it happens here, single-threaded.
    if let Some(path) = resolve_dylib(&cfg.model_dir) {
        tracing::info!(dylib = %path.display(), "resolved onnxruntime library");
        // SAFETY: single-threaded startup; no other thread exists yet.
        unsafe { std::env::set_var("ORT_DYLIB_PATH", &path) };
    }

    // Blocking startup work, deliberately in the single-threaded phase (rule:
    // no blocking calls on async contexts). One engine per data-parallel
    // replica. With N > 1 the intra-op thread budget is PARTITIONED (cores/N
    // per replica): N default pools oversubscribe the cores N-fold and
    // throughput falls with N.
    let intra_threads = if cfg.data_parallel > 1 {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let threads = (cores / cfg.data_parallel).max(1);
        // OpenMP-built ORT wheels ignore the session-level thread option, so
        // the env var must be set before the first parallel region -- i.e.
        // here, before engine construction.
        // SAFETY: no other thread exists yet.
        unsafe { std::env::set_var("OMP_NUM_THREADS", threads.to_string()) };
        tracing::info!(
            replicas = cfg.data_parallel,
            cores,
            threads_per_replica = threads,
            "partitioning the CPU thread budget across data-parallel replicas"
        );
        threads
    } else {
        0 // ORT default: one pool spanning all cores
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
    // The manifest, not the CLI, picks the runtime: a `"runtime": "torch"`
    // package serves through the torch executor regardless of `--engine`
    // (the executor IS an engine choice the packager made) -- the explicit
    // `--engine stub` escape hatch aside.
    // Read cheaply and loudly: a broken manifest must fail here, not at the
    // first request.
    let manifest = std::fs::read_to_string(cfg.model_dir.join("model.json")).map_err(|err| {
        format!(
            "failed to read {}: {err}",
            cfg.model_dir.join("model.json").display()
        )
    })?;
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest).map_err(|err| format!("invalid model.json: {err}"))?;
    // Required, not defaulted: a misspelled runtime must not silently select
    // the other path and fail with a misleading error.
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
            // The stub stays an explicit escape hatch for tokenization-only
            // runs even on torch packages.
            if replica == 0 {
                tracing::warn!(
                    "--engine stub: serving deterministic pseudo-scores, NOT real inference"
                );
            }
            return Ok(Arc::new(StubEngine::new(std::time::Duration::ZERO)));
        }
        if cfg.device == Device::Cpu {
            return Err(
                "this package declares runtime \"torch\", which serves on CUDA only in v0.2 \
                 (the executor loads the checkpoint in bf16 on the GPU); use --device cuda or auto"
                    .to_string(),
            );
        }
        if replica == 0 {
            tracing::info!(
                "runtime \"torch\": serving through the python torch executor (bf16, CUDA)"
            );
        }
        let engine = verdict::torch_engine::TorchEngine::new(&cfg.model_dir)
            .map_err(|verdict::engine::EngineError(msg)| msg)?;
        return Ok(Arc::new(engine));
    }
    match (cfg.engine, cfg.device) {
        (EngineChoice::Stub, _) => {
            if replica == 0 {
                tracing::warn!(
                    "--engine stub: serving deterministic pseudo-scores, NOT real inference"
                );
            }
            Ok(Arc::new(StubEngine::new(std::time::Duration::ZERO)))
        }
        (EngineChoice::Ort, device) => {
            // `auto` probes the runtime for CUDA support (compile-level
            // check; missing driver libs still falls back with ORT's own
            // warning); `cuda` registers the CUDA EP unconditionally.
            let kind = match device {
                Device::Cuda => verdict::engine::DeviceKind::Cuda,
                Device::Cpu => verdict::engine::DeviceKind::Cpu,
                Device::Auto => {
                    // Register-level check only: ORT decides node placement at
                    // load time and may still run everything on CPU.
                    let cuda_registered = ort::execution_providers::CUDA::default()
                        .is_available()
                        .unwrap_or(false);
                    if cuda_registered {
                        tracing::info!(
                            "--device auto: the runtime has the CUDA provider; registering it \
                             (ORT can still place nodes on CPU -- ORT_LOG=verbose reports where)"
                        );
                        verdict::engine::DeviceKind::Cuda
                    } else {
                        tracing::info!(
                            "--device auto: the runtime has no CUDA provider; registering CPU"
                        );
                        verdict::engine::DeviceKind::Cpu
                    }
                }
            };
            // Detect-and-advise only -- no auto mode changes; the choice stays
            // the user's. Fires when serving lands on CPU (explicit or
            // auto-fallback) with a single replica on a big host; threshold 48
            // = where cores/24 starts recommending N >= 2.
            if replica == 0
                && cfg.data_parallel == 1
                && kind == verdict::engine::DeviceKind::Cpu
                && std::thread::available_parallelism().is_ok_and(|n| n.get() >= 48)
            {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(0);
                let suggested = (cores / 24).max(2);
                tracing::info!(
                    cores,
                    suggested_data_parallel = suggested,
                    "many-core CPU host with --data-parallel 1; on hosts like this \
                     --data-parallel ~cores/24 typically serves several times more \
                     requests/s -- sweep N to tune"
                );
            }
            // Loading a 1.5 GB graph takes a minute or two, and nothing else is
            // printed until it finishes: say so, or a first start looks hung.
            let load_started = std::time::Instant::now();
            tracing::info!(
                replica,
                model = %cfg.model_dir.display(),
                device = %kind.name(),
                "loading the model graph (a minute or two on first start; the next line comes when it is ready)"
            );
            let engine =
                match OrtEngine::with_device_and_threads(&cfg.model_dir, kind, intra_threads) {
                    Ok(engine) => engine,
                    // `auto` keeps its contract (probe CUDA, else CPU): a CUDA
                    // registration failure downgrades LOUDLY to CPU. An explicit
                    // `--device cuda` fails hard instead -- silent CPU placement
                    // is worse than an error.
                    Err(EngineError(msg))
                        if device == Device::Auto && kind == verdict::engine::DeviceKind::Cuda =>
                    {
                        tracing::warn!(
                            "--device auto: CUDA registration failed ({msg}); serving on CPU"
                        );
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
                requested = %device.name(),
                registered_ep = %kind.name(),
                model = %cfg.model_dir.display(),
                secs = load_started.elapsed().as_secs(),
                "OrtEngine ready (fp32, TF32 off on CUDA)"
            );
            if kind == verdict::engine::DeviceKind::Cuda {
                tracing::info!(
                    "a registered provider is not a placement check: ORT decides where the nodes \
                     run at load time. ORT_LOG=verbose reports it"
                );
            }
            Ok(Arc::new(engine))
        }
    }
}
