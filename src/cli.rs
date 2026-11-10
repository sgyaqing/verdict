use std::path::PathBuf;
#[derive(Debug, Clone)]
pub enum Command {
    Serve(ServeConfig),
    Download(DownloadConfig),
    Convert(ConvertConfig),
}
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub repo: String,

    pub dest: PathBuf,

    pub progress_json: bool,
}
#[derive(Debug, Clone)]
pub struct ConvertConfig {
    pub input: PathBuf,

    pub output: PathBuf,

    pub progress_json: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineChoice {
    Ort,
    Stub,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cpu,
    Cuda,
}
impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyChoice {
    Shape,
    Fcfs,
    Priority,
}
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub model_dir: PathBuf,

    pub force_download: bool,
    pub host: String,
    pub port: u16,
    pub engine: EngineChoice,
    pub device: Device,
    pub max_batch_size: usize,
    pub graph_max_len: usize,
    pub gpu_memory_utilization: f64,
    pub fork_graph_max_prefix: usize,
    pub fork_graph_max_width: usize,
    pub batch_window_ms: u64,
    pub max_input_tokens: Option<usize>,
    pub queue_cap: usize,
    pub token_budget: usize,

    pub max_body_mb: usize,

    pub served_model_name: Option<String>,

    pub policy: PolicyChoice,

    pub data_parallel: usize,
}
impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            force_download: false,
            served_model_name: None,
            policy: PolicyChoice::Shape,
            data_parallel: 1,
            host: "127.0.0.1".to_string(),
            port: 8080,
            engine: EngineChoice::Ort,
            device: Device::Auto,
            max_batch_size: 16,
            graph_max_len: 1024,
            gpu_memory_utilization: 0.92,
            fork_graph_max_prefix: 4096,
            fork_graph_max_width: 64,
            batch_window_ms: 5,
            max_input_tokens: None,
            queue_cap: 1024,
            token_budget: 16384,
            max_body_mb: 64,
        }
    }
}

pub const HELP: &str = "\
verdict -- a high-performance serving engine for System One decision models
Usage:
  verdict serve (<repo>[:variant] | <model_dir>) [flags]
  verdict download <repo> <local_dir> [--progress json]
  verdict convert <local_dir> <output_local_dir> [--progress json]
A <repo> is a hub repo id (<org>/<name>) holding a verdict-ready layout
(model.json + ONNX graph + tokenizer); repo ids auto-download into the cache
on first use. A <model_dir> is a local directory with the same layout. A repo
variant family (variants.json at its root) is served as <repo>:variant; the
bare repo name lists the variants and exits. A local variant family is served
by its variant subdirectory.
download fetches every file of any hub repo into <local_dir> (a plain snapshot for
conversion or inspection -- no verdict semantics, no variant tags). <local_dir> must
not exist or be empty; an interrupted download resumes where it left off, and
re-running a finished one reports that it is already complete.
convert turns an official encoder checkpoint (a laya repo, its multilingual/
subpackage, or a von checkpoint) into a verdict-ready package: ONNX export
under the pinned stack, tokenizer, manifest and SHA256SUMS. The output
directory must be empty; conversion needs python 3.12 with the pinned export
stack (set VERDICT_CONVERT_PYTHON or run inside the verdict-convert image).
Flags:
  --host ADDR               listen address (default 127.0.0.1)
  --port PORT               listen port (default 8080)
  --engine ort|stub         inference engine (default ort; stub = pseudo-scores)
  --force-download          re-fetch the model package instead of using the
                            cached copy (default: the cache is trusted)
  --device auto|cpu|cuda    execution device (default auto: probe CUDA, else CPU)
  --max-batch-size N        scheduler batch cap in rows (default 16)
  --graph-max-len N         longest prompt captured into a CUDA graph on the
                            torch runtime; longer prompts run eagerly, and a
                            device without room for the ladder lowers it itself
                            (default 1024)
  --gpu-memory-utilization F
                            fraction of GPU memory this instance may use
                            (weights, CUDA graphs, caches); a per-instance
                            limit, so co-located servers each declare their
                            share (default 0.92)
  --fork-graph-max-prefix N longest shared prefix the fork graphs cover
                            (default 4096)
  --fork-graph-max-width N  longest branch suffix the fork graphs cover
                            (default 64)
  --batch-window-ms MS      batch accumulation time cap; ships sooner when full
                            or after 0.5 ms of arrival silence (default 5)
  --max-input-tokens N      reject rendered prompts longer than N tokens
                            (default: off -- the model layer truncates at the
                            model's declared max_len, same as the official SDK)
  --max-body-mb N           request body size cap in MiB (default 64)
  --queue-cap N             bounded queue capacity, full -> 503 (default 1024)
  --served-model-name NAME  id advertised on /v1/models and chat (default: dir name)
  --token-budget N          tokens a batch may hold; rows per batch are capped at
                            budget / length class (default 16384)
  --policy shape|fcfs|priority
                            admission policy (default shape = group rows by length
                            class so one long row cannot pad a whole batch;
                            priority = lower first)
  --data-parallel N         engine replicas (default 1; on many-core CPU hosts
                            N>1 serves far more requests/s)
Download flags:
  --progress json           emit one JSON progress event per line on stdout
                            (file_start / progress / file_done / repo_start /
                            repo_done); logs stay on stderr. Intended for
                            tools driving verdict as a subprocess
Environment:
  HF_ENDPOINT               alternate hub endpoint
  HF_TOKEN / MODELSCOPE_API_TOKEN   gated-repo auth
  VERDICT_USE_MODELSCOPE=1  resolve repo ids against ModelScope
  VERDICT_CACHE             where downloaded packages land (default $XDG_CACHE_HOME/verdict)
  ORT_DYLIB_PATH            onnxruntime library (else model lib/ or beside the binary)
";

pub fn handled_special(args: &[String]) -> bool {
    let arg1 = args.get(1).map(String::as_str);
    let wants_help = arg1.is_some_and(|a| a == "--help" || a == "-h")
        || arg1.is_some_and(|a| a == "serve" || a == "download")
            && args[2..].iter().any(|a| a == "--help" || a == "-h");
    if wants_help {
        print!("{HELP}");
        return true;
    }
    if arg1.is_some_and(|a| a == "--version" || a == "-V") {
        println!("verdict {}", env!("CARGO_PKG_VERSION"));
        return true;
    }
    false
}
const USAGE: &str = "usage: verdict serve (<repo>[:variant] | <model_dir>) [--host ADDR] [--port PORT] \
[--engine ort|stub] [--device auto|cpu|cuda] [--max-batch-size N] [--graph-max-len N] [--gpu-memory-utilization F] \
[--fork-graph-max-prefix N] [--fork-graph-max-width N] [--batch-window-ms MS] \
[--max-input-tokens N] [--max-body-mb N] [--queue-cap N] [--token-budget N] \
[--served-model-name NAME] [--policy shape|fcfs|priority] [--data-parallel N] [--force-download]";

const DOWNLOAD_USAGE: &str = "usage: verdict download <repo> <local_dir> [--progress json]";

pub fn parse_command(args: &[String]) -> Result<Command, String> {
    match args.get(1).map(String::as_str) {
        Some("serve") => parse_args(args).map(Command::Serve),
        Some("download") => parse_download(&args[2..]).map(Command::Download),
        Some("convert") => parse_convert(&args[2..]).map(Command::Convert),
        _ => Err(format!(
            "expected the `serve`, `download` or `convert` subcommand\n{USAGE}\n{DOWNLOAD_USAGE}\n{CONVERT_USAGE}"
        )),
    }
}

const CONVERT_USAGE: &str =
    "usage: verdict convert <local_dir> <output_local_dir> [--progress json]";

fn parse_convert(rest: &[String]) -> Result<ConvertConfig, String> {
    let mut cfg = ConvertConfig {
        input: PathBuf::new(),
        output: PathBuf::new(),
        progress_json: false,
    };
    let mut got_input = false;
    let mut got_output = false;
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--progress" => {
                let value = rest
                    .next()
                    .ok_or_else(|| format!("--progress requires a value\n{CONVERT_USAGE}"))?;
                match value.as_str() {
                    "json" => cfg.progress_json = true,
                    other => {
                        return Err(format!(
                            "--progress expects json, got {other:?}\n{CONVERT_USAGE}"
                        ));
                    }
                }
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag: {flag}\n{CONVERT_USAGE}"));
            }
            input if !got_input => {
                cfg.input = PathBuf::from(input);
                got_input = true;
            }
            output if !got_output => {
                cfg.output = PathBuf::from(output);
                got_output = true;
            }
            extra => {
                return Err(format!(
                    "unexpected extra argument: {extra}\n{CONVERT_USAGE}"
                ));
            }
        }
    }
    if !got_input {
        return Err(format!(
            "a checkpoint directory is required\n{CONVERT_USAGE}"
        ));
    }
    if !got_output {
        return Err(format!("an output directory is required\n{CONVERT_USAGE}"));
    }
    Ok(cfg)
}

fn parse_download(rest: &[String]) -> Result<DownloadConfig, String> {
    let mut cfg = DownloadConfig {
        repo: String::new(),
        dest: PathBuf::new(),
        progress_json: false,
    };
    let mut got_repo = false;
    let mut got_dest = false;
    let mut rest = rest.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--progress" => {
                let value = rest
                    .next()
                    .ok_or_else(|| format!("--progress requires a value\n{DOWNLOAD_USAGE}"))?;
                match value.as_str() {
                    "json" => cfg.progress_json = true,
                    other => {
                        return Err(format!(
                            "--progress expects json, got {other:?}\n{DOWNLOAD_USAGE}"
                        ));
                    }
                }
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown flag: {flag}\n{DOWNLOAD_USAGE}"));
            }
            repo if !got_repo => {
                cfg.repo = repo.to_owned();
                got_repo = true;
            }
            dest if !got_dest => {
                cfg.dest = PathBuf::from(dest);
                got_dest = true;
            }
            extra => {
                return Err(format!(
                    "unexpected extra argument: {extra}\n{DOWNLOAD_USAGE}"
                ));
            }
        }
    }
    if !got_repo {
        return Err(format!(
            "a repo id <org>/<name> is required\n{DOWNLOAD_USAGE}"
        ));
    }
    if !got_dest {
        return Err(format!("an output directory is required\n{DOWNLOAD_USAGE}"));
    }
    Ok(cfg)
}

pub fn parse_args(args: &[String]) -> Result<ServeConfig, String> {
    let mut cfg = ServeConfig::default();
    if args.len() < 2 || args[1] != "serve" {
        return Err(format!("expected the `serve` subcommand\n{USAGE}"));
    }
    let mut rest = args[2..].iter();
    let mut got_model_dir = false;
    while let Some(arg) = rest.next() {
        let mut take_value = |flag: &str| -> Result<String, String> {
            rest.next()
                .cloned()
                .ok_or_else(|| format!("{flag} requires a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--force-download" => cfg.force_download = true,
            "--host" => cfg.host = take_value("--host")?,
            "--port" => cfg.port = parse::<u16>("--port", &take_value("--port")?)?,
            "--served-model-name" => {
                cfg.served_model_name = Some(take_value("--served-model-name")?)
            }
            "--policy" => {
                cfg.policy = match take_value("--policy")?.as_str() {
                    "shape" => PolicyChoice::Shape,
                    "fcfs" => PolicyChoice::Fcfs,
                    "priority" => PolicyChoice::Priority,
                    other => {
                        return Err(format!(
                            "--policy expects shape|fcfs|priority, got {other:?}\n{USAGE}"
                        ));
                    }
                }
            }
            "--engine" => {
                cfg.engine = match take_value("--engine")?.as_str() {
                    "ort" => EngineChoice::Ort,
                    "stub" => EngineChoice::Stub,
                    other => {
                        return Err(format!("--engine expects ort|stub, got {other:?}\n{USAGE}"));
                    }
                };
            }
            "--device" => {
                cfg.device = match take_value("--device")?.as_str() {
                    "auto" => Device::Auto,
                    "cpu" => Device::Cpu,
                    "cuda" => Device::Cuda,
                    other => {
                        return Err(format!(
                            "--device expects auto|cpu|cuda, got {other:?}\n{USAGE}"
                        ));
                    }
                };
            }
            "--max-batch-size" => {
                cfg.max_batch_size =
                    parse::<usize>("--max-batch-size", &take_value("--max-batch-size")?)?;
            }
            "--graph-max-len" => {
                let n = parse::<usize>("--graph-max-len", &take_value("--graph-max-len")?)?;
                if n < 64 {
                    return Err(format!("--graph-max-len must be >= 64\n{USAGE}"));
                }
                cfg.graph_max_len = n;
            }
            "--gpu-memory-utilization" => {
                let f = parse::<f64>(
                    "--gpu-memory-utilization",
                    &take_value("--gpu-memory-utilization")?,
                )?;
                if !f.is_finite() || !(0.0..=1.0).contains(&f) {
                    return Err(format!(
                        "--gpu-memory-utilization expects 0.0..=1.0\n{USAGE}"
                    ));
                }
                cfg.gpu_memory_utilization = f;
            }
            "--fork-graph-max-prefix" => {
                let n = parse::<usize>(
                    "--fork-graph-max-prefix",
                    &take_value("--fork-graph-max-prefix")?,
                )?;
                if n < 128 {
                    return Err(format!("--fork-graph-max-prefix must be >= 128\n{USAGE}"));
                }
                cfg.fork_graph_max_prefix = n;
            }
            "--fork-graph-max-width" => {
                let n = parse::<usize>(
                    "--fork-graph-max-width",
                    &take_value("--fork-graph-max-width")?,
                )?;
                if !(16..=512).contains(&n) {
                    return Err(format!("--fork-graph-max-width expects 16..512\n{USAGE}"));
                }
                cfg.fork_graph_max_width = n;
            }
            "--batch-window-ms" => {
                cfg.batch_window_ms =
                    parse::<u64>("--batch-window-ms", &take_value("--batch-window-ms")?)?;
            }
            "--max-input-tokens" => {
                let n = parse::<usize>("--max-input-tokens", &take_value("--max-input-tokens")?)?;
                if n == 0 {
                    return Err(format!("--max-input-tokens must be >= 1\n{USAGE}"));
                }
                cfg.max_input_tokens = Some(n);
            }
            "--queue-cap" => {
                cfg.queue_cap = parse::<usize>("--queue-cap", &take_value("--queue-cap")?)?
            }
            "--token-budget" => {
                let n = parse::<usize>("--token-budget", &take_value("--token-budget")?)?;
                if n == 0 {
                    return Err(format!("--token-budget must be >= 1\n{USAGE}"));
                }
                cfg.token_budget = n;
            }
            "--max-body-mb" => {
                let n = parse::<usize>("--max-body-mb", &take_value("--max-body-mb")?)?;
                if n == 0 {
                    return Err(format!("--max-body-mb must be >= 1\n{USAGE}"));
                }
                cfg.max_body_mb = n;
            }
            "--data-parallel" => {
                let n = parse::<usize>("--data-parallel", &take_value("--data-parallel")?)?;
                if n == 0 {
                    return Err(format!("--data-parallel must be >= 1\n{USAGE}"));
                }
                if n > 64 {
                    return Err(format!(
                        "--data-parallel is capped at 64 (each replica loads the full \
                         graph into memory); got {n}\n{USAGE}"
                    ));
                }
                cfg.data_parallel = n;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag: {flag}\n{USAGE}")),
            path if !got_model_dir => {
                cfg.model_dir = PathBuf::from(path);
                got_model_dir = true;
            }
            extra => return Err(format!("unexpected extra argument: {extra}\n{USAGE}")),
        }
    }
    if !got_model_dir {
        return Err(format!("a repo id or model directory is required\n{USAGE}"));
    }
    Ok(cfg)
}
fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("{flag} expects a number, got {value:?}\n{USAGE}"))
}
