use std::path::PathBuf;
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
    pub batch_window_ms: u64,
    pub max_input_tokens: Option<usize>,
    pub queue_cap: usize,

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
            policy: PolicyChoice::Fcfs,
            data_parallel: 1,
            host: "127.0.0.1".to_string(),
            port: 8080,
            engine: EngineChoice::Ort,
            device: Device::Auto,
            max_batch_size: 8,
            batch_window_ms: 5,
            max_input_tokens: None,
            queue_cap: 1024,
            max_body_mb: 64,
        }
    }
}

pub const HELP: &str = "\
verdict -- a high-performance serving engine for System One decision models
Usage:
  verdict serve <MODEL_DIR_OR_REPO>[:<VARIANT>] [flags]
<MODEL_DIR_OR_REPO> is a local model directory or a hub repo id (<org>/<name>)
holding a verdict-ready layout (model.json + ONNX graph + tokenizer). Repo ids
auto-download into the cache on first use. A variant family (variants.json at
its root) is served as <repo>:<variant>; the bare name lists the variants and
exits.
Flags:
  --host ADDR               listen address (default 127.0.0.1)
  --port PORT               listen port (default 8080)
  --engine ort|stub         inference engine (default ort; stub = pseudo-scores)
  --force-download          re-fetch the model package instead of using the
                            cached copy (default: the cache is trusted)
  --device auto|cpu|cuda    execution device (default auto: probe CUDA, else CPU)
  --max-batch-size N        scheduler batch cap (default 8)
  --batch-window-ms MS      batch accumulation time cap; ships sooner when full
                            or after 0.5 ms of arrival silence (default 5)
  --max-input-tokens N      reject rendered prompts longer than N tokens
                            (default: off -- the model layer truncates at the
                            model's declared max_len, same as the official SDK)
  --max-body-mb N           request body size cap in MiB (default 64)
  --queue-cap N             bounded queue capacity, full -> 503 (default 1024)
  --served-model-name NAME  id advertised on /v1/models and chat (default: dir name)
  --policy fcfs|priority    admission policy (default fcfs; priority = lower first)
  --data-parallel N         engine replicas (default 1; on many-core CPU hosts
                            N>1 serves far more requests/s)
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
        || (arg1 == Some("serve") && args[2..].iter().any(|a| a == "--help" || a == "-h"));
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
const USAGE: &str = "usage: verdict serve <MODEL>[:VARIANT] [--host ADDR] [--port PORT] \
[--engine ort|stub] [--device auto|cpu|cuda] [--max-batch-size N] [--batch-window-ms MS] \
[--max-input-tokens N] [--max-body-mb N] [--queue-cap N] [--served-model-name NAME] [--policy fcfs|priority] \
[--data-parallel N] [--force-download]";

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
                    "fcfs" => PolicyChoice::Fcfs,
                    "priority" => PolicyChoice::Priority,
                    other => {
                        return Err(format!(
                            "--policy expects fcfs|priority, got {other:?}\n{USAGE}"
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
        return Err(format!("MODEL_DIR is required\n{USAGE}"));
    }
    Ok(cfg)
}
fn parse<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("{flag} expects a number, got {value:?}\n{USAGE}"))
}
