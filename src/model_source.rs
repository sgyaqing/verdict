//! Model acquisition: a model source is either a local directory or a
//! Hugging Face repo id (`<org>/<name>`) holding a verdict-ready layout
//! (model.json + the graph it names + tokenizer/tokenizer.json). Repo ids
//! auto-download into a cache and re-runs hit it -- the vLLM `serve <model>`
//! habit.
//!
//! Backends: the Hugging Face Hub (default; `HF_ENDPOINT` switches to a
//! mirror such as https://hf-mirror.com, `HF_TOKEN` authenticates gated
//! repos) and ModelScope (`VERDICT_USE_MODELSCOPE=1`;
//! `MODELSCOPE_API_TOKEN` for auth). Both hubs download through one shared
//! stall-guarded downloader (per-read timeout, bounded retries, atomic
//! `<file>.part` -> rename); the hf-hub crate is used only to READ an
//! existing standard HF cache, never to fetch. Verdict never converts
//! models (verdict-ready repos are produced offline, GGUF-style), so the
//! hub side is download-only.

use std::io::Read;
use std::path::{Path, PathBuf};

/// Which hub a repo id resolves against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubBackend {
    Hf,
    ModelScope,
}

impl HubBackend {
    /// `VERDICT_USE_MODELSCOPE=1|true` selects ModelScope (vLLM's
    /// `VLLM_USE_MODELSCOPE` pattern).
    pub fn from_env() -> Self {
        match std::env::var("VERDICT_USE_MODELSCOPE").as_deref() {
            Ok("1") | Ok("true") => HubBackend::ModelScope,
            _ => HubBackend::Hf,
        }
    }
}

/// Does this look like a repo id (`<namespace>/<name>`) rather than a path?
pub fn is_repo_id(spec: &str) -> bool {
    !spec.is_empty()
        && !spec.starts_with('/')
        && !spec.starts_with('.')
        && !spec.contains("..")
        && !spec.contains(':')
        && !spec.ends_with('/')
        && spec.matches('/').count() == 1
}

/// Resolve a CLI model spec to a local directory: an existing dir wins
/// as-is; otherwise a repo-shaped id downloads (cache-first) against the
/// env-selected backend.
pub fn resolve(spec: &str) -> Result<PathBuf, String> {
    resolve_backend(spec, HubBackend::from_env())
}

/// Backend-explicit resolve (tests pass the backend directly).
pub fn resolve_backend(spec: &str, backend: HubBackend) -> Result<PathBuf, String> {
    let local = Path::new(spec);
    if local.is_dir() {
        return Ok(local.to_path_buf());
    }
    if is_repo_id(spec) {
        return match backend {
            HubBackend::Hf => download_hf(spec),
            HubBackend::ModelScope => download_modelscope(spec),
        };
    }
    Err(format!(
        "model source {spec:?} is neither an existing directory nor a repo id \
         (expected <org>/<name> for auto-download)"
    ))
}

/// The graph file name from a downloaded manifest.
fn graph_from_manifest(manifest_path: &Path, repo_id: &str) -> Result<String, String> {
    let raw = std::fs::read_to_string(manifest_path)
        .map_err(|err| format!("cached model.json in {repo_id}: {err}"))?;
    let manifest: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid model.json in {repo_id}: {err}"))?;
    manifest
        .get("graph")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| format!("model.json in {repo_id} is missing the 'graph' field"))
}

/// Cache-only lookup for the Hugging Face backend: if the hub cache already
/// holds every file this repo needs, serve from it without contacting the hub.
///
/// Why: hf-hub has no offline switch, and its `get` re-checks the hub even on
/// a cache hit -- on a network that drops packets rather than refusing them,
/// that costs ~20 s of startup (measured). Deliberate trade-off: upstream
/// changes to a cached repo go unnoticed; these repos are released artifacts,
/// not moving targets.
fn from_hf_cache(repo_id: &str) -> Option<PathBuf> {
    let cached = hf_hub::Cache::from_env().model(repo_id.to_string());
    let manifest = cached.get("model.json")?;
    let graph = graph_from_manifest(&manifest, repo_id).ok()?;
    cached.get(&graph)?;
    cached.get("tokenizer/tokenizer.json")?;
    // `<graph>.data` is optional in ONNX and a cache hit cannot tell "repo
    // has none" from "blob deleted": an incomplete cache surfaces as an
    // onnxruntime load failure, never a silently wrong answer.
    manifest.parent().map(Path::to_path_buf)
}

/// Hugging Face backend: honors HF_ENDPOINT / HF_TOKEN. An existing standard
/// HF cache entry is served as-is; fresh downloads go through our own
/// downloader because hf-hub 0.4.3's ureq agent sets no read timeout, so its
/// downloads can stall forever.
fn download_hf(repo_id: &str) -> Result<PathBuf, String> {
    if let Some(snapshot) = from_hf_cache(repo_id) {
        tracing::info!(
            repo = repo_id,
            snapshot = %snapshot.display(),
            "serving the cached snapshot; hub not contacted"
        );
        return materialize(repo_id, &snapshot);
    }

    // HF_ENDPOINT switches the whole path to a mirror (hf-mirror.com).
    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim_end_matches('/').to_owned())
        .unwrap_or_else(|| "https://huggingface.co".to_owned());
    let bearer = std::env::var("HF_TOKEN").ok();
    let dir = verdict_cache_root()
        .join("hub")
        .join(repo_id.replace('/', "--"));

    let manifest_path =
        hf_get(&endpoint, repo_id, bearer.as_deref(), "model.json", &dir).map_err(|err| {
            format!("{err} (is this a verdict-ready repo? see the README model section)")
        })?;
    let graph = graph_from_manifest(&manifest_path, repo_id)?;
    let graph_path = hf_get(&endpoint, repo_id, bearer.as_deref(), &graph, &dir)?;
    // External weights: fetch `<graph>.data` when present (a miss means none).
    let _ = hf_get(
        &endpoint,
        repo_id,
        bearer.as_deref(),
        &format!("{graph}.data"),
        &dir,
    );
    let tokenizer_path = hf_get(
        &endpoint,
        repo_id,
        bearer.as_deref(),
        "tokenizer/tokenizer.json",
        &dir,
    )?;

    tracing::info!(
        hub = "huggingface",
        repo = repo_id,
        cache = %dir.display(),
        graph = %graph_path.display(),
        tokenizer = %tokenizer_path.display(),
        "model fetched (HF_ENDPOINT/HF_TOKEN honored)"
    );
    Ok(dir)
}

/// Cache-first fetch of one file from the Hugging Face Hub (or an
/// HF_ENDPOINT mirror) through our own timeout-guarded downloader.
fn hf_get(
    endpoint: &str,
    repo_id: &str,
    bearer: Option<&str>,
    path: &str,
    dir: &Path,
) -> Result<PathBuf, String> {
    let dest = dir.join(path);
    if dest.is_file() {
        return Ok(dest);
    }
    let url = format!("{endpoint}/{repo_id}/resolve/main/{path}");
    http_download_auth(&url, &dest, bearer)
        .map_err(|err| format!("fetching {path} from {repo_id} on Hugging Face: {err}"))?;
    Ok(dest)
}

/// Copy the whole hub snapshot (a symlink farm into blobs/) into
/// ~/.cache/verdict/hub/<org>--<name>/ as REAL files: onnxruntime's
/// external-data validation resolves symlinks and rejects weights that
/// "escape" the model directory, and `.data` must sit beside its graph.
fn materialize(repo_id: &str, snapshot: &Path) -> Result<PathBuf, String> {
    let dir = verdict_cache_root()
        .join("hub")
        .join(repo_id.replace('/', "--"));
    copy_tree(snapshot, &dir)?;
    tracing::info!(
        hub = "huggingface",
        repo = repo_id,
        snapshot = %snapshot.display(),
        cache = %dir.display(),
        "model fetched and materialized (HF_ENDPOINT/HF_TOKEN honored)"
    );
    Ok(dir)
}

/// Recursive copy that follows symlinks and skips files already in place
/// (the destination is the cache; re-runs are incremental).
fn copy_tree(src: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|err| format!("mkdir {}: {err}", dest.display()))?;
    for entry in std::fs::read_dir(src).map_err(|err| format!("read {}: {err}", src.display()))? {
        let entry = entry.map_err(|err| format!("read {}: {err}", src.display()))?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if from.is_dir() {
            copy_tree(&from, &to)?;
        } else if to.is_file() {
            continue; // cached
        } else {
            std::fs::copy(&from, &to)
                .map_err(|err| format!("copy {} -> {}: {err}", from.display(), to.display()))?;
        }
    }
    Ok(())
}

/// Verdict's own cache root: $XDG_CACHE_HOME/verdict, else ~/.cache/verdict.
fn verdict_cache_root() -> PathBuf {
    std::env::var("XDG_CACHE_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache"))
        })
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("verdict")
}

/// ModelScope backend (VERDICT_USE_MODELSCOPE=1). Download endpoint is the
/// HF-compatible `models/{org}/{name}/resolve/master/{path}` shape; the
/// cache lives under ~/.cache/verdict/modelscope/<org>/<name>/.
fn download_modelscope(repo_id: &str) -> Result<PathBuf, String> {
    let (org, name) = repo_id
        .split_once('/')
        .expect("is_repo_id guarantees one slash");
    let base = std::env::var("XDG_CACHE_HOME")
        .ok()
        .map(|d| PathBuf::from(d).join("verdict"))
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache").join("verdict"))
        })
        .unwrap_or_else(|| PathBuf::from("/tmp").join("verdict"))
        .join("modelscope");
    let dir = base.join(org).join(name);

    let manifest_path = ms_get(repo_id, "model.json", &dir)
        .map_err(|err| format!("{err} (is this repo verdict-ready AND hosted on ModelScope?)"))?;
    let graph = graph_from_manifest(&manifest_path, repo_id)?;
    let graph_path = ms_get(repo_id, &graph, &dir)?;
    // External weights: same rule as the HF backend (a miss means none).
    let _ = ms_get(repo_id, &format!("{graph}.data"), &dir);
    let tokenizer_path = ms_get(repo_id, "tokenizer/tokenizer.json", &dir)?;

    tracing::info!(
        hub = "modelscope",
        repo = repo_id,
        cache = %dir.display(),
        graph = %graph_path.display(),
        tokenizer = %tokenizer_path.display(),
        "model fetched (VERDICT_USE_MODELSCOPE honored)"
    );
    Ok(dir)
}

/// Cache-first fetch of one file from ModelScope.
fn ms_get(repo_id: &str, path: &str, dir: &Path) -> Result<PathBuf, String> {
    let dest = dir.join(path);
    if dest.is_file() {
        return Ok(dest);
    }
    let url = format!("https://modelscope.cn/models/{repo_id}/resolve/master/{path}");
    http_download(&url, &dest)
        .map_err(|err| format!("fetching {path} from {repo_id} on ModelScope: {err}"))?;
    Ok(dest)
}

/// A stalled download must fail loudly, not hang forever. Observed in the
/// wild: a ModelScope CDN connection that had already delivered the
/// whole body never sent the stream terminator, and the read blocked in
/// recvfrom indefinitely -- startup hung with no output. Every read must
/// deliver SOMETHING within this window (ureq's per-read socket timeout);
/// a slow but flowing stream is never cut.
const HTTP_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Attempts per file. Stalls and transport errors retry; 4xx does not.
const HTTP_ATTEMPTS: u32 = 3;

/// Backoff between download attempts.
const HTTP_BACKOFF_SECS: [u64; 2] = [1, 3];

/// A fetch failure, split by whether another attempt can help.
#[derive(Debug)]
pub enum FetchErr {
    /// 4xx or a local-disk error: retrying cannot change the answer.
    Terminal(String),
    /// Stall, transport, or 5xx: worth another attempt.
    Retryable(String),
}

impl FetchErr {
    pub fn message(self) -> String {
        match self {
            FetchErr::Terminal(msg) | FetchErr::Retryable(msg) => msg,
        }
    }
}

/// The agent every hub download goes through: per-read timeout set; connect
/// timeout is ureq's 30 s default. Public via `http_agent_with_read_timeout`
/// for tests; the default agent stays internal.
pub fn http_agent_with_read_timeout(timeout: std::time::Duration) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_read(timeout).build()
}

fn http_agent() -> ureq::Agent {
    http_agent_with_read_timeout(HTTP_READ_TIMEOUT)
}

/// Plain HTTPS download with retries and a progress bar (a multi-hundred-MB
/// graph must not download in silence). Redirects (CDN) are followed by ureq
/// defaults. `MODELSCOPE_API_TOKEN` rides as a Bearer header. Public for the
/// gated network test.
///
/// Cache-poisoning guard: bytes land in `<dest>.part` and are renamed into
/// place only on a complete body -- a killed download must never leave a
/// truncated file where a later run would mistake it for a cache hit.
pub fn http_download(url: &str, dest: &Path) -> Result<(), String> {
    let bearer = std::env::var("MODELSCOPE_API_TOKEN").ok();
    http_download_auth(url, dest, bearer.as_deref())
}

/// `http_download` with an explicit bearer token (the HF path passes
/// HF_TOKEN; ModelScope's env var is read by `http_download`).
pub fn http_download_auth(url: &str, dest: &Path, bearer: Option<&str>) -> Result<(), String> {
    let agent = http_agent();
    let mut last = String::new();
    for attempt in 1..=HTTP_ATTEMPTS {
        match http_download_once(&agent, url, dest, bearer) {
            Ok(()) => return Ok(()),
            Err(err) => {
                let retryable = matches!(err, FetchErr::Retryable(_));
                let msg = err.message();
                if attempt < HTTP_ATTEMPTS && retryable {
                    tracing::warn!(
                        attempt,
                        attempts = HTTP_ATTEMPTS,
                        "download failed, retrying: {msg}"
                    );
                    let backoff = HTTP_BACKOFF_SECS[(attempt - 1) as usize];
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                    last = msg;
                    continue;
                }
                return Err(msg);
            }
        }
    }
    Err(last)
}

/// One download attempt. Public (with `FetchErr` and
/// `http_agent_with_read_timeout`) so tests can drive the stall path with a
/// short timeout instead of a 30 s wait.
pub fn http_download_once(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    bearer: Option<&str>,
) -> Result<(), FetchErr> {
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    let result = download_body(agent, url, dest, &part, bearer);
    if result.is_err() {
        let _ = std::fs::remove_file(&part); // never leave a torn file around
    }
    result
}

fn download_body(
    agent: &ureq::Agent,
    url: &str,
    dest: &Path,
    part: &Path,
    bearer: Option<&str>,
) -> Result<(), FetchErr> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| FetchErr::Terminal(format!("mkdir {}: {err}", parent.display())))?;
    }
    let mut request = agent.get(url);
    if let Some(token) = bearer {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    // ureq 2.x surfaces >=400 as Err(Status(..)) and follows redirects.
    let response = request.call().map_err(|err| match err {
        ureq::Error::Status(code, _) if (400..500).contains(&code) => {
            FetchErr::Terminal(format!("{url}: HTTP {code}"))
        }
        ureq::Error::Status(code, _) => FetchErr::Retryable(format!("{url}: HTTP {code}")),
        ureq::Error::Transport(t) => FetchErr::Retryable(format!("{url}: {t}")),
    })?;
    let total = response
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let pb = indicatif::ProgressBar::new(total);
    pb.set_message(
        dest.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
    );
    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(part)
        .map_err(|err| FetchErr::Terminal(format!("create {}: {err}", part.display())))?;
    let mut buf = [0u8; 256 * 1024];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) || err.to_string().to_lowercase().contains("timed out") =>
            {
                return Err(FetchErr::Retryable(format!(
                    "read {url}: connection stalled (no data within the read timeout)"
                )));
            }
            Err(err) => return Err(FetchErr::Retryable(format!("read {url}: {err}"))),
        };
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])
            .map_err(|err| FetchErr::Terminal(format!("write: {err}")))?;
        pb.inc(n as u64);
    }
    pb.finish();
    drop(file);
    std::fs::rename(part, dest).map_err(|err| {
        FetchErr::Terminal(format!(
            "rename {} -> {}: {err}",
            part.display(),
            dest.display()
        ))
    })
}
