//! Model acquisition: a model source is either a local directory or a
//! Hugging Face repo id (`<org>/<name>`) holding a verdict-ready layout
//! (model.json + the graph it names + tokenizer/tokenizer.json). Repo ids
//! auto-download into a cache and re-runs hit it -- the vLLM `serve <model>`
//! habit.
//!
//! Backends: the Hugging Face Hub (default; `HF_ENDPOINT` switches to an
//! alternate endpoint, `HF_TOKEN` authenticates gated
//! repos) and ModelScope (`VERDICT_USE_MODELSCOPE=1`;
//! `MODELSCOPE_API_TOKEN` for auth). Both hubs download through one shared
//! stall-guarded downloader (per-read timeout, bounded retries, atomic
//! `<file>.part` -> rename). Verdict never converts models (verdict-ready
//! repos are produced offline, GGUF-style), so the hub side is download-only.
//!
//! What differs between the hubs is URL shape, default branch name and auth
//! variable -- nothing about what a package IS. All of it lives in [`Hub`],
//! and the package fetch is written once against it, so a repo of the same
//! name behaves the same on either hub. Both cache to `<cache>/<org>/<name>`
//! (`$VERDICT_CACHE` overrides the root) -- one directory per model, not per
//! hub -- and neither consults the hub's own cache: the hubs hold upstream
//! repos, whose layout is not ours.

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

/// Everything the two hubs do differently, in one place: URL shape, default
/// branch name, auth variable, cache layout, and how a missing file reports
/// itself. The package fetch is written ONCE against this. A per-backend
/// split drifts: one hub silently ends up without a package shape the other
/// supports.
#[derive(Debug, Clone)]
pub struct Hub {
    backend: HubBackend,
    /// Base URL, no trailing slash. For ModelScope the `/models` prefix is
    /// part of the base, not of the file path.
    endpoint: String,
    bearer: Option<String>,
    /// `--force-download`: re-fetch every file and ignore the completion
    /// marker, so an updated or name-colliding upstream package is picked up.
    /// Named after `huggingface_hub`'s `force_download`, which transformers and
    /// vLLM carry too; ModelScope spells the same thing `--force`.
    force: bool,
}

impl Hub {
    pub fn for_backend(backend: HubBackend) -> Self {
        match backend {
            HubBackend::Hf => {
                // HF_ENDPOINT switches the whole path to an alternate
                // endpoint; HF_TOKEN authenticates gated repos.
                let endpoint = std::env::var("HF_ENDPOINT")
                    .ok()
                    .map(|e| e.trim_end_matches('/').to_owned())
                    .unwrap_or_else(|| "https://huggingface.co".to_owned());
                Hub {
                    backend,
                    endpoint,
                    bearer: std::env::var("HF_TOKEN").ok(),
                    force: false,
                }
            }
            HubBackend::ModelScope => Hub {
                backend,
                endpoint: "https://modelscope.cn/models".to_owned(),
                bearer: std::env::var("MODELSCOPE_API_TOKEN").ok(),
                force: false,
            },
        }
    }

    /// An explicitly addressed hub, for a caller the environment does not
    /// name -- tests point this at a throwaway local server.
    pub fn with_endpoint(
        backend: HubBackend,
        endpoint: impl Into<String>,
        bearer: Option<String>,
    ) -> Self {
        Hub {
            backend,
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            bearer,
            force: false,
        }
    }

    /// `--force-download`.
    pub fn forced(mut self, on: bool) -> Self {
        self.force = on;
        self
    }

    pub fn backend(&self) -> HubBackend {
        self.backend
    }

    /// The branch a file resolves to when the manifest pins no revision. The
    /// two hubs name their default branch differently, which is why the pin
    /// is optional: one manifest has to serve both.
    pub fn default_branch(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "main",
            HubBackend::ModelScope => "master",
        }
    }

    /// The name to use in a message the user reads.
    fn display(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "Hugging Face",
            HubBackend::ModelScope => "ModelScope",
        }
    }

    /// The name to use in a log field.
    fn slug(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "huggingface",
            HubBackend::ModelScope => "modelscope",
        }
    }

    /// What to add when the manifest itself could not be fetched -- the two
    /// hubs fail for different reasons and the fix differs.
    fn manifest_hint(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "is this a verdict-ready repo? see the README model section",
            HubBackend::ModelScope => "is this repo verdict-ready AND hosted on ModelScope?",
        }
    }

    /// What to add when the official weights could not be fetched.
    ///
    /// The manifest names the checkpoint as it is on HUGGING FACE; on
    /// ModelScope the same id is asked for. Vendors do not always mirror their
    /// repo there under that name, and when they do not, nothing on our side
    /// can fix it -- so the message has to tell the user which repo is being
    /// looked for where, and what to do instead. Reporting it to the package
    /// maintainer would be the wrong advice for that case.
    fn checkpoint_hint(&self, repo: &str) -> String {
        match self.backend {
            HubBackend::Hf => format!(
                "the checkpoint lives in the official repo {repo}, not in this package; \
                 a missing file means it moved upstream -- report it to the package maintainer"
            ),
            HubBackend::ModelScope => format!(
                "these weights come from the official repo {repo}, which may not be mirrored \
                 on ModelScope under that name; serve it from Hugging Face instead \
                 (drop VERDICT_USE_MODELSCOPE)"
            ),
        }
    }

    fn file_url(&self, repo_id: &str, revision: &str, path: &str) -> String {
        format!("{}/{repo_id}/resolve/{revision}/{path}", self.endpoint)
    }

    /// Where this repo's files land: `<cache>/<org>/<name>`.
    ///
    /// Keyed by repo id alone, NOT by hub: the two hubs are held to the same
    /// content (an operational rule, not an assumption this code can check), so
    /// the same repo id names one artifact however it is fetched. A repo that
    /// exists on only one hub is unaffected either way.
    fn cache_dir(&self, repo_id: &str) -> PathBuf {
        // The model NAME, not the repo id: the two hubs are held to the same
        // content (an operational rule), so `verdict-models/von` on one and
        // `example-org/von` on the other are one artifact and must share a slot.
        // Two DIFFERENT repos that happen to share a name would collide -- a
        // refresh/override flag is the escape hatch, and is not built yet.
        let (_, name) = repo_id
            .split_once('/')
            .expect("is_repo_id guarantees one slash");
        verdict_cache_root().join(name)
    }

    /// Cache-first fetch of one file through the shared stall-guarded
    /// downloader. An already-present file is a hit and touches no network.
    pub fn get(
        &self,
        repo_id: &str,
        revision: &str,
        path: &str,
        dir: &Path,
    ) -> Result<PathBuf, String> {
        let dest = dir.join(path);
        if dest.is_file() && !self.force {
            return Ok(dest);
        }
        let url = self.file_url(repo_id, revision, path);
        match http_download_auth(&url, &dest, self.bearer.as_deref()) {
            Ok(()) => Ok(dest),
            // ModelScope answers a missing file with HTTP 500, not 404.
            // 500 stays retryable at the http_download
            // layer; here a 500 that survived every retry is "absent".
            Err(err) if self.backend == HubBackend::ModelScope && err.contains("HTTP 500") => {
                Err(format!(
                    "fetching {path} from {repo_id} on ModelScope: {err} \
                     (ModelScope reports 500 for missing files -- likely absent)"
                ))
            }
            Err(err) => Err(format!(
                "fetching {path} from {repo_id} on {}: {err}",
                self.display()
            )),
        }
    }

    /// Optional-file probe: a single attempt, no retries -- ModelScope answers
    /// an absent file with HTTP 500, and the retry ladder would burn its full
    /// backoff budget on every first pull of a repo without that file.
    pub fn get_optional(
        &self,
        repo_id: &str,
        revision: &str,
        path: &str,
        dir: &Path,
    ) -> Result<PathBuf, String> {
        let dest = dir.join(path);
        if dest.is_file() && !self.force {
            return Ok(dest);
        }
        let url = self.file_url(repo_id, revision, path);
        let agent = http_agent();
        match http_download_once(&agent, &url, &dest, self.bearer.as_deref()) {
            Ok(()) => Ok(dest),
            Err(_) => Err(format!("optional {path} absent")),
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

/// `verdict serve --force-download`: same as [`resolve`], but every file is
/// re-fetched and the completion marker is ignored, so a package that changed
/// upstream -- or one sharing its name with a different repo -- is picked up.
pub fn resolve_forced(spec: &str, force: bool) -> Result<PathBuf, String> {
    resolve_with(spec, HubBackend::from_env(), force)
}

/// Backend-explicit resolve (tests pass the backend directly).
pub fn resolve_backend(spec: &str, backend: HubBackend) -> Result<PathBuf, String> {
    resolve_with(spec, backend, false)
}

fn resolve_with(spec: &str, backend: HubBackend, force: bool) -> Result<PathBuf, String> {
    let local = Path::new(spec);
    if local.is_dir() {
        return Ok(local.to_path_buf());
    }
    if is_repo_id(spec) {
        return download(&Hub::for_backend(backend).forced(force), spec);
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

/// Fetch a verdict-ready package from `hub`. Two shapes, decided by the
/// manifest alone -- not by which hub it came from:
///
/// - a graph package ships the ONNX graph it names;
/// - a checkpoint package (torch runtime) ships only the manifest and fetches
///   its weights from the official repo it declares.
///
/// Every file is fetched through [`Hub::get`], which serves a file already in
/// our own cache without contacting the hub. A completed package carries a
/// [`COMPLETE`] marker and short-circuits the whole walk, so a warm start
/// makes no request at all -- without it the optional `<graph>.data` probe
/// would ask every time (`<graph>.data` is optional in ONNX, and its absence
/// from the cache cannot be told from "the repo has none"). The hubs' own
/// caches are never consulted: a verdict-ready repo is not the upstream repo,
/// and only the former has the manifest we need.
pub fn download(hub: &Hub, repo_id: &str) -> Result<PathBuf, String> {
    // A package's OWN files are always read at the hub's default branch; only
    // the checkpoint names a revision.
    let dir = hub.cache_dir(repo_id);
    if dir.join(COMPLETE).is_file() && !hub.force {
        tracing::info!(
            hub = hub.slug(),
            repo = repo_id,
            cache = %dir.display(),
            "serving from the cache; hub not contacted"
        );
        return Ok(dir);
    }
    let branch = hub.default_branch();
    // One line when a download actually starts. A warm cache returned above, so
    // this only prints when the hub is about to be hit.
    tracing::info!(
        hub = hub.slug(),
        repo = repo_id,
        cache = %dir.display(),
        "downloading"
    );

    let manifest_path = hub
        .get(repo_id, branch, "model.json", &dir)
        .map_err(|err| format!("{err} ({})", hub.manifest_hint()))?;
    let manifest_raw = std::fs::read_to_string(&manifest_path)
        .map_err(|err| format!("read {}: {err}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_raw)
        .map_err(|err| format!("invalid model.json in {repo_id}: {err}"))?;

    if let Some(decl) = checkpoint_decl(&manifest) {
        let decl = decl.map_err(|err| format!("invalid model.json in {repo_id}: {err}"))?;
        let tokenizer_path = hub.get(repo_id, branch, "tokenizer/tokenizer.json", &dir)?;
        fetch_checkpoint(hub, &dir, &decl)?;
        tracing::info!(
            hub = hub.slug(),
            repo = repo_id,
            checkpoint = %decl.repo,
            revision = %decl.revision_on(hub),
            cache = %dir.display(),
            tokenizer = %tokenizer_path.display(),
            "package fetched; weights fetched from the official repo"
        );
        write_complete(&dir)?;
        return Ok(dir);
    }

    let graph = graph_from_manifest(&manifest_path, repo_id)?;
    let graph_path = hub.get(repo_id, branch, &graph, &dir)?;
    // External weights: fetch `<graph>.data` when present (a miss means none).
    let _ = hub.get_optional(repo_id, branch, &format!("{graph}.data"), &dir);
    let tokenizer_path = hub.get(repo_id, branch, "tokenizer/tokenizer.json", &dir)?;

    tracing::info!(
        hub = hub.slug(),
        repo = repo_id,
        cache = %dir.display(),
        graph = %graph_path.display(),
        tokenizer = %tokenizer_path.display(),
        "model fetched"
    );
    write_complete(&dir)?;
    Ok(dir)
}

/// Marker file in a package directory: every file this package needs is here.
/// Written LAST, so a run killed mid-fetch leaves it absent and the next start
/// resumes file by file. Its presence is the only completeness claim we make --
/// delete it to force a re-check against the hub.
const COMPLETE: &str = ".complete";

fn write_complete(dir: &Path) -> Result<(), String> {
    std::fs::write(dir.join(COMPLETE), b"")
        .map_err(|err| format!("marking {} complete: {err}", dir.display()))
}

/// A manifest's checkpoint declaration: weights that live in an OFFICIAL repo
/// (the packager names the repo and the exact file list; the package itself
/// ships only manifest/tokenizer/checksums, so no model weights are
/// redistributed by us).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointDecl {
    pub repo: String,
    /// The revision the packager pinned, if any. `None` means the hub's own
    /// default branch, which is what lets one manifest serve both hubs: HF
    /// has no branch by that name on ModelScope and vice versa.
    pub revision: Option<String>,
    pub files: Vec<String>,
}

impl CheckpointDecl {
    /// The revision to fetch at, on this hub.
    pub fn revision_on(&self, hub: &Hub) -> &str {
        self.revision.as_deref().unwrap_or(hub.default_branch())
    }
}

/// Parse the `"checkpoint"` field of a model manifest. Present = a
/// torch-runtime package whose weights must be fetched beside model.json.
pub fn checkpoint_decl(manifest: &serde_json::Value) -> Option<Result<CheckpointDecl, String>> {
    let decl = manifest.get("checkpoint")?;
    let bad = |msg: String| Some(Err(msg));
    let obj = match decl.as_object() {
        Some(o) => o,
        None => {
            return bad(
                "model.json 'checkpoint' must be an object: {\"repo\", \"files\" \
                 [, \"revision\"]}"
                    .to_string(),
            );
        }
    };
    let repo = match obj.get("repo").and_then(|v| v.as_str()) {
        Some(r) if is_repo_id(r) => r.to_string(),
        _ => return bad("model.json checkpoint.repo must be an <org>/<name> repo id".to_string()),
    };
    let revision = match obj.get("revision") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_str() {
            Some(r) if !r.is_empty() => Some(r.to_string()),
            // An empty or non-string revision would silently become
            // `resolve//<path>` and 404 -- refuse it here instead.
            _ => {
                return bad(
                    "model.json checkpoint.revision must be a non-empty string when present"
                        .to_string(),
                );
            }
        },
    };
    let files: Vec<String> = match obj.get("files").and_then(|v| v.as_array()) {
        Some(list) => list
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    "model.json checkpoint.files entries must be strings".to_string()
                })
            })
            .collect::<Result<_, _>>()
            .ok()?,
        None => {
            return bad(
                "model.json checkpoint.files must list the files to fetch (config.json, \
                 weights, ...)"
                    .to_string(),
            );
        }
    };
    if files.is_empty() {
        return bad("model.json checkpoint.files is empty".to_string());
    }
    // A path escape in the file list would write outside the package dir.
    if files
        .iter()
        .any(|f| f.is_empty() || f.starts_with('/') || f.starts_with('.') || f.contains(".."))
    {
        return bad("model.json checkpoint.files entries must be plain relative paths".to_string());
    }
    Some(Ok(CheckpointDecl {
        repo,
        revision,
        files,
    }))
}

/// Fetch the declared checkpoint files into the package directory (flat,
/// beside model.json -- the layout TorchEngine and the HF cache both
/// expect). Idempotent per file: an already-present file is a cache hit and
/// touches no network.
pub fn fetch_checkpoint(hub: &Hub, dir: &Path, decl: &CheckpointDecl) -> Result<(), String> {
    let revision = decl.revision_on(hub);
    let mut last: Option<PathBuf> = None;
    for file in &decl.files {
        last = Some(
            hub.get(&decl.repo, revision, file, dir)
                .map_err(|err| format!("{err} ({})", hub.checkpoint_hint(&decl.repo)))?,
        );
    }
    if let Some(p) = last {
        tracing::info!(
            hub = hub.slug(),
            repo = %decl.repo,
            revision = %revision,
            files = decl.files.len(),
            cache = %p.parent().map(|d| d.display().to_string()).unwrap_or_default(),
            "checkpoint fetched from the official repo"
        );
    }
    Ok(())
}

/// Verdict's own cache root: $VERDICT_CACHE if set (the container case -- one
/// mount point holds every model), else $XDG_CACHE_HOME/verdict, else
/// ~/.cache/verdict.
fn verdict_cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("VERDICT_CACHE") {
        return PathBuf::from(dir);
    }
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
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    pb.set_message(name.clone());
    // The bar itself is invisible when stderr is not a terminal -- `docker logs`
    // is a pipe, so inside a container the user would otherwise watch a silent
    // multi-GB download and conclude it hung. Log every file, the way docker
    // pull logs every layer and pip every package; a package is a handful of
    // files, so this is a dozen lines, not noise.

    let started = std::time::Instant::now();
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
    tracing::info!(
        file = %name,
        bytes = pb.position(),
        secs = started.elapsed().as_secs(),
        "downloaded"
    );
    drop(file);
    std::fs::rename(part, dest).map_err(|err| {
        FetchErr::Terminal(format!(
            "rename {} -> {}: {err}",
            part.display(),
            dest.display()
        ))
    })
}
