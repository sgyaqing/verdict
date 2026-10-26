use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HubBackend {
    Hf,
    ModelScope,
}
impl HubBackend {
    pub fn from_env() -> Self {
        match std::env::var("VERDICT_USE_MODELSCOPE").as_deref() {
            Ok("1") | Ok("true") => HubBackend::ModelScope,
            _ => HubBackend::Hf,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hub {
    backend: HubBackend,

    endpoint: String,
    bearer: Option<String>,

    force: bool,
}
impl Hub {
    pub fn for_backend(backend: HubBackend) -> Self {
        match backend {
            HubBackend::Hf => {
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

    pub fn forced(mut self, on: bool) -> Self {
        self.force = on;
        self
    }
    pub fn backend(&self) -> HubBackend {
        self.backend
    }

    pub fn default_branch(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "main",
            HubBackend::ModelScope => "master",
        }
    }

    fn display(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "Hugging Face",
            HubBackend::ModelScope => "ModelScope",
        }
    }

    fn slug(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "huggingface",
            HubBackend::ModelScope => "modelscope",
        }
    }

    fn manifest_hint(&self) -> &'static str {
        match self.backend {
            HubBackend::Hf => "is this a verdict-ready repo? see the README model section",
            HubBackend::ModelScope => "is this repo verdict-ready AND hosted on ModelScope?",
        }
    }

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

    fn cache_dir(&self, repo_id: &str) -> PathBuf {
        let (_, name) = repo_id
            .split_once('/')
            .expect("is_repo_id guarantees one slash");
        verdict_cache_root().join(name)
    }

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

    pub fn list_repo_files(&self, repo_id: &str, revision: &str) -> Result<Vec<String>, String> {
        match self.backend {
            HubBackend::Hf => self.list_repo_files_hf(repo_id, revision),
            HubBackend::ModelScope => self.list_repo_files_ms(repo_id, revision),
        }
    }

    fn list_repo_files_hf(&self, repo_id: &str, revision: &str) -> Result<Vec<String>, String> {
        let agent = http_agent();
        let mut url = format!(
            "{}/api/models/{repo_id}/tree/{revision}?recursive=true",
            self.endpoint
        );
        let mut out = Vec::new();
        loop {
            let mut request = agent.get(&url);
            if let Some(token) = &self.bearer {
                request = request.set("Authorization", &format!("Bearer {token}"));
            }
            let response = request.call().map_err(|err| {
                format!("listing files of {repo_id} on {}: {err}", self.display())
            })?;
            let link = response.header("Link").map(str::to_owned);
            let text = response.into_string().map_err(|err| {
                format!("listing files of {repo_id} on {}: {err}", self.display())
            })?;
            let body: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
                format!("listing files of {repo_id} on {}: {err}", self.display())
            })?;
            let entries = body.as_array().ok_or_else(|| {
                format!(
                    "listing files of {repo_id} on {}: the tree response is not an array",
                    self.display()
                )
            })?;
            for entry in entries {
                if entry.get("type").and_then(|v| v.as_str()) != Some("file") {
                    continue;
                }
                if let Some(path) = entry.get("path").and_then(|v| v.as_str()) {
                    out.push(path.to_string());
                }
            }
            let next = link.as_deref().and_then(|l| {
                l.split(',')
                    .map(str::trim)
                    .find(|part| part.ends_with("rel=\"next\""))
                    .and_then(|part| part.split(';').next())
                    .map(|u| u.trim().trim_matches(|c| c == '<' || c == '>').to_string())
            });
            match next {
                Some(u) if !u.is_empty() => url = u,
                _ => break,
            }
        }
        Ok(out)
    }

    fn list_repo_files_ms(&self, repo_id: &str, revision: &str) -> Result<Vec<String>, String> {
        let base = self
            .endpoint
            .strip_suffix("/models")
            .unwrap_or(&self.endpoint);
        let url =
            format!("{base}/api/v1/models/{repo_id}/repo/files?Revision={revision}&Recursive=true");
        let agent = http_agent();
        let mut request = agent.get(&url);
        if let Some(token) = &self.bearer {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = request
            .call()
            .map_err(|err| format!("listing files of {repo_id} on ModelScope: {err}"))?;
        let text = response
            .into_string()
            .map_err(|err| format!("listing files of {repo_id} on ModelScope: {err}"))?;
        let body: serde_json::Value = serde_json::from_str(&text)
            .map_err(|err| format!("listing files of {repo_id} on ModelScope: {err}"))?;
        let files = body
            .get("Data")
            .and_then(|d| d.get("Files"))
            .and_then(|f| f.as_array())
            .ok_or_else(|| {
                format!("listing files of {repo_id} on ModelScope: no Data.Files array")
            })?;
        let mut out = Vec::new();
        for entry in files {
            if entry.get("Type").and_then(|v| v.as_str()) != Some("blob") {
                continue;
            }
            if let Some(path) = entry.get("Path").and_then(|v| v.as_str()) {
                out.push(path.to_string());
            }
        }
        Ok(out)
    }
}

pub fn is_repo_id(spec: &str) -> bool {
    !spec.is_empty()
        && !spec.starts_with('/')
        && !spec.starts_with('.')
        && !spec.contains("..")
        && !spec.contains(':')
        && !spec.ends_with('/')
        && spec.matches('/').count() == 1
}

pub const VARIANTS_INDEX: &str = "variants.json";

pub fn split_spec(spec: &str) -> Result<(&str, Option<String>), String> {
    let Some((head, tag)) = spec.rsplit_once(':') else {
        return Ok((spec, None));
    };
    if head.is_empty() || (!Path::new(head).is_dir() && !is_repo_id(head)) {
        return Ok((spec, None));
    }
    let tag = tag.to_lowercase();
    if !valid_variant_name(&tag) {
        return Err(format!(
            "invalid variant tag in {spec:?} (a tag matches [a-z0-9][a-z0-9._-], up to 64 chars)"
        ));
    }
    Ok((head, Some(tag)))
}
fn valid_variant_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

fn parse_variants_index(raw: &str, source: &str) -> Result<Vec<String>, String> {
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|err| format!("invalid {VARIANTS_INDEX} in {source}: {err}"))?;
    let list = value
        .get("variants")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{VARIANTS_INDEX} in {source} is missing the 'variants' array"))?;
    let mut out: Vec<String> = Vec::with_capacity(list.len());
    for item in list {
        let name = item.as_str().ok_or_else(|| {
            format!("{VARIANTS_INDEX} in {source}: variant names must be strings")
        })?;
        if !valid_variant_name(name) {
            return Err(format!(
                "{VARIANTS_INDEX} in {source}: invalid variant name {name:?} \
                 ([a-z0-9][a-z0-9._-], up to 64 chars)"
            ));
        }
        if out.iter().any(|seen| seen == name) {
            return Err(format!(
                "{VARIANTS_INDEX} in {source}: duplicate variant {name:?}"
            ));
        }
        out.push(name.to_string());
    }
    if out.is_empty() {
        return Err(format!("{VARIANTS_INDEX} in {source}: 'variants' is empty"));
    }
    Ok(out)
}

fn variant_listing(source: &str, variants: &[String], cached: impl Fn(&str) -> bool) -> String {
    let lines: Vec<String> = variants
        .iter()
        .map(|name| {
            if cached(name) {
                format!("  {name} (cached)")
            } else {
                format!("  {name}")
            }
        })
        .collect();
    format!(
        "{source} is a variant family; pick one with {source}:<variant>\n\
         available:\n{}\n\
         example: verdict serve {source}:{}",
        lines.join("\n"),
        variants[0]
    )
}

pub fn resolve(spec: &str) -> Result<PathBuf, String> {
    resolve_backend(spec, HubBackend::from_env())
}

pub fn resolve_forced(spec: &str, force: bool) -> Result<PathBuf, String> {
    resolve_with(spec, HubBackend::from_env(), force)
}

pub fn resolve_backend(spec: &str, backend: HubBackend) -> Result<PathBuf, String> {
    resolve_with(spec, backend, false)
}
fn resolve_with(spec: &str, backend: HubBackend, force: bool) -> Result<PathBuf, String> {
    let (base, tag) = split_spec(spec)?;
    let local = Path::new(base);
    if local.is_dir() {
        return resolve_local(local, tag.as_deref());
    }
    if is_repo_id(base) {
        return download(
            &Hub::for_backend(backend).forced(force),
            base,
            tag.as_deref(),
        );
    }
    Err(format!(
        "model source {spec:?} is neither an existing directory nor a repo id \
         (expected <org>/<name> for auto-download)"
    ))
}

fn resolve_local(dir: &Path, tag: Option<&str>) -> Result<PathBuf, String> {
    let has_manifest = dir.join("model.json").is_file();
    let has_index = dir.join(VARIANTS_INDEX).is_file();
    match (has_manifest, has_index) {
        (true, true) => Err(format!(
            "{} has both model.json and {VARIANTS_INDEX} -- a package is either \
             plain or a variant family, not both",
            dir.display()
        )),
        (true, false) => match tag {
            None => Ok(dir.to_path_buf()),
            Some(v) => Err(format!(
                "{} has no variants (its model.json is a plain package); \
                 serve it without :{v}",
                dir.display()
            )),
        },
        (false, true) => {
            let index_path = dir.join(VARIANTS_INDEX);
            let raw = std::fs::read_to_string(&index_path)
                .map_err(|err| format!("read {}: {err}", index_path.display()))?;
            let variants = parse_variants_index(&raw, &dir.display().to_string())?;
            let Some(v) = tag else {
                return Err(variant_listing(
                    &dir.display().to_string(),
                    &variants,
                    |name| dir.join(name).join(COMPLETE).is_file(),
                ));
            };
            if !variants.iter().any(|name| name == v) {
                return Err(format!(
                    "unknown variant {v:?} for {} (available: {})",
                    dir.display(),
                    variants.join(", ")
                ));
            }
            let sub = dir.join(v);
            if !sub.join("model.json").is_file() {
                return Err(format!(
                    "variant {v:?} of {} has no model.json in {}",
                    dir.display(),
                    sub.display()
                ));
            }
            Ok(sub)
        }
        (false, false) => Ok(dir.to_path_buf()),
    }
}

fn graph_from_manifest(manifest_path: &Path, repo_id: &str) -> Result<String, String> {
    let raw = std::fs::read_to_string(manifest_path)
        .map_err(|err| format!("cached model.json in {repo_id}: {err}"))?;
    let manifest = crate::manifest::parse_manifest(&raw)
        .map_err(|err| format!("{} (model.json in {repo_id})", err))?;
    manifest
        .get("graph")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| format!("model.json in {repo_id} is missing the 'graph' field"))
}

pub fn download(hub: &Hub, repo_id: &str, variant: Option<&str>) -> Result<PathBuf, String> {
    let dir = hub.cache_dir(repo_id);
    let branch = hub.default_branch();
    let warm = match variant {
        None => dir.clone(),
        Some(v) => dir.join(v),
    };
    if warm.join(COMPLETE).is_file() && !hub.force {
        tracing::info!(
            hub = hub.slug(),
            repo = repo_id,
            cache = %warm.display(),
            "serving from the cache; hub not contacted"
        );
        return Ok(warm);
    }
    tracing::info!(
        hub = hub.slug(),
        repo = repo_id,
        variant = variant.unwrap_or(""),
        cache = %dir.display(),
        "downloading"
    );
    if let Ok(manifest_path) = hub.get_optional(repo_id, branch, "model.json", &dir) {
        if variant.is_some() {
            return Err(format!(
                "{repo_id} has no variants (its root model.json is a plain package); \
                 serve it as {repo_id}"
            ));
        }
        if hub
            .get_optional(repo_id, branch, VARIANTS_INDEX, &dir)
            .is_ok()
        {
            return Err(format!(
                "{repo_id} has both model.json and {VARIANTS_INDEX} at its root -- \
                 a package is either plain or a variant family, not both"
            ));
        }
        fetch_package_body(hub, repo_id, &dir, "", &dir, &manifest_path)?;
        write_complete(&dir)?;
        return Ok(dir);
    }
    let index_path = hub
        .get(repo_id, branch, VARIANTS_INDEX, &dir)
        .map_err(|err| format!("{err} (model.json is absent too; {})", hub.manifest_hint()))?;
    let index_raw = std::fs::read_to_string(&index_path)
        .map_err(|err| format!("read {}: {err}", index_path.display()))?;
    let variants = parse_variants_index(&index_raw, repo_id)?;
    let Some(v) = variant else {
        return Err(variant_listing(repo_id, &variants, |name| {
            dir.join(name).join(COMPLETE).is_file()
        }));
    };
    if !variants.iter().any(|name| name == v) {
        return Err(format!(
            "unknown variant {v:?} for {repo_id} (available: {})",
            variants.join(", ")
        ));
    }
    let vdir = dir.join(v);
    if hub.force && vdir.exists() {
        std::fs::remove_dir_all(&vdir)
            .map_err(|err| format!("resetting {}: {err}", vdir.display()))?;
    }
    let prefix = format!("{v}/");
    let manifest_path = hub.get(repo_id, branch, &format!("{prefix}model.json"), &dir)?;
    fetch_package_body(hub, repo_id, &dir, &prefix, &vdir, &manifest_path)?;
    write_complete(&vdir)?;
    Ok(vdir)
}

fn fetch_package_body(
    hub: &Hub,
    repo_id: &str,
    dir: &Path,
    prefix: &str,
    ckpt_dir: &Path,
    manifest_path: &Path,
) -> Result<(), String> {
    let branch = hub.default_branch();
    let manifest_raw = std::fs::read_to_string(manifest_path)
        .map_err(|err| format!("read {}: {err}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_raw)
        .map_err(|err| format!("invalid model.json in {repo_id}: {err}"))?;
    if let Some(decl) = checkpoint_decl(&manifest) {
        let decl = decl.map_err(|err| format!("invalid model.json in {repo_id}: {err}"))?;
        if decl.files.is_some() {
            hub.get(
                repo_id,
                branch,
                &format!("{prefix}tokenizer/tokenizer.json"),
                dir,
            )?;
        }
        fetch_checkpoint(hub, ckpt_dir, &decl)?;
        tracing::info!(
            hub = hub.slug(),
            repo = repo_id,
            checkpoint = %decl.repo,
            revision = %decl.revision_on(hub),
            cache = %ckpt_dir.display(),
            "package fetched; weights fetched from the official repo"
        );
        return Ok(());
    }
    let graph = graph_from_manifest(manifest_path, repo_id)?;
    let graph_path = hub.get(repo_id, branch, &format!("{prefix}{graph}"), dir)?;
    let _ = hub.get_optional(repo_id, branch, &format!("{prefix}{graph}.data"), dir);
    let tokenizer_path = hub.get(
        repo_id,
        branch,
        &format!("{prefix}tokenizer/tokenizer.json"),
        dir,
    )?;
    tracing::info!(
        hub = hub.slug(),
        repo = repo_id,
        cache = %dir.display(),
        graph = %graph_path.display(),
        tokenizer = %tokenizer_path.display(),
        "model fetched"
    );
    Ok(())
}

const COMPLETE: &str = ".complete";
fn write_complete(dir: &Path) -> Result<(), String> {
    std::fs::write(dir.join(COMPLETE), b"")
        .map_err(|err| format!("marking {} complete: {err}", dir.display()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointDecl {
    pub repo: String,

    pub revision: Option<String>,

    pub files: Option<Vec<String>>,
}
impl CheckpointDecl {
    pub fn revision_on(&self, hub: &Hub) -> &str {
        self.revision.as_deref().unwrap_or(hub.default_branch())
    }
}

pub fn checkpoint_decl(manifest: &serde_json::Value) -> Option<Result<CheckpointDecl, String>> {
    let decl = manifest.get("checkpoint")?;
    let bad = |msg: String| Some(Err(msg));
    let obj = match decl.as_object() {
        Some(o) => o,
        None => {
            return bad(
                "model.json 'checkpoint' must be an object: {\"repo\" [, \"files\"] \
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
            _ => {
                return bad(
                    "model.json checkpoint.revision must be a non-empty string when present"
                        .to_string(),
                );
            }
        },
    };
    let files: Option<Vec<String>> = match obj.get("files") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let list = match v.as_array() {
                Some(l) => l,
                None => {
                    return bad(
                        "model.json checkpoint.files must be an array of relative paths; \
                         drop the field to fetch the whole official repo"
                            .to_string(),
                    );
                }
            };
            let mut files: Vec<String> = Vec::with_capacity(list.len());
            for item in list {
                let Some(f) = item.as_str() else {
                    return bad("model.json checkpoint.files entries must be strings".to_string());
                };
                files.push(f.to_string());
            }
            if files.is_empty() {
                return bad(
                    "model.json checkpoint.files is empty; drop the field to fetch the \
                     whole official repo"
                        .to_string(),
                );
            }
            if files.iter().any(|f| {
                f.is_empty() || f.starts_with('/') || f.starts_with('.') || f.contains("..")
            }) {
                return bad(
                    "model.json checkpoint.files entries must be plain relative paths".to_string(),
                );
            }
            Some(files)
        }
    };
    Some(Ok(CheckpointDecl {
        repo,
        revision,
        files,
    }))
}

pub fn fetch_checkpoint(hub: &Hub, dir: &Path, decl: &CheckpointDecl) -> Result<(), String> {
    let revision = decl.revision_on(hub);
    match &decl.files {
        Some(files) => {
            for file in files {
                hub.get(&decl.repo, revision, file, dir)
                    .map_err(|err| format!("{err} ({})", hub.checkpoint_hint(&decl.repo)))?;
            }
            let missing = missing_weight_shards(dir)?;
            if !missing.is_empty() {
                return Err(format!(
                    "model.json checkpoint.files for {} is incomplete: the safetensors \
                     index references shards not fetched: {}",
                    decl.repo,
                    missing.join(", ")
                ));
            }
            tracing::info!(
                hub = hub.slug(),
                repo = %decl.repo,
                revision = %revision,
                files = files.len(),
                cache = %dir.display(),
                "checkpoint fetched from the official repo"
            );
        }
        None => {
            let listed = hub
                .list_repo_files(&decl.repo, revision)
                .map_err(|err| format!("{err} ({})", hub.checkpoint_hint(&decl.repo)))?;
            let files: Vec<&String> = listed
                .iter()
                .filter(|f| {
                    !Path::new(f)
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with('.'))
                        .unwrap_or(false)
                })
                .collect();
            if files.is_empty() {
                return Err(format!(
                    "the official repo {} on {} listed no files",
                    decl.repo,
                    hub.display()
                ));
            }
            for file in &files {
                hub.get(&decl.repo, revision, file, dir)
                    .map_err(|err| format!("{err} ({})", hub.checkpoint_hint(&decl.repo)))?;
            }
            validate_snapshot(dir, &decl.repo)?;
            tracing::info!(
                hub = hub.slug(),
                repo = %decl.repo,
                revision = %revision,
                files = files.len(),
                cache = %dir.display(),
                "checkpoint snapshot fetched from the official repo"
            );
        }
    }
    Ok(())
}

const SNAPSHOT_REQUIRED: [&str; 5] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
];

fn validate_snapshot(dir: &Path, repo: &str) -> Result<(), String> {
    let mut missing: Vec<String> = SNAPSHOT_REQUIRED
        .iter()
        .filter(|f| !dir.join(f).is_file())
        .map(|f| f.to_string())
        .collect();
    if !dir.join("model.safetensors").is_file() && !has_weight_index(dir) {
        missing.push("model.safetensors (or a sharded *.safetensors.index.json)".to_string());
    }
    missing.extend(missing_weight_shards(dir)?);
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the official repo {repo} does not look like a servable checkpoint; missing: {}",
        missing.join(", ")
    ))
}

fn has_weight_index(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .file_name()
                    .map(|n| n.to_string_lossy().ends_with(".safetensors.index.json"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn missing_weight_shards(dir: &Path) -> Result<Vec<String>, String> {
    let mut missing = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|err| format!("read {}: {err}", dir.display()))?;
    for entry in entries.filter_map(|e| e.ok()) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".safetensors.index.json") {
            continue;
        }
        let raw = std::fs::read_to_string(entry.path())
            .map_err(|err| format!("read {}: {err}", entry.path().display()))?;
        let index: serde_json::Value =
            serde_json::from_str(&raw).map_err(|err| format!("invalid {name}: {err}"))?;
        let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) else {
            return Err(format!("{name} has no weight_map object"));
        };
        let mut shards: Vec<&str> = map.values().filter_map(|v| v.as_str()).collect();
        shards.sort_unstable();
        shards.dedup();
        for shard in shards {
            if !dir.join(shard).is_file() {
                missing.push(shard.to_string());
            }
        }
    }
    Ok(missing)
}

pub fn verdict_cache_root() -> PathBuf {
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

const HTTP_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

const HTTP_ATTEMPTS: u32 = 3;

const HTTP_BACKOFF_SECS: [u64; 2] = [1, 3];

#[derive(Debug)]
pub enum FetchErr {
    Terminal(String),

    Retryable(String),
}
impl FetchErr {
    pub fn message(self) -> String {
        match self {
            FetchErr::Terminal(msg) | FetchErr::Retryable(msg) => msg,
        }
    }
}

pub fn http_agent_with_read_timeout(timeout: std::time::Duration) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_read(timeout).build()
}
fn http_agent() -> ureq::Agent {
    http_agent_with_read_timeout(HTTP_READ_TIMEOUT)
}

pub fn http_download(url: &str, dest: &Path) -> Result<(), String> {
    let bearer = std::env::var("MODELSCOPE_API_TOKEN").ok();
    http_download_auth(url, dest, bearer.as_deref())
}

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
        let _ = std::fs::remove_file(&part);
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
