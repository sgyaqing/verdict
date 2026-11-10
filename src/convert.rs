use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::BufRead as _;
use std::path::{Path, PathBuf};

const GRAPH: &str = "model.onnx";

const COMPARE: &str = include_str!("../convert/compare.py");
const MARKER_HEAD_CONVERTER: &str = include_str!("../convert/marker_head/marker_head_converter.py");
const MARKER_HEAD_PROBE: &str = include_str!("../convert/marker_head/marker_head_probe.py");
const INDEPENDENT_OPTIONS_CONVERTER: &str =
    include_str!("../convert/independent_options/independent_options_converter.py");
const INDEPENDENT_OPTIONS_PROBE: &str =
    include_str!("../convert/independent_options/independent_options_probe.py");

const PINNED: &[(&str, &str)] = &[
    ("torch", "2.14.1"),
    ("transformers", "5.19.0"),
    ("onnxruntime", "1.30.0"),
    ("onnxscript", "0.7.2"),
];
const PINNED_VON: &[(&str, &str)] = &[("von_sdk", "1.3.7")];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    MarkerHead,
    IndependentOptions,
}
impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::MarkerHead => "marker_head",
            Family::IndependentOptions => "independent_options",
        }
    }
}

pub fn convert_command(input: &Path, output: &Path) -> Result<(), String> {
    let input = input
        .canonicalize()
        .map_err(|err| format!("{}: {err}", input.display()))?;
    if !input.is_dir() {
        return Err(format!("{} is not a directory", input.display()));
    }
    if output.exists() && !output.is_dir() {
        return Err(format!(
            "{} exists and is not a directory",
            output.display()
        ));
    }
    if output.is_dir()
        && std::fs::read_dir(output)
            .map_err(|err| format!("read {}: {err}", output.display()))?
            .next()
            .is_some()
    {
        return Err(format!(
            "{} is not empty; clear it to retry",
            output.display()
        ));
    }
    let started = std::time::Instant::now();
    let family = detect(&input)?;
    tracing::info!(family = family.name(), "detected the model");
    crate::progress::convert_start(&input.display().to_string(), family.name());
    crate::progress::convert_stage("probe");
    let python = probe(family)?;
    std::fs::create_dir_all(output).map_err(|err| format!("create {}: {err}", output.display()))?;
    let check = match family {
        Family::MarkerHead => convert_marker_head(&input, output, &python),
        Family::IndependentOptions => convert_independent_options(&input, output, &python),
    }?;
    crate::progress::convert_stage("checksums");
    let sums = write_sums(output)?;
    let mut files = 0usize;
    let mut bytes = 0u64;
    for (_, size) in sums {
        files += 1;
        bytes += size;
    }
    tracing::info!(
        elapsed = %elapsed_label(started.elapsed()),
        output = %output.display(),
        "conversion complete"
    );
    crate::progress::convert_done(
        family.name(),
        files,
        bytes,
        check.as_ref().map(|c| (c.deviation.as_str(), c.large)),
    );
    Ok(())
}

pub fn detect(input: &Path) -> Result<Family, String> {
    let weights = input.join("model.safetensors");
    if !weights.is_file() {
        return Err(format!(
            "{} has no model.safetensors; merge sharded or .bin weights into one file first",
            input.display()
        ));
    }
    let keys = safetensors_keys(&weights)?;
    match signature(&keys) {
        Some(family) => Ok(family),
        None => Err(format!(
            "{}: unknown weight signature; known families: marker_head \
             (encoder./scorer./act_head./type_emb.), independent_options \
             (embeddings./layers./final_norm.)",
            input.display()
        )),
    }
}

fn signature(keys: &[String]) -> Option<Family> {
    let marker_head = carries(keys, &["encoder.", "scorer.", "act_head.", "type_emb."]);
    let independent = carries(keys, &["embeddings.", "layers.", "final_norm."]);
    match (marker_head, independent) {
        (true, false) => Some(Family::MarkerHead),
        (false, true) => Some(Family::IndependentOptions),
        _ => None,
    }
}

fn carries(keys: &[String], prefixes: &[&str]) -> bool {
    prefixes
        .iter()
        .all(|prefix| keys.iter().any(|key| key.starts_with(prefix)))
}

fn safetensors_keys(path: &Path) -> Result<Vec<String>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .map_err(|err| format!("read {} header: {err}", path.display()))?;
    let n = u64::from_le_bytes(len);
    if n == 0 || n > 64 * 1024 * 1024 {
        return Err(format!(
            "{} has a malformed safetensors header length ({n})",
            path.display()
        ));
    }
    let mut buf = vec![0u8; n as usize];
    file.read_exact(&mut buf)
        .map_err(|err| format!("read {} header: {err}", path.display()))?;
    let header: Value = serde_json::from_slice(&buf)
        .map_err(|err| format!("parse {} header: {err}", path.display()))?;
    let object = header
        .as_object()
        .ok_or_else(|| format!("{} header is not an object", path.display()))?;
    Ok(object
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .cloned()
        .collect())
}

fn read_json(path: &Path) -> Result<Value, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    serde_json::from_str(&raw).map_err(|err| format!("parse {}: {err}", path.display()))
}

fn root_json(input: &Path, required: &[&str]) -> Result<(PathBuf, Value), String> {
    let mut hits: Vec<(PathBuf, Value)> = Vec::new();
    let entries =
        std::fs::read_dir(input).map_err(|err| format!("read {}: {err}", input.display()))?;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        if meta.len() > 1_000_000 {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        if let Some(object) = value.as_object()
            && required.iter().all(|key| object.contains_key(*key))
        {
            hits.push((path, value));
        }
    }
    match hits.len() {
        1 => Ok(hits.pop().unwrap()),
        0 => Err(format!(
            "no JSON in {} carries {}; the keys identify the config, not the file name",
            input.display(),
            required.join(" + ")
        )),
        _ => {
            let names: Vec<String> = hits
                .iter()
                .filter_map(|(p, _)| p.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .collect();
            Err(format!(
                "several JSON files in {} carry {}: {}; keep one",
                input.display(),
                required.join(" + "),
                names.join(", ")
            ))
        }
    }
}

fn tokenizer_dir(input: &Path) -> Result<PathBuf, String> {
    let have = |dir: &Path| {
        dir.join("tokenizer.json").is_file() && dir.join("tokenizer_config.json").is_file()
    };
    let nested = input.join("tokenizer");
    if have(&nested) {
        return Ok(nested);
    }
    if have(input) {
        return Ok(input.to_path_buf());
    }
    Err(format!(
        "{} carries no tokenizer (tokenizer.json + tokenizer_config.json at the root or \
         under tokenizer/)",
        input.display()
    ))
}

fn field<'a>(value: &'a Value, source: &Path, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("{} is missing {key:?}", source.display()))
}

fn probe(family: Family) -> Result<PathBuf, String> {
    let candidates: Vec<PathBuf> = match std::env::var("VERDICT_CONVERT_PYTHON") {
        Ok(path) => vec![PathBuf::from(path)],
        Err(_) => vec![PathBuf::from("python3.12")],
    };
    let mut pins: Vec<(&str, &str)> = PINNED.to_vec();
    if family == Family::IndependentOptions {
        pins.extend_from_slice(PINNED_VON);
    }
    let mut probe_src = String::from("import importlib.metadata as md\nproblems=[]\n");
    for (pkg, want) in &pins {
        probe_src.push_str(&format!(
            "try:\n    got = md.version('{pkg}')\nexcept Exception:\n    \
             problems.append('{pkg}: missing (want {want})')\nelse:\n    \
             problems.append('{pkg}: ' + got + ' (want {want})') if \
             got.split('+')[0] != '{want}' else None\n"
        ));
    }
    probe_src.push_str("print('OK' if not problems else '; '.join(problems))\n");
    for candidate in &candidates {
        let Ok(out) = std::process::Command::new(candidate)
            .arg("-c")
            .arg(&probe_src)
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if stdout == "OK" {
            return Ok(candidate.clone());
        }
        return Err(format!(
            "{} does not carry the pinned export stack:\n  {stdout}\n\
             set VERDICT_CONVERT_PYTHON to a pinned environment (the verdict-convert image \
             ships one)",
            candidate.display()
        ));
    }
    Err(format!(
        "no python 3.12 with the pinned export stack found (tried {}); set \
             VERDICT_CONVERT_PYTHON or use the verdict-convert image",
        candidates
            .iter()
            .map(|c| c.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

const DOT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

const TAIL_LINES: usize = 20;

fn clear_status() {
    eprint!("\r{}\r", " ".repeat(14));
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

struct ExportCheck {
    deviation: String,
    large: bool,
}

impl ExportCheck {
    fn emit(&self) {
        if self.large {
            tracing::warn!(
                "ONNX vs torch relative deviation {} is too large",
                self.deviation
            );
        } else {
            tracing::info!("ONNX vs torch relative deviation {}", self.deviation);
        }
    }
}

fn parse_check(line: &str) -> Option<ExportCheck> {
    let rest = line.strip_prefix("[verdict] ")?;
    let mut deviation = None;
    let mut large = None;
    for part in rest.split_whitespace() {
        if let Some(value) = part.strip_prefix("deviation=") {
            deviation = Some(value.to_string());
        } else if let Some(value) = part.strip_prefix("large=") {
            large = Some(value == "yes");
        }
    }
    Some(ExportCheck {
        deviation: deviation?,
        large: large?,
    })
}

fn run_export(mut cmd: std::process::Command, title: &str) -> Result<Option<ExportCheck>, String> {
    cmd.env("PYTHONHASHSEED", "0");
    cmd.env("PYTHONWARNINGS", "ignore");
    cmd.env("TORCH_LOGS", "-all");
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let started = std::time::Instant::now();
    let mut child = cmd.spawn().map_err(|err| format!("spawn {title}: {err}"))?;
    let (tx, rx) = std::sync::mpsc::channel::<(bool, String)>();
    let streams: [(Option<Box<dyn std::io::Read + Send>>, bool); 2] = [
        (
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            false,
        ),
        (
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            true,
        ),
    ];
    for (stream, is_stderr) in streams {
        let Some(stream) = stream else { continue };
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stream).lines() {
                let Ok(line) = line else { break };
                if tx.send((is_stderr, line)).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);
    let mut err_tail: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut out_tail: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut check: Option<ExportCheck> = None;
    let animate = std::io::IsTerminal::is_terminal(&std::io::stderr());
    if animate {
        eprint!("converting");
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
    let mut dots = 0usize;
    loop {
        match rx.recv_timeout(DOT_INTERVAL) {
            Ok((is_stderr, line)) => {
                if animate {
                    clear_status();
                }
                if is_stderr {
                    if err_tail.len() == TAIL_LINES {
                        err_tail.pop_front();
                    }
                    err_tail.push_back(line.clone());
                    eprintln!("{line}");
                } else {
                    if out_tail.len() == TAIL_LINES {
                        out_tail.pop_front();
                    }
                    out_tail.push_back(line.clone());
                    if let Some(parsed) = parse_check(&line) {
                        check = Some(parsed);
                    } else if !line.starts_with("[torch.onnx]") {
                        tracing::info!("{line}");
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !animate {
                    continue;
                }
                dots = dots % 3 + 1;
                eprint!("\rconverting {}{}", ".".repeat(dots), " ".repeat(3 - dots));
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if animate {
        clear_status();
    }
    let status = child.wait().map_err(|err| format!("wait {title}: {err}"))?;
    if !status.success() {
        let mut message = format!(
            "{title} failed (exit {:?}) after {}s",
            status.code(),
            started.elapsed().as_secs()
        );
        if !err_tail.is_empty() {
            message.push_str("\n--- stderr ---\n");
            message.push_str(&err_tail.into_iter().collect::<Vec<_>>().join("\n"));
        }
        if !out_tail.is_empty() {
            message.push_str("\n--- stdout ---\n");
            message.push_str(&out_tail.into_iter().collect::<Vec<_>>().join("\n"));
        }
        return Err(message);
    }
    if let Some(found) = &check {
        found.emit();
    }
    Ok(check)
}

struct Staging {
    path: PathBuf,
}
impl Staging {
    fn new(family: Family) -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!(
            "verdict-convert-{}-{}",
            family.name(),
            std::process::id()
        ));
        if path.exists() {
            std::fs::remove_dir_all(&path)
                .map_err(|err| format!("clear {}: {err}", path.display()))?;
        }
        std::fs::create_dir_all(&path)
            .map_err(|err| format!("create {}: {err}", path.display()))?;
        Ok(Self { path })
    }
    fn file(&self, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
        let path = self.path.join(name);
        write_file(&path, bytes)?;
        Ok(path)
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("create {}: {err}", parent.display()))?;
    }
    std::fs::write(path, bytes).map_err(|err| format!("write {}: {err}", path.display()))
}

fn token_of(tok: &Value, source: &Path, key: &str) -> Result<Value, String> {
    let value = field(tok, source, key)?;
    if value.as_str().is_none() {
        return Err(format!("{key:?} in {} must be a string", source.display()));
    }
    Ok(value.clone())
}

fn sorted_object(value: &Value, source: &Path, key: &str) -> Result<Value, String> {
    let object = field(value, source, key)?
        .as_object()
        .ok_or_else(|| format!("{key:?} in {} must be an object", source.display()))?;
    let mut sorted: std::collections::BTreeMap<String, Value> =
        object.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    sorted.retain(|_, v| !v.is_null());
    let mut out = serde_json::Map::new();
    for (k, v) in sorted {
        out.insert(k, v);
    }
    Ok(Value::Object(out))
}

fn number_field(value: &Value, source: &Path, key: &str) -> Result<Value, String> {
    let v = field(value, source, key)?;
    if !v.is_number() {
        return Err(format!("{key:?} in {} must be a number", source.display()));
    }
    Ok(v.clone())
}

pub fn emit_laya_manifest(input: &Path, graph: &str) -> Result<Vec<u8>, String> {
    let (cfg_path, cfg) = root_json(
        input,
        &[
            "max_len",
            "head_max_len",
            "temperature",
            "temperature_by_options",
        ],
    )?;
    let tok_path = tokenizer_dir(input)?.join("tokenizer_config.json");
    let tok = read_json(&tok_path)?;
    let tokens = (
        token_of(&tok, &tok_path, "cls_token")?,
        token_of(&tok, &tok_path, "sep_token")?,
        token_of(&tok, &tok_path, "mask_token")?,
        token_of(&tok, &tok_path, "pad_token")?,
    );
    let manifest = json!({
        "family": "marker_head",
        "runtime": "ort",
        "graph": graph,
        "head_max_len": number_field(&cfg, &cfg_path, "head_max_len")?,
        "max_len": number_field(&cfg, &cfg_path, "max_len")?,
        "cls_token": tokens.0,
        "sep_token": tokens.1,
        "mask_token": tokens.2,
        "pad_token": tokens.3,
        "temperature": field(&cfg, &cfg_path, "temperature")?.clone(),
        "temperature_by_options": sorted_object(&cfg, &cfg_path, "temperature_by_options")?,
    });
    let text = serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?;
    Ok(text.into_bytes())
}

fn inline_object(entries: &[(&str, &Value)]) -> String {
    let inner: Vec<String> = entries
        .iter()
        .map(|(k, v)| format!("{:?}: {}", k, serde_json::to_string(v).unwrap()))
        .collect();
    format!("{{{}}}", inner.join(", "))
}

pub fn emit_von_manifest(input: &Path) -> Result<Vec<u8>, String> {
    let (cfg_path, cfg) = root_json(input, &["max_position_embeddings"])?;
    let (cal_path, cal) = root_json(input, &["calibration_map", "noul_zero_shot_prior"])?;
    let tok_path = tokenizer_dir(input)?.join("tokenizer_config.json");
    let tok = read_json(&tok_path)?;
    let prior = field(&cal, &cal_path, "noul_zero_shot_prior")?
        .as_object()
        .ok_or_else(|| {
            format!(
                "noul_zero_shot_prior in {} must be an object",
                cal_path.display()
            )
        })?
        .clone();
    let prior_a = number_field(&Value::Object(prior.clone()), &cal_path, "a").map_err(|_| {
        format!(
            "noul_zero_shot_prior in {} is missing \"a\"",
            cal_path.display()
        )
    })?;
    let prior_b = number_field(&Value::Object(prior), &cal_path, "b").map_err(|_| {
        format!(
            "noul_zero_shot_prior in {} is missing \"b\"",
            cal_path.display()
        )
    })?;
    let sliding = match cfg.get("sliding_window") {
        Some(value) if value.is_number() => value.clone(),
        _ => json!(64),
    };
    let manifest = json!({
        "family": "independent_options",
        "runtime": "ort",
        "graph": GRAPH,
        "max_len": number_field(&cfg, &cfg_path, "max_position_embeddings")?,
        "cls_token": token_of(&tok, &tok_path, "cls_token")?,
        "sep_token": token_of(&tok, &tok_path, "sep_token")?,
        "mask_token": token_of(&tok, &tok_path, "mask_token")?,
        "pad_token": token_of(&tok, &tok_path, "pad_token")?,
        "sliding_window": sliding,
        "calibration_formula": field(&cal, &cal_path, "calibration_map")?.clone(),
        "@@noul_band@@": 0,
        "@@noul_prior@@": 0,
    });
    let mut text = serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?;
    text = text.replace(
        "\"@@noul_band@@\": 0",
        &format!(
            "\"noul_band\": {}",
            inline_object(&[("edge", &json!(0.8)), ("slope", &json!(0.1))])
        ),
    );
    text = text.replace(
        "\"@@noul_prior@@\": 0",
        &format!(
            "\"noul_prior\": {}",
            inline_object(&[("a", &prior_a), ("b", &prior_b)])
        ),
    );
    text.push('\n');
    Ok(text.into_bytes())
}

fn copy_tokenizer(source: &Path, output: &Path) -> Result<(), String> {
    let dest = output.join("tokenizer");
    std::fs::create_dir_all(&dest).map_err(|err| format!("create {}: {err}", dest.display()))?;
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(source.join(name), dest.join(name)).map_err(|err| {
            format!(
                "copy {} -> {}: {err}",
                source.join(name).display(),
                dest.join(name).display()
            )
        })?;
    }
    Ok(())
}

fn write_sums(output: &Path) -> Result<Vec<(String, u64)>, String> {
    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    collect_files(output, output, &mut entries)?;
    entries.sort_by_key(|(rel, _)| rel.to_lowercase());
    let mut text = String::new();
    let mut sizes = Vec::new();
    for (rel, path) in &entries {
        let bytes = std::fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        text.push_str(&format!("{:x}  {rel}\n", hasher.finalize()));
        sizes.push((rel.clone(), bytes.len() as u64));
    }
    write_file(&output.join("SHA256SUMS"), text.as_bytes())?;
    Ok(sizes)
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir)
        .map_err(|err| format!("read {}: {err}", dir.display()))?
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "SHA256SUMS" {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| name.clone());
            out.push((rel, path));
        }
    }
    Ok(())
}

fn convert_marker_head(
    input: &Path,
    output: &Path,
    python: &Path,
) -> Result<Option<ExportCheck>, String> {
    let staging = Staging::new(Family::MarkerHead)?;
    staging.file("compare.py", COMPARE.as_bytes())?;
    staging.file("marker_head_probe.py", MARKER_HEAD_PROBE.as_bytes())?;
    let script = staging.file("marker_head_converter.py", MARKER_HEAD_CONVERTER.as_bytes())?;
    crate::progress::convert_stage("export");
    let cfg_path = root_json(
        input,
        &[
            "max_len",
            "head_max_len",
            "temperature",
            "temperature_by_options",
        ],
    )?
    .0;
    let mut cmd = std::process::Command::new(python);
    cmd.arg(&script)
        .arg(input)
        .arg(output)
        .arg(&cfg_path)
        .arg(GRAPH);
    let check = run_export(cmd, "marker_head convert")?;
    if !output.join(GRAPH).is_file() {
        return Err(format!(
            "the export produced no {GRAPH} in {}",
            output.display()
        ));
    }
    let _ = std::fs::remove_file(output.join("laya_config.json"));
    let _ = std::fs::remove_dir_all(output.join("tokenizer"));
    copy_tokenizer(&tokenizer_dir(input)?, output)?;
    let manifest = emit_laya_manifest(input, GRAPH)?;
    write_file(&output.join("model.json"), &manifest)?;
    Ok(check)
}

fn convert_independent_options(
    input: &Path,
    output: &Path,
    python: &Path,
) -> Result<Option<ExportCheck>, String> {
    let staging = Staging::new(Family::IndependentOptions)?;
    staging.file("compare.py", COMPARE.as_bytes())?;
    staging.file(
        "independent_options_probe.py",
        INDEPENDENT_OPTIONS_PROBE.as_bytes(),
    )?;
    let script = staging.file(
        "independent_options_converter.py",
        INDEPENDENT_OPTIONS_CONVERTER.as_bytes(),
    )?;
    crate::progress::convert_stage("export");
    let mut cmd = std::process::Command::new(python);
    cmd.arg(&script).arg(input).arg(output).arg(GRAPH);
    let check = run_export(cmd, "independent_options convert")?;
    if !output.join(GRAPH).is_file() {
        return Err(format!(
            "the export produced no {GRAPH} in {}",
            output.display()
        ));
    }
    if !output.join(format!("{GRAPH}.data")).is_file() {
        return Err(format!(
            "the export produced no {GRAPH}.data in {}",
            output.display()
        ));
    }
    copy_tokenizer(&tokenizer_dir(input)?, output)?;
    let manifest = emit_von_manifest(input)?;
    write_file(&output.join("model.json"), &manifest)?;
    Ok(check)
}

fn elapsed_label(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}
