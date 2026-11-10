use std::io::IsTerminal as _;
use std::io::Write;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Auto,
    Json,
}

const LINE_INTERVAL: Duration = Duration::from_secs(5);
const JSON_INTERVAL: Duration = Duration::from_secs(1);

struct Reporter {
    out: Mutex<Box<dyn Write + Send>>,
}

static REPORTER: RwLock<Option<Arc<Reporter>>> = RwLock::new(None);

pub fn install_json() {
    set_reporter(Arc::new(Reporter {
        out: Mutex::new(Box::new(std::io::stdout())),
    }));
}

pub fn install_json_with_writer(out: Box<dyn Write + Send>) {
    set_reporter(Arc::new(Reporter {
        out: Mutex::new(out),
    }));
}

pub fn uninstall() {
    *REPORTER.write().unwrap() = None;
}

fn set_reporter(reporter: Arc<Reporter>) {
    *REPORTER.write().unwrap() = Some(reporter);
}

fn reporter() -> Option<Arc<Reporter>> {
    REPORTER.read().unwrap().clone()
}

pub fn mode() -> Mode {
    if reporter().is_some() {
        Mode::Json
    } else {
        Mode::Auto
    }
}

fn emit_json(value: serde_json::Value) {
    let Some(reporter) = reporter() else {
        return;
    };
    let Ok(mut out) = reporter.out.lock() else {
        return;
    };
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

pub fn repo_start(repo: &str, backend: &str, dest: &str, files: usize) {
    if mode() != Mode::Json {
        return;
    }
    emit_json(serde_json::json!({
        "event": "repo_start",
        "repo": repo,
        "backend": backend,
        "dest": dest,
        "files": files,
    }));
}

pub fn repo_done(repo: &str, files: usize, bytes: u64, secs: u64) {
    if mode() != Mode::Json {
        return;
    }
    emit_json(serde_json::json!({
        "event": "repo_done",
        "repo": repo,
        "files": files,
        "bytes": bytes,
        "secs": secs,
    }));
}

pub fn convert_start(input: &str, family: &str) {
    if mode() != Mode::Json {
        return;
    }
    emit_json(serde_json::json!({
        "event": "convert_start",
        "input": input,
        "family": family,
    }));
}

pub fn convert_stage(stage: &str) {
    if mode() != Mode::Json {
        return;
    }
    emit_json(serde_json::json!({
        "event": "convert_stage",
        "stage": stage,
    }));
}

pub fn convert_done(family: &str, files: usize, bytes: u64, check: Option<(&str, bool)>) {
    if mode() != Mode::Json {
        return;
    }
    let mut event = serde_json::json!({
        "event": "convert_done",
        "family": family,
        "files": files,
        "bytes": bytes,
    });
    if let Some((deviation, large)) = check {
        event["deviation"] = serde_json::Value::String(deviation.to_string());
        event["deviation_large"] = serde_json::Value::Bool(large);
    }
    emit_json(event);
}

pub struct FileProgress {
    path: String,
    total: u64,
    current: u64,
    started: Instant,
    last_emit: Instant,
    line_mode: bool,
}

impl FileProgress {
    pub fn new(path: &str, total: u64) -> Self {
        let json = mode() == Mode::Json;
        let line_mode = !json && !std::io::stderr().is_terminal();
        if json {
            emit_json(serde_json::json!({
                "event": "file_start",
                "path": path,
                "total": total,
            }));
        } else if line_mode {
            tracing::info!(file = %path, total = %human(total), "fetching");
        }
        Self {
            path: path.to_owned(),
            total,
            current: 0,
            started: Instant::now(),
            last_emit: Instant::now(),
            line_mode,
        }
    }

    pub fn tick(&mut self, n: u64) {
        self.current += n;
        if mode() == Mode::Auto && !self.line_mode {
            return;
        }
        let now = Instant::now();
        let interval = if mode() == Mode::Json {
            JSON_INTERVAL
        } else {
            LINE_INTERVAL
        };
        if now.duration_since(self.last_emit) < interval {
            return;
        }
        self.last_emit = now;
        let rate = (self.current as f64 / self.started.elapsed().as_secs_f64().max(1e-9)) as u64;
        match mode() {
            Mode::Json => emit_json(serde_json::json!({
                "event": "progress",
                "path": self.path,
                "current": self.current,
                "total": self.total,
                "rate_bps": rate,
            })),
            Mode::Auto => {
                let progress = if self.total > 0 {
                    format!(
                        "{}/{} ({}%)",
                        human(self.current),
                        human(self.total),
                        self.current * 100 / self.total
                    )
                } else {
                    human(self.current)
                };
                tracing::info!(file = %self.path, progress = %progress, rate = %human(rate), "downloading");
            }
        }
    }

    pub fn finish(self) {
        let secs = self.started.elapsed().as_secs();
        if mode() == Mode::Json {
            emit_json(serde_json::json!({
                "event": "file_done",
                "path": self.path,
                "bytes": self.current,
                "secs": secs,
            }));
            return;
        }
        tracing::info!(
            file = %self.path,
            bytes = self.current,
            secs,
            "downloaded"
        );
    }
}

pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
