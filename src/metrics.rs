//! Metrics plane: the `metrics` facade behind the Prometheus exporter.
//! Names use the `verdict_` namespace, never model names.
//!
//! Instruments sit where the tracing spans sit: the pipeline facade
//! (requests + latency per task) and the scheduler loop (queue depth,
//! batch occupancy, batches). Macros are no-ops until a recorder is
//! installed.

use std::sync::OnceLock;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Install the global recorder once (idempotent; safe under tests).
pub fn install() {
    HANDLE.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .expect("install the prometheus recorder")
    });
}

/// Render the exposition text for GET /metrics.
pub fn render() -> String {
    HANDLE
        .get()
        .map(|h| h.render())
        .unwrap_or_else(|| "# recorder not installed\n".to_string())
}

/// Record one served decision (called at response assembly, where the latency
/// span ends).
pub fn record_decision(task: &str, latency_ms: u64) {
    metrics::counter!("verdict_requests_total", "task" => task.to_string()).increment(1);
    metrics::histogram!("verdict_decision_latency_seconds", "task" => task.to_string())
        .record(latency_ms as f64 / 1000.0);
}

/// Scheduler loop observability (called from the batching task). The
/// absolute-set gauges carry a `replica` label so N data-parallel batchers
/// do not overwrite each other's values; additive instruments (counter,
/// histogram, in_flight) aggregate across replicas without one.
pub fn record_queue_depth(depth: usize, replica: usize) {
    metrics::gauge!("verdict_queue_depth", "replica" => replica.to_string()).set(depth as f64);
}

pub fn record_batch(b: usize, replica: usize) {
    metrics::gauge!("verdict_batch_rows", "replica" => replica.to_string()).set(b as f64);
    metrics::counter!("verdict_batches_total").increment(1);
}

pub fn record_in_flight(delta: i64) {
    if delta >= 0 {
        metrics::gauge!("verdict_in_flight").increment(delta as f64);
    } else {
        metrics::gauge!("verdict_in_flight").decrement((-delta) as f64);
    }
}
