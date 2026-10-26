use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;
static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

pub fn install() {
    HANDLE.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .expect("install the prometheus recorder")
    });
}

pub fn render() -> String {
    HANDLE
        .get()
        .map(|h| h.render())
        .unwrap_or_else(|| "# recorder not installed\n".to_string())
}

pub fn record_decision(task: &str, latency_ms: u64) {
    metrics::counter!("verdict_requests_total", "task" => task.to_string()).increment(1);
    metrics::histogram!("verdict_decision_latency_seconds", "task" => task.to_string())
        .record(latency_ms as f64 / 1000.0);
}

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
