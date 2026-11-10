//! CUDA-graph coverage reported by the torch executor: what the boot envelope holds, and how many
//! rows missed it by reason. The executor sends cumulative counters with every reply; this keeps the
//! latest snapshot per engine replica so `/graph` can answer without asking the executor anything.

use std::sync::Mutex;

pub const REASONS: [&str; 7] = [
    "len",
    "rows",
    "degraded",
    "fork_prefix",
    "fork_width",
    "fork_rows",
    "fork_degraded",
];

#[derive(Clone, Default)]
struct EngineStats {
    cells: [[u64; 2]; 3],
    envelope: [u64; 4],
    gpu_memory_utilization: f64,
    budget_bytes: u64,
    reserved_bytes: u64,
    transient_bytes: u64,
    served_rows: u64,
    uncovered: [u64; REASONS.len()],
    fork_not_applicable: u64,
}

static ENGINES: Mutex<Vec<EngineStats>> = Mutex::new(Vec::new());

pub fn register() -> usize {
    let mut engines = ENGINES.lock().expect("graph stats mutex poisoned");
    engines.push(EngineStats::default());
    engines.len() - 1
}

pub fn set_ready(slot: usize, ready: &serde_json::Value) {
    let mut engines = ENGINES.lock().expect("graph stats mutex poisoned");
    let Some(entry) = engines.get_mut(slot) else {
        return;
    };
    let cell = |kind: &str| -> [u64; 2] {
        let value = ready
            .get("graph_cells")
            .and_then(|c| c.get(kind))
            .and_then(|v| v.as_array());
        [0, 1].map(|i| {
            value
                .and_then(|v| v.get(i))
                .and_then(|n| n.as_u64())
                .unwrap_or(0)
        })
    };
    entry.cells = [cell("text"), cell("media"), cell("fork")];
    let envelope = |key: &str| {
        ready
            .get("graph_envelope")
            .and_then(|e| e.get(key))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    entry.envelope = [
        envelope("max_len"),
        envelope("max_rows"),
        envelope("fork_max_prefix"),
        envelope("fork_max_width"),
    ];
    entry.gpu_memory_utilization = ready
        .get("graph_envelope")
        .and_then(|e| e.get("gpu_memory_utilization"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    entry.budget_bytes = envelope("budget_bytes");
    entry.reserved_bytes = envelope("reserved_bytes");
    entry.transient_bytes = envelope("transient_bytes");
}

pub fn record(slot: usize, reply: &serde_json::Value) {
    let mut engines = ENGINES.lock().expect("graph stats mutex poisoned");
    let Some(entry) = engines.get_mut(slot) else {
        return;
    };
    let Some(served) = reply.get("served_rows").and_then(|v| v.as_u64()) else {
        return;
    };
    entry.served_rows = served;
    let Some(map) = reply.get("uncovered").and_then(|v| v.as_object()) else {
        return;
    };
    for (i, reason) in REASONS.iter().enumerate() {
        entry.uncovered[i] = map.get(*reason).and_then(|v| v.as_u64()).unwrap_or(0);
    }
    entry.fork_not_applicable = map.get("fork_na").and_then(|v| v.as_u64()).unwrap_or(0);
}

pub fn render() -> serde_json::Value {
    let engines = ENGINES.lock().expect("graph stats mutex poisoned");
    let mut cells = [[0u64; 2]; 3];
    let mut envelope = [0u64; 4];
    let mut gpu_memory_utilization = 0.0f64;
    let mut budget_bytes = 0u64;
    let mut reserved_bytes = 0u64;
    let mut transient_bytes = 0u64;
    let mut served = 0u64;
    let mut uncovered = [0u64; REASONS.len()];
    let mut not_applicable = 0u64;
    for entry in engines.iter() {
        for (total, value) in cells.iter_mut().zip(entry.cells.iter()) {
            total[0] += value[0];
            total[1] += value[1];
        }
        for (max, value) in envelope.iter_mut().zip(entry.envelope.iter()) {
            *max = (*max).max(*value);
        }
        gpu_memory_utilization = gpu_memory_utilization.max(entry.gpu_memory_utilization);
        budget_bytes = budget_bytes.max(entry.budget_bytes);
        reserved_bytes += entry.reserved_bytes;
        transient_bytes = transient_bytes.max(entry.transient_bytes);
        served += entry.served_rows;
        for (total, value) in uncovered.iter_mut().zip(entry.uncovered.iter()) {
            *total += *value;
        }
        not_applicable += entry.fork_not_applicable;
    }
    let ratio = |n: u64| {
        if served == 0 {
            0.0
        } else {
            n as f64 / served as f64
        }
    };
    serde_json::json!({
        "envelope": {
            "max_len": envelope[0],
            "max_rows": envelope[1],
            "fork_max_prefix": envelope[2],
            "fork_max_width": envelope[3],
            "gpu_memory_utilization": gpu_memory_utilization,
            "budget_bytes": budget_bytes,
            "reserved_bytes": reserved_bytes,
            "transient_bytes": transient_bytes,
        },
        "cells": {
            "text": {"captured": cells[0][0], "planned": cells[0][1]},
            "media": {"captured": cells[1][0], "planned": cells[1][1]},
            "fork": {"captured": cells[2][0], "planned": cells[2][1]},
        },
        "served_rows": served,
        "eager": {
            "total": ratio(uncovered[0] + uncovered[1] + uncovered[2]),
            "len": ratio(uncovered[0]),
            "rows": ratio(uncovered[1]),
            "degraded": ratio(uncovered[2]),
        },
        "fork": {
            "total": ratio(uncovered[3] + uncovered[4] + uncovered[5] + uncovered[6]),
            "prefix": ratio(uncovered[3]),
            "width": ratio(uncovered[4]),
            "rows": ratio(uncovered[5]),
            "degraded": ratio(uncovered[6]),
            "not_applicable": ratio(not_applicable),
        },
        "replicas": engines.len(),
    })
}
