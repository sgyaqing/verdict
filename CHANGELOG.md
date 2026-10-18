# Changelog

## 0.2.0 — 2026-10-17

- **Two new models**: decider-2b (Qwen3.5-2B, 32768-token context, bf16) and von (ModernBERT-large, 8192-token context), on both Hugging Face and ModelScope.
- **PyTorch runtime**: the Rust engine drives a pinned PyTorch stack running the official checkpoint — for model families with no ONNX export path.
- **Model package upgrade**: a package can declare its upstream checkpoint instead of carrying weights (decider-2b fetches from the official `Mapika/decider-2b` repo on first serve); downloader rewritten in-house (resumable, atomic); single `~/.cache/verdict` cache; new `--force-download` flag and `VERDICT_CACHE` variable.

Images on Docker Hub: `sgyaqing/verdict:0.2.0-cpu` and `sgyaqing/verdict:0.2.0-cuda` (linux/amd64).

## 0.1.0 — 2026-10-11

A high-performance serving engine for System One decision models.

First public release. Verdict serves decision models (Laya and Laya Multilingual today, Qwen-based System One models next) the way vLLM serves LLMs: one binary, one command, an HTTP API -- and answers verified against the official model, not assumed.

## What's in

- **Real inference** on ONNX Runtime, on CPU and CUDA: answers verified against the official model -- bit-for-bit identical on CPU, within 1e-4 on CUDA.
- **Native protocol**: `/v1/choice`, `/v1/score`, `/v1/noul` -- the three System One task primitives (the same vocabulary TypeSafe's Jev exposes), with validation, a unified error shape, usage accounting, and a 1 MiB body cap.
- **`/v1/systemone` and `/v1/decisions`**: both standard decision protocols. systemone is the category's native API (defined by Jev; a Jev client works by changing its base URL); decisions is OpenAI's cross-ecosystem protocol (for LiteLLM's unified endpoint and OpenAI-ecosystem clients). The same question returns the same numbers through both.
- **Two Verdict-ready models**: `laya` (English) and `laya-multilingual` (100+ languages), hosted on both Hugging Face and ModelScope.
- **Dynamic batching**: bounded queue, windowed batch assembly, mixed-task batches, ragged-K tensors, token-budget admission, assembly/execution overlap.
- **OpenAI-compatible layer**: `/v1/chat/completions` and `/v1/models`; decision fields ride as top-level extensions via `extra_body`.
- **Operations**: `--policy priority` scheduling, `--data-parallel N` engine replicas (the high-throughput shape on many-core CPU hosts), `/metrics` Prometheus endpoint.

## Performance

RTX 3080, concurrency 32:

| Setup | Throughput | P50 |
|---|---:|---:|
| laya · FastAPI + torch | 37.8 rps | 844 ms |
| laya · Verdict | 408 rps | 78 ms |
| laya-multilingual · FastAPI + torch | 46.0 rps | 691 ms |
| laya-multilingual · Verdict | 1059 rps | 30 ms |

On CPU (96-core x86), Verdict with `--data-parallel 4` serves 3.8x-4.2x
FastAPI + torch (3.8x for laya-multilingual, 4.2x for laya). Full tables in the README.

## Requirements

Images on Docker Hub: `sgyaqing/verdict:0.1.0-cpu` and `sgyaqing/verdict:0.1.0-cuda` (linux/amd64). The CUDA image needs an NVIDIA GPU with compute capability 7.5+ and driver 580+.
