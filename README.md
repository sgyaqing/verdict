<p align="center">
  <img src="assets/logo.svg" alt="Verdict" width="300">
</p>

Verdict is a high-performance serving engine for System One decision models.

Up to **66x** the throughput of the model authors' own servers, with the same answers as the official SDKs (see **Performance** and **Correctness**).

<details open>
<summary><strong>OneJev1.6-9b</strong></summary>

<p align="center">
  <img src="assets/bench-onejev-9b-text-tp.svg" alt="OneJev1.6-9b text throughput vs concurrency, RTX 4090" width="49%">
  <img src="assets/bench-onejev-9b-text-lat.svg" alt="OneJev1.6-9b text P99 latency vs concurrency, RTX 4090" width="49%">
  <br><b>text</b>
</p>

<p align="center">
  <img src="assets/bench-onejev-9b-image-tp.svg" alt="OneJev1.6-9b image throughput vs concurrency, RTX 4090" width="49%">
  <img src="assets/bench-onejev-9b-image-lat.svg" alt="OneJev1.6-9b image P99 latency vs concurrency, RTX 4090" width="49%">
  <br><b>image</b>
</p>

<p align="center">
  <img src="assets/bench-onejev-9b-video-tp.svg" alt="OneJev1.6-9b video throughput vs concurrency, RTX 4090" width="49%">
  <img src="assets/bench-onejev-9b-video-lat.svg" alt="OneJev1.6-9b video P99 latency vs concurrency, RTX 4090" width="49%">
  <br><b>video</b>
</p>

</details>

<details open>
<summary><strong>decider-2b</strong></summary>

<p align="center">
  <img src="assets/bench-decider-gpu-tp.svg" alt="decider-2b throughput vs concurrency, RTX 4090" width="49%">
  <img src="assets/bench-decider-gpu-lat.svg" alt="decider-2b P99 latency vs concurrency, RTX 4090" width="49%">
  <br><b>GPU only</b>
</p>

</details>

<details>
<summary><strong>von</strong></summary>

<p align="center">
  <img src="assets/bench-von-gpu-tp.svg" alt="von throughput vs concurrency, RTX 4090" width="49%">
  <img src="assets/bench-von-gpu-lat.svg" alt="von P99 latency vs concurrency, RTX 4090" width="49%">
  <br><b>GPU</b>
</p>

<p align="center">
  <img src="assets/bench-von-cpu-tp.svg" alt="von throughput vs concurrency, CPU" width="49%">
  <img src="assets/bench-von-cpu-lat.svg" alt="von P99 latency vs concurrency, CPU" width="49%">
  <br><b>CPU</b>
</p>

</details>

<details>
<summary><strong>laya-multilingual</strong></summary>

<p align="center">
  <img src="assets/bench-laya-ml-gpu-tp.svg" alt="laya-multilingual throughput vs concurrency, RTX 4090" width="49%">
  <img src="assets/bench-laya-ml-gpu-lat.svg" alt="laya-multilingual P99 latency vs concurrency, RTX 4090" width="49%">
  <br><b>GPU</b>
</p>

<p align="center">
  <img src="assets/bench-laya-ml-cpu-tp.svg" alt="laya-multilingual throughput vs concurrency, CPU" width="49%">
  <img src="assets/bench-laya-ml-cpu-lat.svg" alt="laya-multilingual P99 latency vs concurrency, CPU" width="49%">
  <br><b>CPU</b>
</p>

The official server refused requests at concurrency 64, so its line ends at concurrency 8.

</details>

<details>
<summary><strong>laya</strong></summary>

<p align="center">
  <img src="assets/bench-laya-gpu-tp.svg" alt="laya throughput vs concurrency, RTX 4090" width="49%">
  <img src="assets/bench-laya-gpu-lat.svg" alt="laya P99 latency vs concurrency, RTX 4090" width="49%">
  <br><b>GPU</b>
</p>

<p align="center">
  <img src="assets/bench-laya-cpu-tp.svg" alt="laya throughput vs concurrency, CPU" width="49%">
  <img src="assets/bench-laya-cpu-lat.svg" alt="laya P99 latency vs concurrency, CPU" width="49%">
  <br><b>CPU</b>
</p>

The official server refused requests at concurrency 64, so its line ends at concurrency 8.

</details>

## Quick start

```bash
# the -v mount caches the model: the first run downloads it, later runs reuse the cache
docker run --rm --gpus all -p 8080:8080 \
    -v ~/.cache/verdict:/root/.cache/verdict \
    sgyaqing/verdict:latest-cuda serve verdict-models/OneJev1.6:9b --host 0.0.0.0

# ModelScope:
docker run --rm --gpus all -p 8080:8080 \
    -e VERDICT_USE_MODELSCOPE=1 \
    -v ~/.cache/verdict:/root/.cache/verdict \
    sgyaqing/verdict:latest-cuda serve sgyaqing/OneJev1.6:9b --host 0.0.0.0
```

Once started, you can test it with:

```bash
curl -s localhost:8080/v1/choice -H 'content-type: application/json' -d '{
  "question": "What should I do for dinner tonight?",
  "state": "Fridge is empty, it'\''s late, and I'\''m tired.",
  "options": ["order delivery", "cook pasta", "skip dinner"]}'
```

## Supported Models

| Verdict-ready&nbsp;model | Languages | One&nbsp;command |
|---|---|---|
| [`verdict-models/OneJev1.6`](https://huggingface.co/verdict-models/OneJev1.6) (variant family: 0.8b / 4b / 9b / 27b) | English | `verdict serve verdict-models/OneJev1.6:9b` |
| [`verdict-models/decider-2b`](https://huggingface.co/verdict-models/decider-2b) | English | `verdict serve verdict-models/decider-2b` |
| [`verdict-models/von`](https://huggingface.co/verdict-models/von) | English | `verdict serve verdict-models/von` |
| [`verdict-models/laya-multilingual`](https://huggingface.co/verdict-models/laya-multilingual) | 100+ languages | `verdict serve verdict-models/laya-multilingual` |
| [`verdict-models/laya`](https://huggingface.co/verdict-models/laya) | English | `verdict serve verdict-models/laya` |

All five are also mirrored on ModelScope, in the `sgyaqing` namespace:

```bash
# fetches from ModelScope instead of Hugging Face
VERDICT_USE_MODELSCOPE=1 verdict serve sgyaqing/von
```

### Variant families

Some packages are variant families — several model sizes under one repo, picked with a `:variant` tag:

```bash
verdict serve verdict-models/decider-2b     # a plain package: one model per repo
verdict serve verdict-models/OneJev1.6:9b   # a variant family: pick a size with :variant
```

Omitting the tag on a variant family is an error; the output lists the available variants:

```console
$ verdict serve verdict-models/OneJev1.6
error: verdict-models/OneJev1.6 is a variant family; pick one with verdict-models/OneJev1.6:<variant>
available:
  0.8b
  4b
  9b
  27b
example: verdict serve verdict-models/OneJev1.6:0.8b
```

The same `:variant` syntax works for a family directory served by local path.

## API

### `/v1/systemone`

The System One category's native API, defined by Jev and implemented by Laya's runtime, Ollama, LiteLLM, and more.

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Fridge is empty, it'\''s late, and I'\''m tired.",
  "questions": {
    "dinner": {"type": "choice", "instructions": "What should I do for dinner tonight?",
               "criteria": {"order delivery": null, "cook pasta": null, "skip dinner": null}},
    "too_late": {"type": "noul", "instructions": "Is it too late to start cooking?"}
  }}'
```

```json
{
  "model": "OneJev1.6:9b",
  "answers": {
    "dinner": {"type": "choice", "choice": "skip dinner",
               "probabilities": {"order delivery": 0.0124, "cook pasta": 0.0786, "skip dinner": 0.909},
               "confidence": 0.6895},
    "too_late": {"type": "noul", "noul": 0.5068}
  },
  "usage": {"input_tokens": 88, "output_tokens": 0}
}
```

With an image — `assets` carries the bytes, and the reference name in `state` marks where the image goes:

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"screen": "shot-1", "task": "Pay the open invoice from ACME"},
  "questions": {
    "shape": {"type": "choice", "instructions": "Which shape is blue?",
              "criteria": {"circle": "a round shape", "rectangle": "a box", "none": "no blue shape"}}
  },
  "assets": {"shot-1": {"type": "image", "format": "png", "data_base64": "<base64 image bytes>"}}
}'
```

With a video — caller-extracted frames plus the capture rate:

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"recording": "rec-1", "goal": "Sort the invoices by vendor"},
  "questions": {
    "next": {"type": "choice", "instructions": "What should the agent do next?",
             "criteria": {"click": "click an element", "stop": "stop", "scroll": "scroll the list"}}
  },
  "assets": {"rec-1": {"type": "video", "fps": 2.0,
                       "frames": [{"format": "png", "data_base64": "<base64 frame>"},
                                  {"format": "png", "data_base64": "<base64 frame>"}]}}
}'
```

Media needs a media-capable model (such as the `OneJev1.6` variants); a text-only model rejects `assets` with an error.

With the official `typesafe-sdk`:

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080")
r = client.system_one(
    "Fridge is empty, it's late, and I'm tired.",
    {
        "dinner": {"type": "choice",
                   "instructions": "What should I do for dinner tonight?",
                   "criteria": {"order delivery": None, "cook pasta": None, "skip dinner": None}},
        "too_late": {"type": "noul", "instructions": "Is it too late to start cooking?"},
    },
)
r.answers["dinner"].choice        # "skip dinner"
r.answers["dinner"].confidence    # 0.6895
```

### `/v1/decisions`

The OpenAI Decisions protocol, defined by OpenAI and implemented by LiteLLM, SGLang, and more.

```bash
curl -s localhost:8080/v1/decisions -H 'content-type: application/json' -d '{
  "model": "OneJev1.6:9b",
  "input": "Fridge is empty, it'\''s late, and I'\''m tired.",
  "questions": [
    {"type": "choice", "name": "dinner", "instructions": "What should I do for dinner tonight?",
     "choices": [{"value": "order delivery"}, {"value": "cook pasta"}, {"value": "skip dinner"}]},
    {"type": "predicate", "name": "too_late", "instructions": "Is it too late to start cooking?"}
  ]}'
```

```json
{
  "model": "OneJev1.6:9b",
  "answers": [
    {"type": "choice", "name": "dinner", "choice": "skip dinner", "confidence": 0.6895,
     "probabilities": [{"value": "order delivery", "probability": 0.0124},
                       {"value": "cook pasta", "probability": 0.0786},
                       {"value": "skip dinner", "probability": 0.909}]},
    {"type": "predicate", "name": "too_late", "probability": 0.5068}
  ],
  "usage": {"input_tokens": 88, "output_tokens": 0, "total_tokens": 88,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}}
}
```

With an image — the Responses-API shape: `input` is a user message whose `content` mixes `input_text` and `input_image` parts (data URIs only):

```bash
curl -s localhost:8080/v1/decisions -H 'content-type: application/json' -d '{
  "model": "OneJev1.6:9b",
  "input": [{"role": "user", "content": [
    {"type": "input_text", "text": "Task: Pay the open invoice from ACME"},
    {"type": "input_image", "image_url": "data:image/png;base64,<base64 image bytes>"}]}],
  "questions": [
    {"type": "choice", "name": "shape", "instructions": "Which shape is blue?",
     "choices": [{"value": "circle"}, {"value": "rectangle"}, {"value": "none"}]}]}'
```

Verdict 0.3 accepts text and images on `/v1/decisions`, but not video.

With the official `openai` SDK (>= 3.26):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
d = client.decisions.create(
    model="OneJev1.6:9b",
    input="Fridge is empty, it's late, and I'm tired.",
    questions=[
        {"type": "choice", "name": "dinner",
         "instructions": "What should I do for dinner tonight?",
         "choices": [{"value": "order delivery"}, {"value": "cook pasta"}, {"value": "skip dinner"}]},
        {"type": "predicate", "name": "too_late",
         "instructions": "Is it too late to start cooking?"},
    ],
)
d.answers[0].choice        # "skip dinner"
d.answers[0].confidence    # 0.6895
```

### Single-question endpoints

One question per call; text only.

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/choice` | `{"question": "Dinner tonight?", "state": "Fridge is empty, it's late.", "options": ["order delivery", "cook pasta"]}` | `{"choice":1,"probabilities":[0.0963,0.9037],"usage":{"input_tokens":25},"latency_ms":22}` |
| `POST /v1/score` | `{"question": "How polite is this reply?", "state": "Support reply: we refunded your order, sorry for the trouble.", "levels": ["terse", "neutral", "warm"]}` | `{"score":0.7665,"probabilities":[0.2348,0.7639,0.0013],"usage":{"input_tokens":44},"latency_ms":26}` |
| `POST /v1/noul` | `{"claim": "The customer got their money back", "state": "Support reply: we refunded your order, sorry for the trouble."}` | `{"probability":0.9587,"usage":{"input_tokens":45},"latency_ms":25}` |
| `GET /health` | | `{"status":"ok","version":"0.3.0"}` |
| `GET /metrics` | | Prometheus text, not JSON: `verdict_requests_total{task="choice"} 1` … |

### OpenAI-compatible layer

`/v1/chat/completions` and `/v1/models`, for SDK compatibility:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
r = client.chat.completions.create(
    model="<served-model-name>",
    messages=[{"role": "user", "content": "What should I do for dinner tonight?"}],
    extra_body={
        "state": "Fridge is empty, it's late, and I'm tired.",
        "options": ["order delivery", "cook pasta", "skip dinner"],
    },
)
print(r.choices[0].message.content)   # the selected option, e.g. "skip dinner"
print(r.model_extra["verdict"])       # {"choice": 2, "probabilities": [...]}
```

Decision fields ride as top-level extensions via `extra_body`; whichever is present picks the task: `options` → choice, `levels` → score, `claim` → noul. Text only.

### Errors

All errors share one JSON structure -- sending one option to `/v1/choice`, for example:

```json
{"error":{"type":"invalid_request","message":"options must contain at least 2 items (got 1)"}}
```

| `type` | HTTP | when |
|---|---|---|
| `invalid_request` | 400 | malformed body, failed validation, prompts above the input token budget |
| `not_found` | 404 | unknown route |
| `payload_too_large` | 413 | body above `--max-body-mb` (default 64 MiB) |
| `internal` | 500 | engine failure |
| `overloaded` | 503 | queue full -- the request is refused, never dropped |

## Performance

All numbers measured on one machine (RTX 4090, AMD EPYC 7543) — except OneJev1.6-27b, measured on one A100 (80 GB), and decider-2b, measured on a second RTX 4090 host (48 GB) — end to end over HTTP, each model against its author's own server (`laya-serve` 0.4.7, `von serve` from von-sdk 1.3.7, `decider.serve` from decider-ai 1.9.0, `qev` 0.1.4 for OneJev1.6). CPU measurements were taken on the same host with the GPU idle. A ratio above 1x means Verdict is ahead. Servers run with their default flags unless a table notes otherwise — the parameters behind every published number are named next to it. Click a model to expand its tables. OneJev1.6 and decider-2b cells are 10-second duration windows after a warm matrix.

<details>
<summary><strong>OneJev1.6-0.8b</strong></summary>

**text**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 116.7 | 87.5 | 1.33x | 8.90&nbsp;/&nbsp;9.30 | 11.37&nbsp;/&nbsp;11.93 | 1.28x&nbsp;/&nbsp;1.28x |
| 8 | 245.7 | 89.9 | 2.73x | 33.16&nbsp;/&nbsp;35.56 | 89.42&nbsp;/&nbsp;91.05 | 2.70x&nbsp;/&nbsp;2.56x |
| 64 | 289.5 | 89.6 | 3.23x | 225.43&nbsp;/&nbsp;228.53 | 693.37&nbsp;/&nbsp;928.39 | 3.08x&nbsp;/&nbsp;4.06x |

**image**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37.5 | 38.0 | 0.99x | 26.86&nbsp;/&nbsp;28.70 | 26.31&nbsp;/&nbsp;28.73 | 0.98x&nbsp;/&nbsp;1.00x |
| 8 | 135.6 | 38.4 | 3.53x | 59.09&nbsp;/&nbsp;60.98 | 205.37&nbsp;/&nbsp;422.47 | 3.48x&nbsp;/&nbsp;6.93x |
| 64 | 141.2 | 44.8 | 3.15x | 472.29&nbsp;/&nbsp;479.76 | 1639.10&nbsp;/&nbsp;1697.90 | 3.47x&nbsp;/&nbsp;3.54x |

**video**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30.1 | 26.8 | 1.12x | 32.24&nbsp;/&nbsp;40.02 | 31.30&nbsp;/&nbsp;35.30 | 0.97x&nbsp;/&nbsp;0.88x |
| 8 | 109.2 | 31.5 | 3.47x | 72.57&nbsp;/&nbsp;83.08 | 251.40&nbsp;/&nbsp;464.83 | 3.46x&nbsp;/&nbsp;5.59x |
| 64 | 115.2 | 32.0 | 3.60x | 583.80&nbsp;/&nbsp;608.71 | 2018.50&nbsp;/&nbsp;2143.50 | 3.46x&nbsp;/&nbsp;3.52x |

</details>

<details>
<summary><strong>OneJev1.6-4b</strong></summary>

**text**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38.1 | 31.3 | 1.22x | 25.98&nbsp;/&nbsp;27.01 | 31.99&nbsp;/&nbsp;32.38 | 1.23x&nbsp;/&nbsp;1.20x |
| 8 | 47.3 | 32.3 | 1.46x | 163.49&nbsp;/&nbsp;185.22 | 253.52&nbsp;/&nbsp;254.76 | 1.55x&nbsp;/&nbsp;1.38x |
| 64 | 55.2 | 32.0 | 1.72x | 1275.20&nbsp;/&nbsp;1425.70 | 2017.70&nbsp;/&nbsp;2243.00 | 1.58x&nbsp;/&nbsp;1.57x |

**image**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.2 | 16.8 | 1.02x | 57.51&nbsp;/&nbsp;60.47 | 57.68&nbsp;/&nbsp;60.92 | 1.00x&nbsp;/&nbsp;1.01x |
| 8 | 51.6 | 17.6 | 2.93x | 156.16&nbsp;/&nbsp;160.33 | 461.55&nbsp;/&nbsp;706.21 | 2.96x&nbsp;/&nbsp;4.41x |
| 64 | 56.8 | 19.2 | 2.96x | 1254.90&nbsp;/&nbsp;1267.90 | 4074.00&nbsp;/&nbsp;4076.10 | 3.25x&nbsp;/&nbsp;3.21x |

**video**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.2 | 13.2 | 1.15x | 65.60&nbsp;/&nbsp;73.36 | 64.40&nbsp;/&nbsp;71.10 | 0.98x&nbsp;/&nbsp;0.97x |
| 8 | 45.3 | 16.0 | 2.83x | 176.36&nbsp;/&nbsp;239.65 | 512.21&nbsp;/&nbsp;572.87 | 2.90x&nbsp;/&nbsp;2.39x |
| 64 | 51.2 | 19.2 | 2.67x | 1409.70&nbsp;/&nbsp;1514.20 | 4451.10&nbsp;/&nbsp;4452.30 | 3.16x&nbsp;/&nbsp;2.94x |

</details>

<details open>
<summary><strong>OneJev1.6-9b</strong></summary>

**text**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.8 | 20.9 | 1.14x | 42.28&nbsp;/&nbsp;43.17 | 47.76&nbsp;/&nbsp;48.04 | 1.13x&nbsp;/&nbsp;1.11x |
| 8 | 28.4 | 21.0 | 1.35x | 281.16&nbsp;/&nbsp;285.22 | 380.48&nbsp;/&nbsp;380.88 | 1.35x&nbsp;/&nbsp;1.34x |
| 64 | 28.5 | 21.0 | 1.36x | 2244.50&nbsp;/&nbsp;2249.60 | 3044.40&nbsp;/&nbsp;3065.40 | 1.36x&nbsp;/&nbsp;1.36x |

**image**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.4 | 12.5 | 1.00x | 80.33&nbsp;/&nbsp;83.66 | 80.02&nbsp;/&nbsp;86.61 | 1.00x&nbsp;/&nbsp;1.04x |
| 8 | 20.3 | 12.6 | 1.62x | 224.80&nbsp;/&nbsp;1010.90 | 636.24&nbsp;/&nbsp;637.31 | 2.83x&nbsp;/&nbsp;0.63x |
| 64 | 29.0 | 12.4 | 2.33x | 792.10&nbsp;/&nbsp;1716.60 | 4013.70&nbsp;/&nbsp;4014.40 | 5.07x&nbsp;/&nbsp;2.34x |

**video**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.2 | 11.3 | 0.99x | 88.76&nbsp;/&nbsp;98.35 | 87.95&nbsp;/&nbsp;91.22 | 0.99x&nbsp;/&nbsp;0.93x |
| 8 | 32.1 | 11.3 | 2.84x | 242.53&nbsp;/&nbsp;250.81 | 707.92&nbsp;/&nbsp;719.27 | 2.92x&nbsp;/&nbsp;2.87x |
| 64 | 25.3 | 11.4 | 2.22x | 848.00&nbsp;/&nbsp;1977.00 | 4388.30&nbsp;/&nbsp;4389.50 | 5.17x&nbsp;/&nbsp;2.22x |

</details>

<details>
<summary><strong>OneJev1.6-27b</strong> (A100 80 GB)</summary>

**text**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.8 | 8.5 | 1.15x | 102.31&nbsp;/&nbsp;103.45 | 117.76&nbsp;/&nbsp;120.00 | 1.15x&nbsp;/&nbsp;1.16x |
| 8 | 12.9 | 9.1 | 1.42x | 640.28&nbsp;/&nbsp;736.38 | 940.88&nbsp;/&nbsp;942.44 | 1.47x&nbsp;/&nbsp;1.28x |
| 64 | 17.6 | 12.8 | 1.37x | 5095.90&nbsp;/&nbsp;5592.40 | 8209.50&nbsp;/&nbsp;8210.10 | 1.61x&nbsp;/&nbsp;1.47x |

**image**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.7 | 2.5 | 2.28x | 178.32&nbsp;/&nbsp;180.22 | 177.50&nbsp;/&nbsp;5888.90 | 1.00x&nbsp;/&nbsp;32.7x |
| 8 | 14.3 | 5.6 | 2.55x | 555.69&nbsp;/&nbsp;729.59 | 1416.60&nbsp;/&nbsp;1529.50 | 2.55x&nbsp;/&nbsp;2.10x |
| 64 | 20.0 | n/a&nbsp;&dagger; | &mdash; | 4457.90&nbsp;/&nbsp;4884.30 | &mdash; | &mdash; |

**video**

| concurrency | Verdict&nbsp;rps | `qev`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `qev`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.3 | 5.4 | 0.98x | 188.07&nbsp;/&nbsp;208.21 | 184.72&nbsp;/&nbsp;190.44 | 0.98x&nbsp;/&nbsp;0.91x |
| 8 | 14.4 | 6.0 | 2.40x | 574.03&nbsp;/&nbsp;579.80 | 1487.60&nbsp;/&nbsp;1496.60 | 2.59x&nbsp;/&nbsp;2.58x |
| 64 | 19.5 | n/a&nbsp;&dagger; | &mdash; | 4572.40&nbsp;/&nbsp;4861.90 | &mdash; | &mdash; |

At image and video `c=64`, `qev` completed **zero** responses inside the 10-second window: under sustained load its queue bleeds over from the preceding cell and never drains, so every request was still in flight at the deadline (the per-request timeout is 60&nbsp;s — nothing errored). Those `qev` cells are therefore reported as n/a, not as a measured zero. For reference, a supplementary run from a drained queue with an extended per-request timeout collected 23 and 8 responses (8.7 and 7.2 rps; P50/P99 2883.3/4128.1 and 1523.2/1523.3 ms) — a recovery-condition estimate, not the sustained-load protocol result, so no ratios are derived from it. `qev`'s image `c=1` P99 reflects a stall inside its own window (its P50 there is 177.5 ms).

</details>

<details open>
<summary><strong>decider-2b</strong></summary>

**GPU**

| concurrency | Verdict&nbsp;rps | `decider.serve`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `decider.serve`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 140.1 | 136.6 | 1.03x | 6.90&nbsp;/&nbsp;7.89 | 7.29&nbsp;/&nbsp;7.74 | 1.06&nbsp;/&nbsp;0.98x |
| 8 | 620.7 | 332.5 | 1.87x | 12.88&nbsp;/&nbsp;13.76 | 23.91&nbsp;/&nbsp;29.22 | 1.86&nbsp;/&nbsp;2.12x |
| 64 | 672.6 | 596.1 | 1.13x | 95.89&nbsp;/&nbsp;97.93 | 95.36&nbsp;/&nbsp;303.14 | 1.00&nbsp;/&nbsp;3.09x |

</details>

<details>
<summary><strong>von</strong></summary>

**GPU**

| concurrency | Verdict&nbsp;rps | `von serve`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `von serve`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 113.8 | 46.4 | 2.45x | 8.61&nbsp;/&nbsp;11.98 | 22.89&nbsp;/&nbsp;28.46 | 2.66&nbsp;/&nbsp;2.38x |
| 8 | 536.4 | 31.2 | 17.2x | 14.86&nbsp;/&nbsp;16.61 | 250.91&nbsp;/&nbsp;435.99 | 16.9&nbsp;/&nbsp;26.2x |
| 64 | 1476.1 | 22.3 | 66.1x | 42.58&nbsp;/&nbsp;53.24 | 2848.80&nbsp;/&nbsp;3333.40 | 66.9&nbsp;/&nbsp;62.6x |

**CPU**

| concurrency | Verdict&nbsp;rps | `von serve`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `von serve`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.9 | 3.8 | 3.11x | 83.11&nbsp;/&nbsp;97.55 | 272.53&nbsp;/&nbsp;409.13 | 3.28&nbsp;/&nbsp;4.19x |
| 8 | 20.7 | 5.0 | 4.17x | 383.82&nbsp;/&nbsp;449.74 | 1607.00&nbsp;/&nbsp;1927.80 | 4.19&nbsp;/&nbsp;4.29x |
| 64 | 21.2 | 4.8 | 4.37x | 2989.60&nbsp;/&nbsp;3430.80 | 13644.60&nbsp;/&nbsp;15478.20 | 4.56&nbsp;/&nbsp;4.51x |

</details>

<details>
<summary><strong>laya-multilingual</strong></summary>

**GPU**

| concurrency | Verdict&nbsp;rps | `laya-serve`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `laya-serve`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 123.2 | 49.1 | 2.51x | 8.03&nbsp;/&nbsp;10.29 | 18.54&nbsp;/&nbsp;28.92 | 2.31&nbsp;/&nbsp;2.81x |
| 8 | 553.4 | 46.3 | 11.9x | 13.99&nbsp;/&nbsp;19.79 | 184.36&nbsp;/&nbsp;204.24 | 13.2&nbsp;/&nbsp;10.3x |
| 64 | 2534.4 | refused&nbsp;requests | — | 24.67&nbsp;/&nbsp;32.23 | refused&nbsp;requests | — |

**CPU**

| concurrency | Verdict&nbsp;rps | `laya-serve`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `laya-serve`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 19.7 | 6.2 | 3.16x | 50.51&nbsp;/&nbsp;64.37 | 161.13&nbsp;/&nbsp;189.54 | 3.19&nbsp;/&nbsp;2.94x |
| 8 | 33.4 | 5.7 | 5.84x | 238.10&nbsp;/&nbsp;269.73 | 1396.60&nbsp;/&nbsp;1661.30 | 5.87&nbsp;/&nbsp;6.16x |
| 64 | 48.6 | refused&nbsp;requests | — | 1312.40&nbsp;/&nbsp;1422.60 | refused&nbsp;requests | — |

</details>

<details>
<summary><strong>laya</strong></summary>

**GPU**

| concurrency | Verdict&nbsp;rps | `laya-serve`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `laya-serve`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 121.4 | 34.4 | 3.53x | 8.37&nbsp;/&nbsp;11.07 | 28.58&nbsp;/&nbsp;32.73 | 3.42&nbsp;/&nbsp;2.96x |
| 8 | 501.4 | 35.4 | 14.2x | 15.99&nbsp;/&nbsp;22.42 | 227.96&nbsp;/&nbsp;250.60 | 14.3&nbsp;/&nbsp;11.2x |
| 64 | 1353.0 | refused&nbsp;requests | — | 46.71&nbsp;/&nbsp;65.19 | refused&nbsp;requests | — |

**CPU**

| concurrency | Verdict&nbsp;rps | `laya-serve`&nbsp;rps | rps&nbsp;ratio | Verdict&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | `laya-serve`&nbsp;P50&nbsp;/&nbsp;P99&nbsp;(ms) | P50&nbsp;/&nbsp;P99&nbsp;ratio |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 11.8 | 4.4 | 2.67x | 84.21&nbsp;/&nbsp;100.87 | 223.81&nbsp;/&nbsp;273.16 | 2.66&nbsp;/&nbsp;2.71x |
| 8 | 18.4 | 4.3 | 4.23x | 436.70&nbsp;/&nbsp;533.71 | 1882.50&nbsp;/&nbsp;2131.30 | 4.31&nbsp;/&nbsp;3.99x |
| 64 | 16.9 | refused&nbsp;requests | — | 3796.80&nbsp;/&nbsp;4233.90 | refused&nbsp;requests | — |

</details>

## Correctness

Verdict runs its own inference pipeline, graded against a golden standard: the official SDK's offline output on CPU fp32. In accuracy PKs, Verdict and the model author's own server are graded against the same golden standard on one RTX 4090 (CUDA) — worst-case deviation across the whole corpus:

| Model | Verdict | official&nbsp;server | result |
|---|---:|---:|:--|
| laya | 0.0 | 0.943 | Verdict ahead |
| laya-multilingual | 1e-4 | 1.74 | Verdict ahead |
| von | 0.0 | 0.0 | tied |
| decider-2b | 2.00e-2 | 1.35e-2 | tied |
| OneJev1.6-0.8b | 2.49e-2 | 8.55e-1 | Verdict ahead |
| OneJev1.6-4b | 4.95e-2 | 9.03e-1 | Verdict ahead |
| OneJev1.6-9b | 4.59e-2 | 8.99e-1 | Verdict ahead |

## Configuration

One subcommand: `verdict serve <MODEL_DIR_OR_REPO>`.

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | 127.0.0.1&nbsp;/&nbsp;8080 | address and port to serve on |
| `--device` | auto | `auto` uses CUDA when available, else CPU; `cpu` / `cuda` force the choice |
| `--max-batch-size` | 8 | max requests per batch |
| `--batch-window-ms` | 5 | upper bound on batch accumulation time, reached only when `--max-batch-size` is large relative to this window; a batch ships sooner when it fills up or after 0.5&nbsp;ms of arrival silence |
| `--max-input-tokens` | off | reject rendered prompts longer than this; unset, an over-long prompt is truncated at the model's declared `max_len`, same as the official SDK |
| `--max-body-mb` | 64 | request body size cap in MiB |
| `--queue-cap` | 1024 | max queued requests; excess requests are rejected with 503 |
| `--served-model-name` | model directory's name | the model name clients address |
| `--policy` | fcfs | `fcfs` serves in arrival order; `priority` serves lower request priority first |
| `--data-parallel` | 1 | N engine replicas; helps on many-core CPU hosts |
| `--force-download` | — | if passed, re-download the model package instead of using the cache |

Environment variables:

| Variable | Effect |
|---|---|
| `HF_ENDPOINT` | Alternate Hugging Face endpoint, for fetching through a mirror. |
| `HF_TOKEN` | Auth for gated Hugging Face repos. |
| `VERDICT_USE_MODELSCOPE` | `1` or `true` resolves all repo ids against ModelScope; unset or any other value keeps everything on Hugging Face. |
| `MODELSCOPE_API_TOKEN` | Auth for ModelScope repos. |
| `VERDICT_CACHE` | Model cache root; default `$XDG_CACHE_HOME/verdict`, i.e. `~/.cache/verdict`. |
| `ORT_DYLIB_PATH` | Path to the onnxruntime library; default: the model's `lib/` or beside the binary. |

## Requirements

For the CUDA image:

| | Minimum |
|---|---|
| GPU | compute capability **7.5** or newer |
| Driver | **580** or newer on Linux |

`decider-2b` and the `OneJev1.6` variants serve through the PyTorch runtime, which runs only on CUDA GPUs.

## License

Apache-2.0

## Acknowledgments

Several recipes reimplement behaviour specified by the official implementations
of the models they serve, and parts of the scheduler follow published design
patterns from other serving projects. The projects, their licenses and what we
take from each are listed in [NOTICE](NOTICE).
