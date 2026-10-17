#!/usr/bin/env python3
"""Verdict torch runtime executor: one generic forward worker per engine
replica. It owns exactly one job -- run the checkpoint's forward pass and
return the letter logits at the requested slots. Row building, slot choice,
calibration and the HTTP protocol live on the Rust side; this process is a
kernel executor, not a serving stack, and holds no vendor inference code.

Protocol: one JSON object per line on stdin, one JSON reply per line on
stdout; bulk tensors travel as base64 little-endian bytes
(`<ids/mask/slots>_b64` in, `logits_b64` out) -- JSON text of a batch's 8k
floats costs ~10 ms per round trip in dumps/parse, b64 is two memcpys.
Control fields (ready/error/fwd_ms) stay plain JSON values. Startup prints
one {"ready"} line ONLY after the model is loaded and a warmup forward has
run (torch / triton kernels compile on the first call; without the warmup
the first real request pays ~1 s).

Eager attention is not forced: transformers' "eager" attention
materializes a 4D mask via per-call host->device tensor copies, which CUDA
graph capture forbids; sdpa with is_causal builds none (and matches the
reference's own backend choice). The cuDNN SDPA backend is disabled because
it returns wrong (finite) output for masked rectangular attention on this
torch/CUDA generation.
"""

import base64
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LETTERS = 255


class GraphRunner:
    """CUDA-graph wrapper around the letter forward: capture one graph per
    (batch bucket, length bucket)
    over a static input buffer, replay by copying the real batch in and
    slicing the slot rows out of the static output. Shapes outside the
    bucket grid (or beyond the graph cap) fall back to the same eager
    forward the ungraphed path uses -- a new shape must never force a
    capture mid-flight.

    No attention mask in the captured region: the model
    is causal, so right-padding can never be attended by real tokens, and a
    mask built INSIDE the forward (transformers' eager_mask) allocates a
    fresh tensor per call -- that host->device copy is illegal during graph
    capture (unpinned). The eager fallback path keeps the explicit mask, so
    the smoke compares graph-unmasked vs eager-masked outputs directly.
    """

    B_BUCKETS = (1, 2, 4, 8, 16, 32)
    L_BUCKETS = (64, 128, 192, 256, 384)
    MAX_GRAPHS = 48

    def __init__(self, model, head_w, pad_id, warmup=True, compile_fwd=True):
        self.model = model
        self.head_w = head_w
        self.pad_id = pad_id
        self.dev = "cuda"
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}
        # The GRAPH captures the COMPILED forward; the
        # eager fallback stays uncompiled ("a new shape must not compile").
        # dynamo's default cache limits stop compiling part way through the
        # warm-up grid and silently leave the rest eager -- raise them, sized
        # by the grid the same way.
        if compile_fwd:
            from torch import _dynamo

            n = len(self.B_BUCKETS) * len(self.L_BUCKETS)
            _dynamo.config.cache_size_limit = max(256, n + 16)
            _dynamo.config.accumulated_cache_size_limit = max(1 << 16, 64 * n)
            for k in ("accumulated_recompile_limit", "recompile_limit"):
                if hasattr(_dynamo.config, k):
                    setattr(_dynamo.config, k, max(1 << 16, 64 * n))
            self._fwd_impl = torch.compile(self._eager_core, dynamic=False)
        else:
            self._fwd_impl = self._eager_core
        if warmup:
            for l in (64, 192):
                for b in (1, 4, 16, 32):
                    self.capture(b, l)
                for b in (2, 8):
                    self.capture(b, l)

    def _eager_core(self, ids):
        """The compiled/fallback forward body: backbone (no cache) + letter
        readout."""
        hs = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        return (hs @ self.head_w.T).float()  # [B, L, 255]

    def _fwd(self, ids):
        return self._fwd_impl(ids)

    @torch.no_grad()
    def capture(self, b, l):
        s_ids = torch.full((b, l), self.pad_id, dtype=torch.long, device=self.dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._fwd(s_ids)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            s_out = self._fwd(s_ids)
        self.graphs[(b, l)] = (s_ids, s_out, g)

    def _eager_readout(self, ids, mask, slot_batch, slot_idx):
        """Uncompiled fallback for shapes outside the captured grid -- the
        vendor keeps this path eager too (never compile a new shape)."""
        hs = self.model.model(input_ids=ids, attention_mask=mask,
                              use_cache=False).last_hidden_state
        return (hs[slot_batch, slot_idx] @ self.head_w.T).float()

    def plan(self, b, l):
        length = next((x for x in self.L_BUCKETS if l <= x), None)
        batch = next((x for x in self.B_BUCKETS if b <= x), None)
        if length is None or batch is None:
            return None
        return (batch, length)

    @torch.no_grad()
    def run(self, ids, mask, slot_batch, slot_idx):
        b, l = ids.shape
        key = self.plan(b, l)
        if key is None:
            return self._eager_readout(ids, mask, slot_batch, slot_idx)
        if key not in self.graphs and len(self.graphs) < self.MAX_GRAPHS:
            self.capture(*key)
        entry = self.graphs.get(key)
        if entry is None:
            return self._eager_readout(ids, mask, slot_batch, slot_idx)
        # Pad the BATCH up to the bucket: the
        # graph is shape-specialized (torch.compile dynamic=False), so an
        # unrounded batch size (3, 5, 6, 7...) would trigger a fresh compile
        # on first sight -- a large p99 stall. Padded rows carry
        # pad ids and are cut away by the slot slice below (their slots are
        # never requested).
        bb, ll = key
        s_ids, s_out, g = entry
        s_ids.fill_(self.pad_id)
        s_ids[:b, :l].copy_(ids)
        g.replay()
        # clone before the next replay overwrites the static output
        return s_out[slot_batch, slot_idx].float().clone()


def _i64(b64):
    """base64 little-endian int64 vector -> cuda tensor (input ids, masks,
    slot indices). The .copy() makes the buffer writable for torch."""
    return torch.from_numpy(
        np.frombuffer(base64.b64decode(b64), dtype="<i8").copy()
    ).to("cuda")


def _f32_b64(t):
    """f32 tensor -> base64 little-endian bytes for the wire."""
    return base64.b64encode(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).decode("ascii")


def derive_letter_ids(tok):
    """The letter-readout family's label table: upper-case names A..Z, then
    AA, AB, ... keeping every name that encodes to exactly one token, in
    that order, until the table is full. Mirrors the Rust rule in
    src/tokenize.rs (derive_letter_table) -- a tokenizer that cannot fill
    the table fails here, not later."""
    def single(name):
        ids = tok.encode(name, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    table = []
    for c in range(ord("A"), ord("Z") + 1):
        i = single(chr(c))
        if i is not None:
            table.append(i)
    done = False
    for a in range(ord("A"), ord("Z") + 1):
        if done:
            break
        for b in range(ord("A"), ord("Z") + 1):
            if len(table) >= MAX_LETTERS:
                done = True
                break
            i = single(chr(a) + chr(b))
            if i is not None:
                table.append(i)
    if len(table) < MAX_LETTERS:
        raise RuntimeError(
            f"this tokenizer has only {len(table)} single-token upper-case letter "
            f"names; the letter-readout family needs {MAX_LETTERS}"
        )
    if len(set(table)) != len(table):
        raise RuntimeError("two option labels tokenize to the same id")
    return table


def main():
    ckpt = sys.argv[1]
    # --device auto reaches this process on a CUDA-less host (the CLI's
    # CPU refusal only fires for an explicit `cpu`); fail with the reason,
    # not with a downstream torch error.
    if not torch.cuda.is_available():
        print(json.dumps({
            "error": "the torch runtime requires CUDA: torch.cuda.is_available() is False "
                     "on this host (v0.2 serves torch packages on CUDA only)"
        }), flush=True)
        sys.exit(2)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    # The cuDNN SDPA
    # backend returns wrong (finite) output for masked rectangular attention
    # on this torch/CUDA generation; captured graphs keep the backend they
    # were captured with, so it must be off BEFORE any capture.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    model.eval()
    # Package layout ships the tokenizer in a tokenizer/ subdir; an official
    # checkpoint snapshot has it at the root. Try both, loudly.
    tok_dir = os.path.join(ckpt, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir if os.path.isfile(os.path.join(tok_dir, "tokenizer.json")) else ckpt
    )
    letter_ids = torch.tensor(derive_letter_ids(tokenizer), dtype=torch.long,
                              device="cuda")
    head_w = model.get_output_embeddings().weight[letter_ids]  # [255, H]

    # Pad token: Qwen-family tokenizers define no pad_token_id. The name is
    # manifest data (model.json's pad_token), resolved through the tokenizer
    # exactly like the Rust side (token_to_id), never hardcoded.
    manifest_path = os.path.join(ckpt, "model.json")
    pad_name = "<|endoftext|>"
    if os.path.isfile(manifest_path):
        pad_name = json.load(open(manifest_path)).get("pad_token", pad_name)
    pad_id = tokenizer.convert_tokens_to_ids(pad_name)
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    warmup_ids = torch.ones(1, 4, dtype=torch.long, device="cuda")
    warmup_mask = torch.ones(1, 4, dtype=torch.long, device="cuda")
    with torch.no_grad():
        _ = model.model(input_ids=warmup_ids, attention_mask=warmup_mask)

    # CUDA graphs + torch.compile on by default: both are needed to reach the
    # intended throughput and latency. The env switches are A/B escape
    # hatches, not product flags.
    runner = None
    if os.environ.get("VERDICT_TORCH_GRAPHS", "1") != "0":
        runner = GraphRunner(
            model,
            head_w,
            pad_id,
            compile_fwd=os.environ.get("VERDICT_TORCH_COMPILE", "1") != "0",
        )

    print(json.dumps({"ready": True, "load_s": time.time() - t0,
                      "letters": MAX_LETTERS,
                      "graphs": len(runner.graphs) if runner else 0}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as err:
            print(json.dumps({"error": f"bad request json: {err}"}), flush=True)
            continue
        if req.get("cmd") == "quit":
            break
        try:
            t = time.time()
            slot_batch = _i64(req["slot_batch_b64"])
            b = slot_batch.shape[0]
            ids = _i64(req["input_ids_b64"]).view(b, -1)
            mask = _i64(req["attention_mask_b64"]).view(b, -1)
            slot_idx = _i64(req["slot_idx_b64"])
            if runner is not None:
                logits = runner.run(ids, mask, slot_batch, slot_idx)
            else:
                with torch.no_grad():
                    hs = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
                    logits = (hs[slot_batch, slot_idx] @ head_w.T).float()  # [N, 255]
            print(json.dumps({"logits_b64": _f32_b64(logits),
                              "n_rows": logits.shape[0],
                              "fwd_ms": (time.time() - t) * 1000.0}), flush=True)
        except Exception as err:  # noqa: BLE001 -- a request-level failure is a
            # reply, never a crash: the engine must be able to propagate it.
            print(json.dumps({"error": f"forward failed: {err}"}), flush=True)


if __name__ == "__main__":
    main()
