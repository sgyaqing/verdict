#!/usr/bin/env python3
"""Verdict torch runtime executor: one forward worker per engine replica.

stdin/stdout carry one JSON object per line; bulk tensors are base64 little-endian bytes; {"ready"} follows the model load and a warmup forward.

Attention stays sdpa with is_causal: transformers' eager attention builds a 4D mask with a per-call host->device copy, which graph capture forbids, and the cuDNN SDPA backend returns wrong output for masked rectangular attention on this torch/CUDA generation."""

import base64
import collections
import hashlib
import copy
import io
import json
import re
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LETTERS = 255

def _double_ladder(cap: int, floor: int = 1) -> tuple:
    """Bucket ladder: every count maps up to one of these sizes."""
    out, v = [], floor
    while v < cap:
        out.append(v)
        v *= 2
    out.append(cap)
    return tuple(out)

GRAPH_ROWS_CAP = int(os.environ.get("VERDICT_GRAPH_ROWS_CAP", 8))
GRAPH_ROW_LADDER = _double_ladder(GRAPH_ROWS_CAP)
FORK_ROWS_CAP = int(os.environ.get("VERDICT_FORK_ROWS_CAP", GRAPH_ROWS_CAP))
MERGE_PAD_LIMIT = int(os.environ.get("VERDICT_FORK_MERGE_PAD", 512))
GRAPH_WIDTH_STEP = int(os.environ.get("VERDICT_GRAPH_WIDTH_STEP", 16))
GRAPH_PREFIX_STEP = int(os.environ.get("VERDICT_GRAPH_PREFIX_STEP", 32))
GRAPH_MAX_LEN = int(os.environ.get("VERDICT_GRAPH_MAX_LEN", 512))

GRAPH_LEN_BUCKETS_DEFAULT = (64, 96, 128, 160, 192, 256, 320, 400, 512, 640, 800, 1024)
FORK_GRAPH_MAX_PREFIX = int(os.environ.get("VERDICT_FORK_GRAPH_MAX_PREFIX", 4096))
FORK_GRAPH_MAX_WIDTH = int(os.environ.get("VERDICT_FORK_GRAPH_MAX_WIDTH", 64))

COVERAGE = collections.Counter()
SERVED_ROWS = [0]

def note_uncovered(reason: str, rows: int) -> None:
    """Rows that ran without the graph their class would have given them, by why."""
    if reason:
        COVERAGE[reason] += rows

FATAL_CUDA_MARKERS = ("illegal memory access", "Sticky error", "cudaErrorIllegalAddress",
                      "CUDA_ERROR_ILLEGAL_ADDRESS", "device-side assert",
                      "unspecified launch failure")

def _fatal_cuda(err) -> bool:
    """A sticky CUDA fault leaves the context unusable; the executor must die so the server restarts it."""
    text = f"{type(err).__name__}: {err}"
    return any(marker in text for marker in FATAL_CUDA_MARKERS)

def chunk_index_tables(cu_seqlens, chunk_size, n_cap, nt):
    """FLA chunk_indices [nt, 2] and chunk_offsets [n_cap + 1] built from cu_seqlens with
    device ops only; the caller fixes n_cap and nt so a captured graph replays any
    segment structure. Entries past the real chunk count resolve to the last segment."""
    cu = cu_seqlens.to(torch.int32)
    counts = torch.div(cu[1:] - cu[:-1] + chunk_size - 1, chunk_size,
                       rounding_mode="floor")
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    idx = torch.arange(nt, dtype=torch.int32, device=cu.device)
    seq = torch.searchsorted(offsets[1:], idx, right=True).clamp(max=n_cap - 1)
    return torch.stack([seq, idx - offsets[seq]], 1), offsets



class PackedRunner:
    """Flattens a padded [B, L] batch into one varlen [1, T] forward: full-attention
    layers run FA2 varlen (flash_attn), GDN layers take cu_seqlens into the FLA chunk
    kernel, and seq_idx keeps the short conv from bleeding across packed boundaries.
    Eager only; no CUDA graphs yet."""

    def __init__(self, model, head_w, pad_id):
        self.model = model
        self.head_w = head_w
        self.pad_id = pad_id
        self.dev = next(model.parameters()).device

    def run(self, ids, mask, slot_batch, slot_idx, host=None):
        dev = ids.device
        if host is not None:
            ids_np, mask_np = host
            keep = mask_np.astype(bool)
            lens_np = keep.sum(1).astype("<i8")
            flat_np = ids_np[keep]
            pos_np = np.broadcast_to(
                np.arange(ids_np.shape[1], dtype="<i8"), ids_np.shape)[keep]
            cu_np = np.zeros(lens_np.shape[0] + 1, dtype="<i4")
            np.cumsum(lens_np, out=cu_np[1:])
            seq_np = np.repeat(np.arange(lens_np.shape[0], dtype="<i4"), lens_np)
            flat = torch.from_numpy(flat_np).to(dev).unsqueeze(0)
            flat_pos = torch.from_numpy(pos_np).to(dev).unsqueeze(0)
            cu = torch.from_numpy(cu_np).to(dev)
            seq_idx = torch.from_numpy(seq_np).to(dev).unsqueeze(0)
            max_len = int(lens_np.max())
        else:
            lens = mask.sum(1)
            max_len = int(lens.max())
            nz = mask.flatten().nonzero(as_tuple=True)[0]
            flat = ids.flatten()[nz].unsqueeze(0)
            pos = torch.arange(ids.shape[1], device=dev).expand_as(ids)
            flat_pos = pos.flatten()[nz].unsqueeze(0)
            seq_idx = torch.arange(ids.shape[0], device=dev, dtype=torch.int32)
            seq_idx = seq_idx.repeat_interleave(lens).unsqueeze(0)
            cu = torch.zeros(ids.shape[0] + 1, dtype=torch.int32, device=dev)
            cu[1:] = torch.cumsum(lens, 0).to(torch.int32)
        with torch.inference_mode():
            hs = self.model.model(
                input_ids=flat,
                position_ids=flat_pos,
                attention_mask=None,
                cu_seq_lens_q=cu,
                cu_seq_lens_k=cu,
                max_length_q=max_len,
                max_length_k=max_len,
                seq_idx=seq_idx,
                use_cache=False,
            ).last_hidden_state
        flat_slot = cu[:-1].long() + slot_idx
        hs_rows = hs[0][flat_slot]
        return (hs_rows.float() @ self.head_w.T), None


class GraphRunner:
    """CUDA graphs over the letter forward, one per (rows, length) bucket.

    The captured region carries no attention mask: the model is causal so right padding is never attended, and a mask built inside the forward allocates a tensor per call, which capture forbids."""

    B_BUCKETS = GRAPH_ROW_LADDER
    PREWARM_MISS_LIMIT = 3

    def __init__(self, model, head_w, pad_id, max_len=32768, compile_fwd=False,
                 graph_max_len: int = GRAPH_MAX_LEN):
        self.model = model
        self.head_w = head_w
        self.pad_id = pad_id
        self.dev = "cuda"
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.L_BUCKETS = self._l_buckets(max_len, graph_max_len)
        n_cells = len(self.B_BUCKETS) * len(self.L_BUCKETS)
        self.compiled = bool(compile_fwd)
        if compile_fwd:
            from torch import _dynamo

            _dynamo.config.cache_size_limit = max(256, 4 * n_cells)
            _dynamo.config.accumulated_cache_size_limit = max(1 << 16, 64 * n_cells)
            for k in ("accumulated_recompile_limit", "recompile_limit"):
                if hasattr(_dynamo.config, k):
                    setattr(_dynamo.config, k, max(1 << 16, 64 * n_cells))
            self._fwd_impl = torch.compile(self._eager_core, dynamic=True)
        else:
            self._fwd_impl = self._eager_core

    @staticmethod
    def _l_buckets(max_len: int, cap: int = GRAPH_MAX_LEN, steps: int = 5,
                   measured: bool = True) -> tuple:
        """Length buckets from 64 up to min(cap, max(384, max_len // 8)); the measured list for text, geometric in `steps` for media."""
        cap = min(cap, max(384, max_len // 8))
        explicit = os.environ.get("VERDICT_GRAPH_LEN_BUCKETS", "")
        if explicit:
            picked = sorted({int(x) for x in explicit.split(",") if x.strip()})
            out = tuple(x for x in picked if 0 < x <= cap)
            if out:
                return out
        out = [x for x in GRAPH_LEN_BUCKETS_DEFAULT if x <= cap] if measured else []
        if len(out) >= 2:
            while out[-1] < cap:
                out.append(min(max(int(round(out[-1] * 1.25)), out[-1] + 1), cap))
        else:
            ratio = (cap / 64) ** (1.0 / steps)
            out, v = [64], 64.0
            for _ in range(steps):
                v = min(round(v * ratio), cap)
                if v > out[-1]:
                    out.append(v)
            if out[-1] < cap:
                out[-1] = cap
        return tuple(out)

    def compile_selfcheck(self, tolerance: float = 5e-2):
        """Compiled-vs-eager gate on softmax probabilities, run at load."""
        vocab = self.model.get_input_embeddings().weight.shape[0]
        gen = torch.Generator(device=self.dev).manual_seed(20261023)
        ids = torch.randint(1, vocab, (2, 64), generator=gen, device=self.dev)
        with torch.no_grad():
            a = torch.softmax(self._fwd_impl(ids).float(), dim=-1)
            b = torch.softmax(self._eager_core(ids).float(), dim=-1)
        diff = float((a - b).abs().max())
        return diff <= tolerance, (
            f"max diff {diff:.2e} < tol {tolerance:.0e}")

    def _eager_core(self, ids):
        """The eager forward body: backbone (no cache) + letter readout."""
        hs = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        return (hs.float() @ self.head_w.T)

    def _fwd(self, ids):
        return self._fwd_impl(ids)

    @torch.no_grad()
    def capture(self, b, l):
        t0 = time.perf_counter()
        _require_room()
        s_ids = torch.full((b, l), self.pad_id, dtype=torch.long, device=self.dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._fwd(s_ids)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()

        def run():
            return self._fwd(s_ids)

        s_out, g = _capture_graph(run, self.pool)
        self.graphs[(b, l)] = (s_ids, s_out, g)
        print(f"[executor] captured a text graph for {b} rows x {l} tokens "
              f"in {(time.perf_counter() - t0) * 1e3:.0f} ms "
              f"(free {torch.cuda.mem_get_info()[0] >> 20} MiB)",
              file=sys.stderr, flush=True)
        return self.graphs[(b, l)]

    def prewarm(self) -> None:
        """Capture every (rows, length) cell, largest first; a cell that does not fit stays on the lazy path."""
        keys = sorted(((b, l) for b in self.B_BUCKETS for l in self.L_BUCKETS),
                      key=lambda k: k[0] * k[1], reverse=True)
        for key in keys:
            try:
                self.capture(*key)
            except (GraphMemoryLow, torch.OutOfMemoryError, RuntimeError) as err:
                if _fatal_cuda(err):
                    raise
                torch.cuda.empty_cache()
                print(f"[executor] text graph {key[0]}x{key[1]} not prewarmed "
                      f"({type(err).__name__}: {err}); eager for the life of the process",
                      file=sys.stderr, flush=True)
        print(f"[executor] text graphs prewarmed {len(self.graphs)}/{len(keys)} "
              f"over rows {list(self.B_BUCKETS)} x lengths {list(self.L_BUCKETS)}",
              file=sys.stderr, flush=True)

    def _eager_readout(self, ids, mask, slot_batch, slot_idx):
        """Eager readout with the all-ones mask omitted (the causal fast path needs none) and inference_mode on."""
        kw = {} if bool(mask.all()) else {"attention_mask": mask}
        with torch.inference_mode():
            hs = self.model.model(input_ids=ids, use_cache=False, **kw).last_hidden_state
            logits = hs[slot_batch, slot_idx].float() @ self.head_w.T
            return logits

    def plan(self, b, l):
        """(rows bucket, length bucket), or None when outside the grid."""
        length = next((x for x in self.L_BUCKETS if l <= x), None)
        batch = next((x for x in self.B_BUCKETS if b <= x), None)
        if length is None or batch is None:
            return None
        return (batch, length)

    def stats(self):
        return len(self.graphs), len(self.B_BUCKETS) * len(self.L_BUCKETS)

    def why(self, b, l):
        """Why a batch has no graph: past the ladder, or a cell the boot ladder could not fit."""
        if l > self.L_BUCKETS[-1]:
            return "len"
        if b > self.B_BUCKETS[-1]:
            return "rows"
        return "degraded"

    @torch.no_grad()
    def run(self, ids, mask, slot_batch, slot_idx):
        """Graphs path: only cells captured at boot replay; every other shape runs eagerly for the life of the process."""
        b, l = ids.shape
        key = self.plan(b, l)
        entry = self.graphs.get(key) if key is not None else None
        if entry is None:
            return self._eager_readout(ids, mask, slot_batch, slot_idx), self.why(b, l)
        bb, ll = key
        s_ids, s_out, g = entry
        s_ids.fill_(self.pad_id)
        s_ids[:b, :l].copy_(ids)
        g.replay()
        return s_out[slot_batch, slot_idx].float().clone(), None

DEV = "cuda"

MM_PROFILE = os.environ.get("VERDICT_MM_PROFILE") == "1"

BATCH_PROFILE = os.environ.get("VERDICT_BATCH_PROFILE") == "1"
_BATCH_LAST = [None]


_TORCH_PROFILE = os.environ.get("VERDICT_TORCH_PROFILE", "")
_TPROF = {"prof": None, "seen": 0}


def _tprof_tick():
    """Start torch.profiler after 20 served batches and dump a chrome trace
    after VERDICT_TORCH_PROFILE more batches."""
    if not _TORCH_PROFILE:
        return
    st = _TPROF
    st["seen"] += 1
    if st["seen"] == 20:
        st["prof"] = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
        st["prof"].start()
        print(f"[executor] torch profiler on for {_TORCH_PROFILE} batches",
              file=sys.stderr, flush=True)
    elif st["seen"] == 20 + int(_TORCH_PROFILE) and st["prof"] is not None:
        st["prof"].stop()
        out = os.environ.get("VERDICT_TORCH_PROFILE_OUT",
                             "/tmp/verdict-torch-profile.json")
        st["prof"].export_chrome_trace(out)
        print(f"[executor] torch profile dumped to {out}",
              file=sys.stderr, flush=True)
        st["prof"] = None


def _batch_prof(b, l, rows, kind, started):
    if not BATCH_PROFILE:
        return
    now = time.perf_counter()
    gap = (now - _BATCH_LAST[0]) * 1e3 if _BATCH_LAST[0] is not None else -1.0
    _BATCH_LAST[0] = now
    real = rows.sum(1).float()
    pre = FORK_STATS.get("prefix", -1.0) if kind == "fork" else -1.0
    br = FORK_STATS.get("branch", -1.0) if kind == "fork" else -1.0
    mt = FORK_STATS.get("multi", -1.0) if kind == "fork" else -1.0
    asm = FORK_STATS.get("asm", -1.0) if kind == "fork" else -1.0
    cpy = FORK_STATS.get("copy", -1.0) if kind == "fork" else -1.0
    print(f"[batch-prof] ts={time.time():.3f} b={b} l={l} real_max={int(real.max())} "
          f"real_mean={float(real.mean()):.1f} kind={kind} "
          f"wall={(time.time() - started) * 1000.0:.2f} gap={gap:.2f} "
          f"read={READ_MS[0]:.2f} prev_emit={EMIT_LAST[0]:.2f} "
          f"ipc_in={IPC_LAST[0]:.2f} forkcall={PHASE_STATS['forkcall']:.2f} "
          f"rest={PHASE_STATS['rest']:.2f} "
          f"prefix={pre:.2f} branch={br:.2f} multi={mt:.0f} groups={FORK_STATS.get('groups', -1.0):.0f} "
          f"asm={asm:.2f} copy={cpy:.2f} "
          f"phit={PREFIX_STATS['hit']} pmiss={PREFIX_STATS['miss']} "
          f"pmb={PREFIX_STATS['bytes'] // (1 << 20)}",
          file=sys.stderr, flush=True)

MM_DRIFT_CHECK = os.environ.get("VERDICT_MM_DRIFT_CHECK", "0") == "1"

MM_FORK = [os.environ.get("VERDICT_MM_FORK", "1") != "0"]
MM_GRAPHS = [os.environ.get("VERDICT_MM_GRAPHS", "1") != "0"]
MM_BATCH = [os.environ.get("VERDICT_MM_BATCH", "1") != "0"]
TEXT_FORK = [os.environ.get("VERDICT_TEXT_FORK", "1") != "0"]
TEXT_FORK_MIN_PREFIX = GRAPH_PREFIX_STEP

FORK_STATS = {}

PREFIX_STATS = {"hit": 0, "miss": 0, "bytes": 0}

IPC_LAST = [0.0]

READ_MS = [0.0]

EMIT_LAST = [0.0]

PHASE_STATS = {"forkcall": -1.0, "rest": -1.0}
FORK_MIN_ROWS = 6

def compile_cache_dir(manifest: dict, ckpt: str):
    """Point TORCHINDUCTOR_CACHE_DIR at <cache>/<pkg>/compile/<key>; the key names everything that can change the artifacts."""
    root = os.environ.get("VERDICT_COMPILE_CACHE")
    if not root:
        return None
    import transformers

    parts = [
        torch.__version__,
        transformers.__version__,
        str(torch.version.cuda),
        os.environ.get("VERDICT_EXECUTOR_SHA", "0"),
        json.dumps(manifest.get("checkpoint"), sort_keys=True),
    ]
    try:
        with open(os.path.join(ckpt, "config.json"), "rb") as f:
            parts.append(hashlib.sha256(f.read()).hexdigest())
    except OSError:
        parts.append("no-config")
    total = 0
    try:
        for name in os.listdir(ckpt):
            p = os.path.join(ckpt, name)
            if os.path.isfile(p):
                total += os.path.getsize(p)
    except OSError:
        pass
    parts.append(str(total))
    if torch.cuda.is_available():
        parts.append(torch.cuda.get_device_name(0))
        parts.append(str(torch.cuda.get_device_capability(0)))
    key = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    d = os.path.join(root, key)
    os.makedirs(d, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = d
    label = os.path.join(d, "key.json")
    if not os.path.exists(label):
        with open(label, "w") as f:
            json.dump({"key": key, "parts": parts,
                       "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      f, indent=2)
    stale = [e for e in os.listdir(root)
             if e != key and os.path.isdir(os.path.join(root, e))]
    for e in stale:
        shutil.rmtree(os.path.join(root, e), ignore_errors=True)
    if stale:
        print(f"[executor] removed {len(stale)} outdated compile "
              f"cache{'s' if len(stale) > 1 else ''}",
              file=sys.stderr, flush=True)
    return d



_VLA_OPS_FILES = {'chunk.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\n\nimport torch\n\nfrom .chunk_delta_h import chunk_gated_delta_rule_fwd_h\nfrom .chunk_o import chunk_fwd_o\nfrom .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd\nfrom .cumsum import chunk_local_cumsum\nfrom .l2norm import l2norm_fwd\nfrom .solve_tril import solve_tril\nfrom .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard\nfrom .wy_fast import recompute_w_u_fwd\n\n\ndef chunk_gated_delta_rule_fwd(\n    q: torch.Tensor,\n    k: torch.Tensor,\n    v: torch.Tensor,\n    g: torch.Tensor,\n    beta: torch.Tensor,\n    scale: float,\n    initial_state: torch.Tensor,\n    output_final_state: bool,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    chunk_offsets: torch.Tensor | None = None,\n    core_attn_out: torch.Tensor | None = None,\n):\n    g = chunk_local_cumsum(\n        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices\n    )\n    # obtain WY representation. u is actually the new v.\n    A = chunk_scaled_dot_kkt_fwd(\n        k=k,\n        beta=beta,\n        g=g,\n        cu_seqlens=cu_seqlens,\n        chunk_indices=chunk_indices,\n        output_dtype=torch.float32,\n    )\n    A = solve_tril(\n        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype\n    )\n    w, u = recompute_w_u_fwd(\n        k=k,\n        v=v,\n        beta=beta,\n        A=A,\n        g_cumsum=g,\n        cu_seqlens=cu_seqlens,\n        chunk_indices=chunk_indices,\n    )\n    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(\n        k=k,\n        w=w,\n        u=u,\n        g=g,\n        initial_state=initial_state,\n        output_final_state=output_final_state,\n        cu_seqlens=cu_seqlens,\n        chunk_indices=chunk_indices,\n        chunk_offsets=chunk_offsets,\n    )\n    o = chunk_fwd_o(\n        q=q,\n        k=k,\n        v=v_new,\n        h=h,\n        g=g,\n        scale=scale,\n        cu_seqlens=cu_seqlens,\n        chunk_indices=chunk_indices,\n        core_attn_out=core_attn_out,\n    )\n    if SUPPRESS_LEVEL < 3:\n        return g, o, A, final_state, None, None, None\n    elif SUPPRESS_LEVEL >= 3:\n        return g, o, A, final_state, w, h, v_new\n\n\nclass ChunkGatedDeltaRuleFunction(torch.autograd.Function):\n    @staticmethod\n    @input_guard\n    @torch.amp.custom_fwd(device_type="cuda")\n    def forward(\n        ctx,\n        q: torch.Tensor,\n        k: torch.Tensor,\n        v: torch.Tensor,\n        g: torch.Tensor,\n        beta: torch.Tensor,\n        scale: float,\n        initial_state: torch.Tensor,\n        output_final_state: bool,\n        cu_seqlens: torch.Tensor | None = None,\n        chunk_indices: torch.Tensor | None = None,\n        chunk_offsets: torch.Tensor | None = None,\n        use_qk_l2norm_in_kernel: bool = False,\n        core_attn_out: torch.Tensor | None = None,\n    ):\n        if use_qk_l2norm_in_kernel:\n            q = l2norm_fwd(q)\n            k = l2norm_fwd(k)\n\n        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(\n            q=q,\n            k=k,\n            v=v,\n            g=g,\n            beta=beta,\n            scale=scale,\n            initial_state=initial_state,\n            output_final_state=output_final_state,\n            cu_seqlens=cu_seqlens,\n            chunk_indices=chunk_indices,\n            chunk_offsets=chunk_offsets,\n            core_attn_out=core_attn_out,\n        )\n        ctx.scale = scale\n        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel\n        if core_attn_out is not None:\n            assert not torch.is_grad_enabled(), (\n                "core_attn_out buffer reuse is only supported for inference"\n            )\n            assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"\n        return o.to(q.dtype), final_state\n\n\n@torch.compiler.disable\ndef chunk_gated_delta_rule(\n    q: torch.Tensor,\n    k: torch.Tensor,\n    v: torch.Tensor,\n    g: torch.Tensor,\n    beta: torch.Tensor,\n    scale: float = None,\n    initial_state: torch.Tensor = None,\n    output_final_state: bool = False,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    chunk_offsets: torch.Tensor | None = None,\n    use_qk_l2norm_in_kernel: bool = False,\n    core_attn_out: torch.Tensor | None = None,\n):\n    r"""\n    Args:\n        q (torch.Tensor):\n            Queries of shape `[B, T, H, K]`.\n        k (torch.Tensor):\n            Keys of shape `[B, T, H, K]`.\n        v (torch.Tensor):\n            Values of shape `[B, T, H, V]`.\n        g (torch.Tensor):\n            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.\n        beta (torch.Tensor):\n            Betas of shape `[B, T, H]`.\n        scale (Optional[int]):\n            Scale factor for the RetNet attention scores.\n            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.\n        initial_state (Optional[torch.Tensor]):\n            Initial state of shape `[N, H, V, K]` for `N` input sequences.\n            For equal-length input sequences, `N` equals the batch size `B`.\n            Default: `None`.\n        output_final_state (Optional[bool]):\n            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.\n        cu_seqlens (torch.Tensor):\n            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,\n            consistent with the FlashAttention API.\n    Returns:\n        o (torch.Tensor):\n            Outputs of shape `[B, T, H, V]`.\n        final_state (torch.Tensor):\n            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.\n\n    Examples::\n        >>> import torch\n        >>> import torch.nn.functional as F\n        >>> from einops import rearrange\n        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule\n        # inputs with equal lengths\n        >>> B, T, H, K, V = 4, 2048, 4, 512, 512\n        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=\'cuda\')\n        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device=\'cuda\'), p=2, dim=-1)\n        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=\'cuda\')\n        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device=\'cuda\').sigmoid()\n        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device=\'cuda\'))\n        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device=\'cuda\')\n        >>> o, ht = chunk_gated_delta_rule(\n            q, k, v, g, beta,\n            initial_state=h0,\n            output_final_state=True\n        )\n        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required\n        >>> q, k, v, beta, g = map(lambda x: rearrange(x, \'b t ... -> 1 (b t) ...\'), (q, k, v, beta, g))\n        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected\n        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)\n        >>> o_var, ht_var = chunk_gated_delta_rule(\n            q, k, v, g, beta,\n            initial_state=h0,\n            output_final_state=True,\n            cu_seqlens=cu_seqlens\n        )\n    """\n    assert q.dtype == k.dtype == v.dtype\n    assert q.dtype != torch.float32, (\n        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."\n    )\n    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."\n    if cu_seqlens is not None:\n        if q.shape[0] != 1:\n            raise ValueError(\n                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."\n                f"Please flatten variable-length inputs before processing."\n            )\n        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:\n            raise ValueError(\n                f"The number of initial states is expected to be equal to the number of input sequences, "\n                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."\n            )\n    if scale is None:\n        scale = k.shape[-1] ** -0.5\n    o, final_state = ChunkGatedDeltaRuleFunction.apply(\n        q,\n        k,\n        v,\n        g,\n        beta,\n        scale,\n        initial_state,\n        output_final_state,\n        cu_seqlens,\n        chunk_indices,\n        chunk_offsets,\n        use_qk_l2norm_in_kernel,\n        core_attn_out,\n    )\n    return o, final_state\n', 'chunk_delta_h.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\n\nimport torch\n\nimport triton\nimport triton.language as tl\n\nfrom .index import prepare_chunk_indices, prepare_chunk_offsets\nfrom .op import exp, exp2\nfrom .utils import FLA_CHUNK_SIZE, use_cuda_graph\n\nNUM_WARPS = [2, 4, 8, 16]\n# Triton\'s AMD backend fails to lower this kernel with num_stages=4.\n_CHUNK_DELTA_H_NUM_STAGES = [2, 3] if torch.version.hip else [2, 3, 4]\n\n\n@triton.heuristics(\n    {\n        "USE_G": lambda args: args["g"] is not None,\n        "USE_GK": lambda args: args["gk"] is not None,\n        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,\n        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,\n        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,\n        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,\n    }\n)\n@triton.autotune(\n    configs=[\n        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)\n        for num_warps in [2, 4]\n        for num_stages in _CHUNK_DELTA_H_NUM_STAGES\n        for BV in [32, 64]\n    ],\n    key=["H", "K", "V", "BT"],\n    use_cuda_graph=use_cuda_graph,\n)\n@triton.jit(do_not_specialize=["T"])\ndef chunk_gated_delta_rule_fwd_kernel_h_blockdim64(\n    k,\n    v,\n    w,\n    v_new,\n    g,\n    gk,\n    h,\n    h0,\n    ht,\n    cu_seqlens,\n    chunk_offsets,\n    T,\n    H: tl.constexpr,\n    Hg: tl.constexpr,\n    K: tl.constexpr,\n    V: tl.constexpr,\n    BT: tl.constexpr,\n    BV: tl.constexpr,\n    USE_G: tl.constexpr,\n    USE_GK: tl.constexpr,\n    USE_INITIAL_STATE: tl.constexpr,\n    STORE_FINAL_STATE: tl.constexpr,\n    SAVE_NEW_VALUE: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n    USE_EXP2: tl.constexpr,\n):\n    i_v, i_nh = tl.program_id(0), tl.program_id(1)\n    i_n, i_h = i_nh // H, i_nh % H\n    if IS_VARLEN:\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n        NT = tl.cdiv(T, BT)\n        boh = tl.load(chunk_offsets + i_n).to(tl.int32)\n    else:\n        bos, eos = i_n * T, i_n * T + T\n        NT = tl.cdiv(T, BT)\n        boh = i_n * NT\n\n    # [BV, BK]\n    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)\n    if K > 64:\n        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)\n    if K > 128:\n        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)\n    if K > 192:\n        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)\n\n    # calculate offset\n    h += ((boh * H + i_h) * V * K).to(tl.int64)\n    v += ((bos * H + i_h) * V).to(tl.int64)\n    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)\n    w += ((bos * H + i_h) * K).to(tl.int64)\n    if SAVE_NEW_VALUE:\n        v_new += ((bos * H + i_h) * V).to(tl.int64)\n    stride_v = H * V\n    stride_h = H * V * K\n    stride_k = Hg * K\n    stride_w = H * K\n    if USE_INITIAL_STATE:\n        h0 = h0 + i_nh * V * K\n    if STORE_FINAL_STATE:\n        ht = ht + i_nh * V * K\n\n    # load initial state\n    if USE_INITIAL_STATE:\n        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))\n        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)\n        if K > 64:\n            p_h0_2 = tl.make_block_ptr(\n                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)\n            )\n            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)\n        if K > 128:\n            p_h0_3 = tl.make_block_ptr(\n                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)\n            )\n            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)\n        if K > 192:\n            p_h0_4 = tl.make_block_ptr(\n                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)\n            )\n            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)\n\n    # main recurrence\n    for i_t in range(NT):\n        p_h1 = tl.make_block_ptr(\n            h + i_t.to(tl.int64) * stride_h,\n            (V, K),\n            (K, 1),\n            (i_v * BV, 0),\n            (BV, 64),\n            (1, 0),\n        )\n        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))\n        if K > 64:\n            p_h2 = tl.make_block_ptr(\n                h + i_t.to(tl.int64) * stride_h,\n                (V, K),\n                (K, 1),\n                (i_v * BV, 64),\n                (BV, 64),\n                (1, 0),\n            )\n            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))\n        if K > 128:\n            p_h3 = tl.make_block_ptr(\n                h + i_t.to(tl.int64) * stride_h,\n                (V, K),\n                (K, 1),\n                (i_v * BV, 128),\n                (BV, 64),\n                (1, 0),\n            )\n            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))\n        if K > 192:\n            p_h4 = tl.make_block_ptr(\n                h + i_t.to(tl.int64) * stride_h,\n                (V, K),\n                (K, 1),\n                (i_v * BV, 192),\n                (BV, 64),\n                (1, 0),\n            )\n            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))\n\n        p_w = tl.make_block_ptr(\n            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)\n        )\n        b_w = tl.load(p_w, boundary_check=(0, 1))\n        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))\n        if K > 64:\n            p_w = tl.make_block_ptr(\n                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)\n            )\n            b_w = tl.load(p_w, boundary_check=(0, 1))\n            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))\n        if K > 128:\n            p_w = tl.make_block_ptr(\n                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)\n            )\n            b_w = tl.load(p_w, boundary_check=(0, 1))\n            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))\n        if K > 192:\n            p_w = tl.make_block_ptr(\n                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)\n            )\n            b_w = tl.load(p_w, boundary_check=(0, 1))\n            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))\n        p_v = tl.make_block_ptr(\n            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)\n        )\n        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v\n\n        if SAVE_NEW_VALUE:\n            p_v = tl.make_block_ptr(\n                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)\n            )\n            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))\n\n        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1\n        if USE_G:\n            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T\n            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)\n            p_g = tl.make_block_ptr(\n                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)\n            )\n            b_g = tl.load(p_g, boundary_check=(0,))\n            if USE_EXP2:\n                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]\n                b_g_last = exp2(b_g_last)\n            else:\n                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]\n                b_g_last = exp(b_g_last)\n            b_h1 *= b_g_last\n            if K > 64:\n                b_h2 *= b_g_last\n            if K > 128:\n                b_h3 *= b_g_last\n            if K > 192:\n                b_h4 *= b_g_last\n\n        if USE_GK:\n            o_k1 = tl.arange(0, 64)\n            b_gk_last1 = tl.load(\n                gk + (bos + last_idx) * H * K + i_h * K + o_k1,\n                mask=(o_k1 < K),\n                other=0.0,\n            )\n            if USE_EXP2:\n                b_h1 *= exp2(b_gk_last1)[None, :]\n            else:\n                b_h1 *= exp(b_gk_last1)[None, :]\n            if K > 64:\n                o_k2 = 64 + o_k1\n                b_gk_last2 = tl.load(\n                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,\n                    mask=(o_k2 < K),\n                    other=0.0,\n                )\n                if USE_EXP2:\n                    b_h2 *= exp2(b_gk_last2)[None, :]\n                else:\n                    b_h2 *= exp(b_gk_last2)[None, :]\n            if K > 128:\n                o_k3 = 128 + o_k1\n                b_gk_last3 = tl.load(\n                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,\n                    mask=(o_k3 < K),\n                    other=0.0,\n                )\n                if USE_EXP2:\n                    b_h3 *= exp2(b_gk_last3)[None, :]\n                else:\n                    b_h3 *= exp(b_gk_last3)[None, :]\n            if K > 192:\n                o_k4 = 192 + o_k1\n                b_gk_last4 = tl.load(\n                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,\n                    mask=(o_k4 < K),\n                    other=0.0,\n                )\n                if USE_EXP2:\n                    b_h4 *= exp2(b_gk_last4)[None, :]\n                else:\n                    b_h4 *= exp(b_gk_last4)[None, :]\n        b_v = b_v.to(k.dtype.element_ty)\n\n        p_k = tl.make_block_ptr(\n            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)\n        )\n        b_k = tl.load(p_k, boundary_check=(0, 1))\n        b_h1 += tl.trans(tl.dot(b_k, b_v))\n        if K > 64:\n            p_k = tl.make_block_ptr(\n                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)\n            )\n            b_k = tl.load(p_k, boundary_check=(0, 1))\n            b_h2 += tl.trans(tl.dot(b_k, b_v))\n        if K > 128:\n            p_k = tl.make_block_ptr(\n                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)\n            )\n            b_k = tl.load(p_k, boundary_check=(0, 1))\n            b_h3 += tl.trans(tl.dot(b_k, b_v))\n        if K > 192:\n            p_k = tl.make_block_ptr(\n                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)\n            )\n            b_k = tl.load(p_k, boundary_check=(0, 1))\n            b_h4 += tl.trans(tl.dot(b_k, b_v))\n    # epilogue\n    if STORE_FINAL_STATE:\n        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))\n        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))\n        if K > 64:\n            p_ht = tl.make_block_ptr(\n                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)\n            )\n            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))\n        if K > 128:\n            p_ht = tl.make_block_ptr(\n                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)\n            )\n            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))\n        if K > 192:\n            p_ht = tl.make_block_ptr(\n                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)\n            )\n            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))\n\n\ndef chunk_gated_delta_rule_fwd_h(\n    k: torch.Tensor,\n    w: torch.Tensor,\n    u: torch.Tensor,\n    g: torch.Tensor | None = None,\n    gk: torch.Tensor | None = None,\n    initial_state: torch.Tensor | None = None,\n    output_final_state: bool = False,\n    chunk_size: int = FLA_CHUNK_SIZE,\n    save_new_value: bool = True,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    chunk_offsets: torch.Tensor | None = None,\n    use_exp2: bool = False,\n) -> tuple[torch.Tensor, torch.Tensor]:\n    # This kernel is slightly different from fla to support Q/K with different head numbers.\n    # In fla, Q/K always have the same head number, so Hg is always equal to H.\n    B, T, Hg, K, V = *k.shape, u.shape[-1]\n    H = u.shape[-2]\n    BT = chunk_size\n\n    if chunk_indices is None and cu_seqlens is not None:\n        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)\n    # N: the actual number of sequences in the batch with either equal or variable lengths\n    if cu_seqlens is None:\n        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None\n    else:\n        N, NT = len(cu_seqlens) - 1, len(chunk_indices)\n        if chunk_offsets is None:\n            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)\n    assert K <= 256, "current kernel does not support head dimension larger than 256."\n\n    h = k.new_empty(B, NT, H, V, K)\n    final_state = (\n        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None\n    )\n\n    v_new = torch.empty_like(u) if save_new_value else None\n\n    def grid(meta):\n        return (triton.cdiv(V, meta["BV"]), N * H)\n\n    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](\n        k=k,\n        v=u,\n        w=w,\n        v_new=v_new,\n        g=g,\n        gk=gk,\n        h=h,\n        h0=initial_state,\n        ht=final_state,\n        cu_seqlens=cu_seqlens,\n        chunk_offsets=chunk_offsets,\n        T=T,\n        H=H,\n        Hg=Hg,\n        K=K,\n        V=V,\n        BT=BT,\n        USE_EXP2=use_exp2,\n    )\n    return h, v_new, final_state\n', 'chunk_o.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n\n# ruff: noqa: E501\n\n\nimport torch\n\nimport triton\nimport triton.language as tl\n\nfrom .index import prepare_chunk_indices\nfrom .op import exp\nfrom .utils import FLA_CHUNK_SIZE, check_shared_mem, is_nvidia_hopper\n\nBKV_LIST = [64, 128] if check_shared_mem() else [32, 64]\nNUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]\n\n\n@triton.heuristics(\n    {\n        "USE_G": lambda args: args["g"] is not None,\n        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,\n    }\n)\n@triton.autotune(\n    configs=[\n        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)\n        for BK in BKV_LIST\n        for BV in BKV_LIST\n        for num_warps in NUM_WARPS\n        for num_stages in [2, 3, 4]\n    ],\n    key=["H", "K", "V", "BT"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef chunk_fwd_kernel_o(\n    q,\n    k,\n    v,\n    h,\n    g,\n    o,\n    cu_seqlens,\n    chunk_indices,\n    scale,\n    T,\n    H: tl.constexpr,\n    Hg: tl.constexpr,\n    K: tl.constexpr,\n    V: tl.constexpr,\n    BT: tl.constexpr,\n    BK: tl.constexpr,\n    BV: tl.constexpr,\n    USE_G: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n):\n    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n    i_b, i_h = i_bh // H, i_bh % H\n\n    if IS_VARLEN:\n        i_tg = i_t\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n        NT = tl.cdiv(T, BT)\n    else:\n        NT = tl.cdiv(T, BT)\n        i_tg = i_b * NT + i_t\n        bos, eos = i_b * T, i_b * T + T\n\n    # offset calculation\n    q += (bos * Hg + i_h // (H // Hg)) * K\n    k += (bos * Hg + i_h // (H // Hg)) * K\n    v += (bos * H + i_h) * V\n    o += (bos * H + i_h) * V\n    h += (i_tg * H + i_h).to(tl.int64) * V * K\n\n    b_o = tl.zeros([BT, BV], dtype=tl.float32)\n    b_A = tl.zeros([BT, BT], dtype=tl.float32)\n\n    for i_k in range(tl.cdiv(K, BK)):\n        p_q = tl.make_block_ptr(\n            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)\n        )\n        p_k = tl.make_block_ptr(\n            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)\n        )\n        p_h = tl.make_block_ptr(\n            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)\n        )\n        # [BT, BK]\n        b_q = tl.load(p_q, boundary_check=(0, 1))\n        # [BK, BT]\n        b_k = tl.load(p_k, boundary_check=(0, 1))\n        # [BV, BK]\n        b_h = tl.load(p_h, boundary_check=(0, 1))\n\n        # [BT, BK] @ [BK, BV] -> [BT, BV]\n        b_o += tl.dot(b_q, tl.trans(b_h))\n        # [BT, BK] @ [BK, BT] -> [BT, BT]\n        b_A += tl.dot(b_q, b_k)\n\n    if USE_G:\n        g += bos * H + i_h\n        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))\n        b_g = tl.load(p_g, boundary_check=(0,))\n        b_o = b_o * exp(b_g)[:, None]\n        b_A = b_A * exp(b_g[:, None] - b_g[None, :])\n\n    o_t = i_t * BT + tl.arange(0, BT)\n    m_t = o_t < T\n    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)\n    b_A = tl.where(m_A, b_A, 0)\n\n    p_v = tl.make_block_ptr(\n        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)\n    )\n    p_o = tl.make_block_ptr(\n        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)\n    )\n    b_v = tl.load(p_v, boundary_check=(0, 1))\n\n    # to fix mma -> mma layout conversion\n    # already solved by triton v3.2 or higher\n    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale\n    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))\n\n\ndef chunk_fwd_o(\n    q: torch.Tensor,\n    k: torch.Tensor,\n    v: torch.Tensor,\n    h: torch.Tensor,\n    g: torch.Tensor | None = None,  # cumsum of log decay\n    scale: float | None = None,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    chunk_size: int = FLA_CHUNK_SIZE,\n    core_attn_out: torch.Tensor | None = None,\n) -> torch.Tensor:\n    B, T, Hg, K, V = *q.shape, v.shape[-1]\n    H = v.shape[-2]\n    BT = chunk_size\n    if chunk_indices is None and cu_seqlens is not None:\n        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)\n    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)\n    if scale is None:\n        scale = k.shape[-1] ** -0.5\n\n    if core_attn_out is not None:\n        assert core_attn_out.numel() >= v.numel(), (\n            f"core_attn_out too small: {core_attn_out.numel()} < {v.numel()}"\n        )\n        o = core_attn_out[: v.numel()].view(*v.shape)\n    else:\n        o = torch.empty_like(v)\n\n    def grid(meta):\n        return (triton.cdiv(V, meta["BV"]), NT, B * H)\n\n    chunk_fwd_kernel_o[grid](\n        q,\n        k,\n        v,\n        h,\n        g,\n        o,\n        cu_seqlens,\n        chunk_indices,\n        scale,\n        T=T,\n        H=H,\n        Hg=Hg,\n        K=K,\n        V=V,\n        BT=BT,\n    )\n    return o\n', 'chunk_scaled_dot_kkt.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\n\nimport torch\n\nclass _P:\n    @staticmethod\n    def is_rocm():\n        return False\n    @staticmethod\n    def is_cuda_alike():\n        return True\n    @staticmethod\n    def is_arch_support_pdl():\n        return False\ncurrent_platform = _P()\nimport triton\nimport triton.language as tl\n\nfrom .index import prepare_chunk_indices\nfrom .op import exp\nfrom .utils import FLA_CHUNK_SIZE\n\n# On RDNA (gfx11xx/gfx12xx) WMMA only\n# accepts 16-bit/int inputs, so a widened (e.g. fp32) tl.dot is lowered to a\n# software matmul (~190x amdgcn-stage blowup). There we cast both operands down\n# to k\'s native storage dtype (bf16/fp16) so fast WMMA is used instead.\n_CAST_DOT_TO_K_DTYPE = False\n_CAST_DOT_TO_K_DTYPE = False\n\n\n@triton.heuristics(\n    {\n        "USE_G": lambda args: args["g"] is not None,\n        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,\n    }\n)\n@triton.autotune(\n    configs=[\n        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)\n        for BK in [32, 64, 128]\n        for num_warps in [2, 4, 8]\n        for num_stages in [2, 3, 4]\n    ],\n    key=["H", "K", "BT", "IS_VARLEN"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef chunk_scaled_dot_kkt_fwd_kernel(\n    k,\n    beta,\n    g,\n    A,\n    cu_seqlens,\n    chunk_indices,\n    T,\n    H: tl.constexpr,\n    Hg: tl.constexpr,\n    K: tl.constexpr,\n    BT: tl.constexpr,\n    BK: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n    USE_G: tl.constexpr,\n    CAST_DOT_TO_K_DTYPE: tl.constexpr,\n):\n    i_t, i_bh = tl.program_id(0), tl.program_id(1)\n    i_b, i_h = i_bh // H, i_bh % H\n    if IS_VARLEN:\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n    else:\n        bos, eos = i_b * T, i_b * T + T\n    o_t = i_t * BT + tl.arange(0, BT)\n    m_t = o_t < T\n\n    p_beta = tl.make_block_ptr(\n        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)\n    )\n    b_beta = tl.load(p_beta, boundary_check=(0,))\n\n    b_A = tl.zeros([BT, BT], dtype=tl.float32)\n    for i_k in range(tl.cdiv(K, BK)):\n        p_k = tl.make_block_ptr(\n            k + (bos * Hg + i_h // (H // Hg)) * K,\n            (T, K),\n            (Hg * K, 1),\n            (i_t * BT, i_k * BK),\n            (BT, BK),\n            (1, 0),\n        )\n        b_k = tl.load(p_k, boundary_check=(0, 1))\n        b_kb = b_k * b_beta[:, None]\n        if CAST_DOT_TO_K_DTYPE:\n            # RDNA: force operands to k\'s native dtype so WMMA is used.\n            b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))\n        else:\n            # Keep the promoted precision of the beta-scaled operand (WGMMA/MFMA).\n            b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))\n\n    if USE_G:\n        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))\n        b_g = tl.load(p_g, boundary_check=(0,))\n        b_g_diff = b_g[:, None] - b_g[None, :]\n        b_A = b_A * exp(b_g_diff)\n\n    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)\n    b_A = tl.where(m_A, b_A, 0)\n    p_A = tl.make_block_ptr(\n        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)\n    )\n    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))\n\n\ndef chunk_scaled_dot_kkt_fwd(\n    k: torch.Tensor,\n    g: torch.Tensor | None = None,\n    beta: torch.Tensor | None = None,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    chunk_size: int = FLA_CHUNK_SIZE,\n    output_dtype: torch.dtype = torch.float32,\n) -> torch.Tensor:\n    r"""\n    Compute beta * K * K^T.\n\n    Args:\n        k (torch.Tensor):\n            The key tensor of shape `[B, T, H, K]`.\n        beta (torch.Tensor):\n            The beta tensor of shape `[B, T, H]`.\n        g (torch.Tensor):\n            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.\n        cu_seqlens (torch.Tensor):\n            The cumulative sequence lengths of the input tensor.\n            Default: None\n        chunk_indices (torch.Tensor):\n            Pre-computed chunk indices. If None and cu_seqlens is provided,\n            computed internally. Default: None\n        chunk_size (int):\n            The chunk size. Default: 64.\n        output_dtype (torch.dtype):\n            The dtype of the output tensor. Default: `torch.float32`\n\n    Returns:\n        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.\n    """\n    # This kernel is slightly different from fla to support Q/K with different head numbers.\n    # In fla, Q/K always have the same head number, so Hg is always equal to H.\n    B, T, Hg, K = k.shape\n    H = beta.shape[-1]\n    BT = chunk_size\n    if chunk_indices is None and cu_seqlens is not None:\n        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)\n    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)\n\n    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)\n    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](\n        k=k,\n        g=g,\n        beta=beta,\n        A=A,\n        cu_seqlens=cu_seqlens,\n        chunk_indices=chunk_indices,\n        T=T,\n        H=H,\n        Hg=Hg,\n        K=K,\n        BT=BT,\n        CAST_DOT_TO_K_DTYPE=_CAST_DOT_TO_K_DTYPE,\n    )\n    return A\n', 'cumsum.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\n\nimport torch\n\nimport triton\nimport triton.language as tl\n\nfrom .index import prepare_chunk_indices\nfrom .utils import check_shared_mem, input_guard\n\nBS_LIST = [32, 64] if check_shared_mem() else [16, 32]\n\n\n@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})\n@triton.autotune(\n    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],\n    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef chunk_local_cumsum_scalar_kernel(\n    s,\n    o,\n    cu_seqlens,\n    chunk_indices,\n    T,\n    B: tl.constexpr,\n    H: tl.constexpr,\n    BT: tl.constexpr,\n    REVERSE: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n    HEAD_FIRST: tl.constexpr,\n):\n    i_t, i_bh = tl.program_id(0), tl.program_id(1)\n    i_b, i_h = i_bh // H, i_bh % H\n    if IS_VARLEN:\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n    else:\n        bos, eos = i_b * T, i_b * T + T\n\n    if HEAD_FIRST:\n        p_s = tl.make_block_ptr(\n            s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)\n        )\n        p_o = tl.make_block_ptr(\n            o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,)\n        )\n    else:\n        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))\n        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))\n    # [BT]\n    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)\n    b_o = tl.cumsum(b_s, axis=0)\n    if REVERSE:\n        b_z = tl.sum(b_s, axis=0)\n        b_o = -b_o + b_z[None] + b_s\n    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))\n\n\n@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})\n@triton.autotune(\n    configs=[\n        triton.Config({"BS": BS}, num_warps=num_warps)\n        for BS in BS_LIST\n        for num_warps in [2, 4, 8]\n    ],\n    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef chunk_local_cumsum_vector_kernel(\n    s,\n    o,\n    cu_seqlens,\n    chunk_indices,\n    T,\n    B: tl.constexpr,\n    H: tl.constexpr,\n    S: tl.constexpr,\n    BT: tl.constexpr,\n    BS: tl.constexpr,\n    REVERSE: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n    HEAD_FIRST: tl.constexpr,\n):\n    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)\n    i_b, i_h = i_bh // H, i_bh % H\n    if IS_VARLEN:\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n    else:\n        bos, eos = i_b * T, i_b * T + T\n\n    o_i = tl.arange(0, BT)\n    if REVERSE:\n        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)\n    else:\n        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)\n\n    if HEAD_FIRST:\n        p_s = tl.make_block_ptr(\n            s + (bos * H + i_h * T) * S,\n            (T, S),\n            (S, 1),\n            (i_t * BT, i_s * BS),\n            (BT, BS),\n            (1, 0),\n        )\n        p_o = tl.make_block_ptr(\n            o + (bos * H + i_h * T) * S,\n            (T, S),\n            (S, 1),\n            (i_t * BT, i_s * BS),\n            (BT, BS),\n            (1, 0),\n        )\n    else:\n        p_s = tl.make_block_ptr(\n            s + (bos * H + i_h) * S,\n            (T, S),\n            (H * S, 1),\n            (i_t * BT, i_s * BS),\n            (BT, BS),\n            (1, 0),\n        )\n        p_o = tl.make_block_ptr(\n            o + (bos * H + i_h) * S,\n            (T, S),\n            (H * S, 1),\n            (i_t * BT, i_s * BS),\n            (BT, BS),\n            (1, 0),\n        )\n    # [BT, BS]\n    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)\n    b_o = tl.dot(m_s, b_s, allow_tf32=False)\n    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))\n\n\ndef chunk_local_cumsum_scalar(\n    g: torch.Tensor,\n    chunk_size: int,\n    reverse: bool = False,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    head_first: bool = False,\n    output_dtype: torch.dtype | None = torch.float,\n) -> torch.Tensor:\n    if head_first:\n        B, H, T = g.shape\n    else:\n        B, T, H = g.shape\n    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (\n        "chunk_size must be a power of 2"\n    )\n    if chunk_indices is None and cu_seqlens is not None:\n        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)\n    BT = chunk_size\n    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)\n    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)\n    grid = (NT, B * H)\n    chunk_local_cumsum_scalar_kernel[grid](\n        g_org,\n        g,\n        cu_seqlens,\n        chunk_indices,\n        T=T,\n        B=B,\n        H=H,\n        BT=BT,\n        HEAD_FIRST=head_first,\n        REVERSE=reverse,\n    )\n    return g\n\n\ndef chunk_local_cumsum_vector(\n    g: torch.Tensor,\n    chunk_size: int,\n    reverse: bool = False,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    head_first: bool = False,\n    output_dtype: torch.dtype | None = torch.float,\n) -> torch.Tensor:\n    if head_first:\n        B, H, T, S = g.shape\n    else:\n        B, T, H, S = g.shape\n    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (\n        "chunk_size must be a power of 2"\n    )\n    if chunk_indices is None and cu_seqlens is not None:\n        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)\n    BT = chunk_size\n    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)\n\n    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)\n\n    def grid(meta):\n        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)\n\n    # keep cumulative normalizer in fp32\n    # this kernel is equivalent to\n    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)\n    chunk_local_cumsum_vector_kernel[grid](\n        g_org,\n        g,\n        cu_seqlens,\n        chunk_indices,\n        T=T,\n        B=B,\n        H=H,\n        S=S,\n        BT=BT,\n        HEAD_FIRST=head_first,\n        REVERSE=reverse,\n    )\n    return g\n\n\n@input_guard\ndef chunk_local_cumsum(\n    g: torch.Tensor,\n    chunk_size: int,\n    reverse: bool = False,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    head_first: bool = False,\n    output_dtype: torch.dtype | None = torch.float,\n    **kwargs,\n) -> torch.Tensor:\n    if cu_seqlens is not None:\n        assert g.shape[0] == 1, (\n            "Only batch size 1 is supported when cu_seqlens are provided"\n        )\n    if len(g.shape) == 3:\n        return chunk_local_cumsum_scalar(\n            g,\n            chunk_size,\n            reverse,\n            cu_seqlens,\n            chunk_indices,\n            head_first,\n            output_dtype,\n        )\n    elif len(g.shape) == 4:\n        return chunk_local_cumsum_vector(\n            g,\n            chunk_size,\n            reverse,\n            cu_seqlens,\n            chunk_indices,\n            head_first,\n            output_dtype,\n        )\n    else:\n        raise ValueError(\n            f"Unsupported input shape {g.shape}. "\n            f"which should be (B, T, H, D) if `head_first=False` "\n            f"or (B, H, T, D) otherwise"\n        )\n', 'fused_norm_gate.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source was licensed under the MIT license.\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\n\nimport torch\nimport torch.nn as nn\n\nclass CustomOp:\n    @staticmethod\n    def register(name):\n        def deco(cls):\n            return cls\n        return deco\nclass _P:\n    @staticmethod\n    def is_rocm():\n        return False\n    @staticmethod\n    def is_cuda_alike():\n        return True\n    @staticmethod\n    def is_arch_support_pdl():\n        return False\ncurrent_platform = _P()\nimport triton\nimport triton.language as tl\ndef cdiv(a, b):\n    return (a + b - 1) // b\n\ndef next_power_of_2(n):\n    return 1 << (n - 1).bit_length()\n\n@triton.heuristics(\n    {\n        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,\n        "HAS_RESIDUAL": lambda args: args["residual"] is not None,\n        "HAS_WEIGHT": lambda args: args["w"] is not None,\n        "HAS_BIAS": lambda args: args["b"] is not None,\n    }\n)\n@triton.jit\ndef layer_norm_gated_fwd_kernel(\n    x,  # pointer to the input\n    g,  # pointer to the gate\n    y,  # pointer to the output\n    w,  # pointer to the weights\n    b,  # pointer to the biases\n    residual,  # pointer to the residual\n    residual_out,  # pointer to the residual\n    mean,  # pointer to the mean\n    rstd,  # pointer to the 1/std\n    eps,  # epsilon to avoid division by zero\n    T,  # number of rows in x\n    H: tl.constexpr,  # number of heads\n    g_stride_n,\n    D: tl.constexpr,  # number of columns in x\n    BT: tl.constexpr,\n    BD: tl.constexpr,\n    ACTIVATION: tl.constexpr,\n    IS_RMS_NORM: tl.constexpr,\n    STORE_RESIDUAL_OUT: tl.constexpr,\n    HAS_RESIDUAL: tl.constexpr,\n    HAS_WEIGHT: tl.constexpr,\n    HAS_BIAS: tl.constexpr,\n    launch_pdl: tl.constexpr,\n):\n    i_t = tl.program_id(0)\n\n    o_d = tl.arange(0, BD)\n    m_d = o_d < D\n\n    if launch_pdl:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n\n    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))\n    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)\n    if HAS_RESIDUAL:\n        p_res = tl.make_block_ptr(\n            residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)\n        )\n        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)\n    if STORE_RESIDUAL_OUT:\n        p_res_out = tl.make_block_ptr(\n            residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0)\n        )\n        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))\n    if not IS_RMS_NORM:\n        b_mean = tl.sum(b_x, axis=1) / D\n        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))\n        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))\n        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)\n        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D\n    else:\n        b_xbar = tl.where(m_d[None, :], b_x, 0.0)\n        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D\n    b_rstd = 1 / tl.sqrt(b_var + eps)\n\n    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))\n    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))\n\n    if HAS_WEIGHT:\n        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)\n    if HAS_BIAS:\n        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)\n    b_x_hat = (\n        (b_x - b_mean[:, None]) * b_rstd[:, None]\n        if not IS_RMS_NORM\n        else b_x * b_rstd[:, None]\n    )\n    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat\n    if HAS_BIAS:\n        b_y = b_y + b_b[None, :]\n\n    # swish/sigmoid output gate\n    o_t = i_t * BT + tl.arange(0, BT)\n    o_g = (o_t // H) * g_stride_n + (o_t % H) * D\n    b_g = tl.load(\n        g + o_g[:, None] + o_d[None, :],\n        mask=(o_t[:, None] < T) & m_d[None, :],\n        other=0.0,\n    ).to(tl.float32)\n    if ACTIVATION == "swish" or ACTIVATION == "silu":\n        b_y = b_y * b_g * tl.sigmoid(b_g)\n    elif ACTIVATION == "sigmoid":\n        b_y = b_y * tl.sigmoid(b_g)\n\n    # Write output\n    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))\n    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))\n\n\n@triton.heuristics(\n    {\n        "STORE_RESIDUAL_OUT": lambda args: args["residual_out"] is not None,\n        "HAS_RESIDUAL": lambda args: args["residual"] is not None,\n        "HAS_WEIGHT": lambda args: args["w"] is not None,\n        "HAS_BIAS": lambda args: args["b"] is not None,\n    }\n)\n@triton.jit\ndef layer_norm_gated_fwd_kernel1(\n    x,  # pointer to the input\n    g,  # pointer to the gate\n    y,  # pointer to the output\n    w,  # pointer to the weights\n    b,  # pointer to the biases\n    residual,  # pointer to the residual\n    residual_out,  # pointer to the residual\n    mean,  # pointer to the mean\n    rstd,  # pointer to the 1/std\n    eps,  # epsilon to avoid division by zero\n    D: tl.constexpr,  # number of columns in x\n    BD: tl.constexpr,\n    ACTIVATION: tl.constexpr,\n    IS_RMS_NORM: tl.constexpr,\n    STORE_RESIDUAL_OUT: tl.constexpr,\n    HAS_RESIDUAL: tl.constexpr,\n    HAS_WEIGHT: tl.constexpr,\n    HAS_BIAS: tl.constexpr,\n    launch_pdl: tl.constexpr,\n):\n    i_t = tl.program_id(0)\n    x += i_t * D\n    y += i_t * D\n    g += i_t * D\n    if HAS_RESIDUAL:\n        residual += i_t * D\n    if STORE_RESIDUAL_OUT:\n        residual_out += i_t * D\n\n    if launch_pdl:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n\n    o_d = tl.arange(0, BD)\n    m_d = o_d < D\n    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)\n    if HAS_RESIDUAL:\n        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)\n    if STORE_RESIDUAL_OUT:\n        tl.store(residual_out + o_d, b_x, mask=m_d)\n    if not IS_RMS_NORM:\n        b_mean = tl.sum(b_x, axis=0) / D\n        tl.store(mean + i_t, b_mean)\n        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)\n        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D\n    else:\n        b_xbar = tl.where(m_d, b_x, 0.0)\n        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D\n    b_rstd = 1 / tl.sqrt(b_var + eps)\n    tl.store(rstd + i_t, b_rstd)\n\n    if HAS_WEIGHT:\n        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)\n    if HAS_BIAS:\n        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)\n    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd\n    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat\n    if HAS_BIAS:\n        b_y = b_y + b_b\n\n    # swish/sigmoid output gate\n    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)\n    if ACTIVATION == "swish" or ACTIVATION == "silu":\n        b_y = b_y * b_g * tl.sigmoid(b_g)\n    elif ACTIVATION == "sigmoid":\n        b_y = b_y * tl.sigmoid(b_g)\n\n    # Write output\n    tl.store(y + o_d, b_y, mask=m_d)\n\n\ndef layer_norm_gated_fwd(\n    x: torch.Tensor,\n    g: torch.Tensor,\n    weight: torch.Tensor,\n    bias: torch.Tensor,\n    activation: str = "swish",\n    eps: float = 1e-5,\n    residual: torch.Tensor = None,\n    out_dtype: torch.dtype = None,\n    residual_dtype: torch.dtype = None,\n    is_rms_norm: bool = False,\n    H: int = 1,\n    g_stride_n: int | None = None,\n):\n    if residual is not None:\n        residual_dtype = residual.dtype\n    T, D = x.shape\n    if g_stride_n is None:\n        g_stride_n = D\n    assert T % H == 0\n    if residual is not None:\n        assert residual.shape == (T, D)\n    if weight is not None:\n        assert weight.shape == (D,)\n    if bias is not None:\n        assert bias.shape == (D,)\n    # allocate output\n    y = x if out_dtype is None else torch.empty_like(x, dtype=out_dtype)\n    if residual is not None or (\n        residual_dtype is not None and residual_dtype != x.dtype\n    ):\n        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)\n    else:\n        residual_out = None\n    mean = (\n        torch.empty((T,), dtype=torch.float, device=x.device)\n        if not is_rms_norm\n        else None\n    )\n    rstd = torch.empty((T,), dtype=torch.float, device=x.device)\n    # Less than 64KB per feature: enqueue fused kernel\n    MAX_FUSED_SIZE = 65536 // x.element_size()\n    BD = min(MAX_FUSED_SIZE, next_power_of_2(D))\n    if D > BD:\n        raise RuntimeError("This layer norm doesn\'t support feature dim >= 64KB.")\n    if D <= 512:\n        BT = 16\n        layer_norm_gated_fwd_kernel[(cdiv(T, BT),)](\n            x=x,\n            g=g,\n            y=y,\n            w=weight,\n            b=bias,\n            residual=residual,\n            residual_out=residual_out,\n            mean=mean,\n            rstd=rstd,\n            eps=eps,\n            T=T,\n            H=H,\n            g_stride_n=g_stride_n,\n            D=D,\n            BD=BD,\n            BT=BT,\n            ACTIVATION=activation,\n            IS_RMS_NORM=is_rms_norm,\n            num_warps=8,\n            launch_pdl=current_platform.is_arch_support_pdl(),\n        )\n    else:\n        layer_norm_gated_fwd_kernel1[(T,)](\n            x=x,\n            g=g,\n            y=y,\n            w=weight,\n            b=bias,\n            residual=residual,\n            residual_out=residual_out,\n            mean=mean,\n            rstd=rstd,\n            eps=eps,\n            D=D,\n            BD=BD,\n            ACTIVATION=activation,\n            IS_RMS_NORM=is_rms_norm,\n            num_warps=4,\n            launch_pdl=current_platform.is_arch_support_pdl(),\n        )\n    # residual_out is None if residual is None and residual_dtype == input_dtype\n    return y, mean, rstd, residual_out if residual_out is not None else x\n\n\ndef rms_norm_gated(\n    x: torch.Tensor,\n    g: torch.Tensor,\n    weight: torch.Tensor,\n    bias: torch.Tensor,\n    activation: str = "swish",\n    residual: torch.Tensor | None = None,\n    prenorm: bool = False,\n    residual_in_fp32: bool = False,\n    eps: float = 1e-6,\n):\n    x_shape_og = x.shape\n    # reshape input data into 2D tensor\n    x = x.contiguous().reshape(-1, x.shape[-1])\n    D = x.shape[-1]\n    # The tiled kernel supports row-strided gates; kernel1 does not.\n    if D <= 512:\n        H = 1 if g.ndim == 2 else g.shape[-2]\n        g = g.view(-1, H, D)\n        g_stride_n = g.stride(0)\n    else:\n        g = g.contiguous()\n        H = 1\n        g_stride_n = D\n    if residual is not None:\n        assert residual.shape == x_shape_og\n        residual = residual.contiguous().reshape(-1, residual.shape[-1])\n    residual_dtype = (\n        residual.dtype\n        if residual is not None\n        else (torch.float if residual_in_fp32 else None)\n    )\n    y, _, _, residual_out = layer_norm_gated_fwd(\n        x=x,\n        g=g,\n        weight=weight,\n        bias=bias,\n        activation=activation,\n        eps=eps,\n        residual=residual,\n        residual_dtype=residual_dtype,\n        is_rms_norm=True,\n        H=H,\n        g_stride_n=g_stride_n,\n    )\n    y = y.reshape(x_shape_og)\n    return y if not prenorm else (y, residual_out.reshape(x_shape_og))\n\n\n@CustomOp.register("fused_rms_norm_gated")\nclass FusedRMSNormGated(CustomOp):\n    def __init__(\n        self,\n        hidden_size: int,\n        elementwise_affine: bool = True,\n        eps: float = 1e-5,\n        activation: str = "swish",\n        device: torch.device | None = None,\n        dtype: torch.dtype | None = None,\n    ) -> None:\n        factory_kwargs = {"device": device, "dtype": dtype}\n        super().__init__()\n\n        self.hidden_size = hidden_size\n        self.elementwise_affine = elementwise_affine\n        self.eps = eps\n        self.activation = activation\n\n        if self.activation not in ["swish", "silu", "sigmoid"]:\n            raise ValueError(f"Unsupported activation: {self.activation}")\n\n        if elementwise_affine:\n            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))\n        else:\n            self.register_parameter("weight", None)\n        self.register_parameter("bias", None)\n\n    def forward_native(\n        self,\n        x: torch.Tensor,\n        g: torch.Tensor,\n        residual: torch.Tensor | None = None,\n        prenorm: bool = False,\n        residual_in_fp32: bool = False,\n    ) -> torch.Tensor:\n        """Decomposed PyTorch ops for torch.compile/inductor fusion."""\n        # TODO(https://github.com/vllm-project/vllm/issues/36175): implement\n        # native residual/prenorm path and unify with RMSNormGated.\n        # For now, fall back to the triton kernel.\n        if residual is not None or prenorm:\n            return self.forward_cuda(x, g, residual, prenorm, residual_in_fp32)\n        x_float = x.float()\n        variance = x_float.pow(2).mean(dim=-1, keepdim=True)\n        x_normed = x_float * torch.rsqrt(variance + self.eps)\n        if self.weight is not None:\n            x_normed = x_normed * self.weight.float()\n        g_float = g.float()\n        if self.activation in ("swish", "silu"):\n            out = x_normed * g_float * torch.sigmoid(g_float)\n        else:  # sigmoid\n            out = x_normed * torch.sigmoid(g_float)\n        return out.to(x.dtype)\n\n    def forward_cuda(\n        self,\n        x: torch.Tensor,\n        g: torch.Tensor,\n        residual: torch.Tensor | None = None,\n        prenorm: bool = False,\n        residual_in_fp32: bool = False,\n    ) -> torch.Tensor:\n        return rms_norm_gated(\n            x,\n            g,\n            self.weight,\n            self.bias,\n            self.activation,\n            residual=residual,\n            eps=self.eps,\n            prenorm=prenorm,\n            residual_in_fp32=residual_in_fp32,\n        )\n', 'index.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\nimport torch\n\ndef gpu_sync_allowed():\n    import contextlib\n    return contextlib.nullcontext()\nimport triton\n\nfrom .utils import tensor_cache\n\n\n@tensor_cache\ndef prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:\n    return cu_seqlens[1:] - cu_seqlens[:-1]\n\n\n@tensor_cache\ndef prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:\n    # This will be fixed by https://github.com/vllm-project/vllm/pull/51540.\n    with gpu_sync_allowed():\n        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()\n    indices = torch.cat([torch.arange(n) for n in chunk_counts])\n    chunk_indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)\n    return chunk_indices.to(\n        device=cu_seqlens.device, dtype=cu_seqlens.dtype, non_blocking=True\n    )\n\n\n@tensor_cache\ndef prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:\n    return torch.cat(\n        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]\n    ).cumsum(-1)\n', 'l2norm.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n\nimport os\n\nimport torch\n\nimport triton\nimport triton.language as tl\n\nBT_LIST = [8, 16, 32, 64, 128]\n\nUSE_DEFAULT_FLA_NORM = int(os.getenv("USE_DEFAULT_FLA_NORM", "0"))\n\n\n@triton.autotune(\n    configs=[\n        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]\n    ],\n    key=["D"],\n)\n@triton.jit\ndef l2norm_fwd_kernel1(\n    x,\n    y,\n    D,\n    BD: tl.constexpr,\n    eps,\n):\n    i_t = tl.program_id(0)\n    x += i_t * D\n    y += i_t * D\n    # Compute mean and variance\n    cols = tl.arange(0, BD)\n    mask = cols < D\n    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)\n    b_var = tl.sum(b_x * b_x, axis=0)\n    b_rstd = 1 / tl.sqrt(b_var + eps)\n    # tl.store(Rstd + i_t, rstd)\n    # Normalize and apply linear transformation\n    b_y = b_x * b_rstd\n    tl.store(y + cols, b_y, mask=mask)\n\n\n@triton.autotune(\n    configs=[\n        triton.Config({"BT": BT}, num_warps=num_warps)\n        for num_warps in [1, 2, 4, 8, 16]\n        for BT in BT_LIST\n    ],\n    key=["D"],\n)\n@triton.jit(do_not_specialize=["NB"])\ndef l2norm_fwd_kernel(\n    x,\n    y,\n    eps,\n    NB,\n    T,\n    D: tl.constexpr,\n    BT: tl.constexpr,\n    BD: tl.constexpr,\n):\n    i_t = tl.program_id(0)\n    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))\n    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)\n    b_var = tl.sum(b_x * b_x, axis=1)\n    b_y = b_x / tl.sqrt(b_var + eps)[:, None]\n    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))\n    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))\n\n\n@triton.jit\ndef l2norm_fwd_kernel2(\n    X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr\n):\n    xoffset = tl.program_id(0) * MBLOCK\n    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]\n    xmask = row_idx < M\n    rindex = tl.arange(0, BD)[None, :]\n    cmask = rindex < N\n    mask = xmask & cmask\n    xs = tl.load(X + (rindex + N * row_idx), mask, other=0.0).to(tl.float32)\n    square = tl.broadcast_to(xs * xs, [MBLOCK, BD])\n    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]\n    rsqrt = tl.rsqrt(square_sum + eps)\n    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)\n\n\ndef l2norm_fwd(\n    x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None\n):\n    x_shape_og = x.shape\n    x = x.view(-1, x.shape[-1])\n    # allocate output\n    if output_dtype is None:\n        y = torch.empty_like(x)\n    else:\n        y = torch.empty_like(x, dtype=output_dtype)\n    assert y.stride(-1) == 1\n    T, D = x.shape[0], x.shape[-1]\n    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)\n    # Less than 64KB per feature: enqueue fused kernel\n    MAX_FUSED_SIZE = 65536 // x.element_size()\n    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))\n    if D > BD:\n        raise RuntimeError("This layer doesn\'t support feature dim >= 64KB.")\n\n    if not USE_DEFAULT_FLA_NORM:\n        MBLOCK = 32\n        # M, N = x.shape\n        l2norm_fwd_kernel2[(triton.cdiv(T, MBLOCK),)](\n            x,\n            y,\n            eps,\n            T,\n            D,\n            BD,\n            MBLOCK,\n        )\n    else:\n        if D <= 512:\n            NB = triton.cdiv(T, 2048)\n\n            def grid(meta):\n                return (triton.cdiv(T, meta["BT"]),)\n\n            l2norm_fwd_kernel[grid](\n                x,\n                y,\n                eps,\n                NB=NB,\n                T=T,\n                D=D,\n                BD=BD,\n            )\n        else:\n            l2norm_fwd_kernel1[(T,)](\n                x,\n                y,\n                eps=eps,\n                D=D,\n                BD=BD,\n            )\n\n    return y.view(x_shape_og)\n', 'op.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n\nimport os\n\nimport triton\nimport triton.language as tl\ntry:\n    from triton.language.extra import libdevice as tldevice\nexcept Exception:\n    tldevice = tl\n\nfrom .utils import is_gather_supported\n\nif os.environ.get("FLA_USE_FAST_OPS", "0") == "1":\n    exp = tldevice.fast_expf\n    exp2 = tl.exp2\n    log = tldevice.fast_logf\n    log2 = tldevice.fast_log2f\nelse:\n    exp = tl.exp\n    exp2 = tl.exp2\n    log = tl.log\n    log2 = tl.log2\n\n\nif not is_gather_supported:\n\n    @triton.jit\n    def gather(src, index, axis, _builder=None):\n        """\n        Gather operation that works when tl.gather is not supported.\n        This is a fallback implementation that returns None.\n        Just to make triton compiler happy.\n        """\n        return None\nelse:\n    gather = tl.gather\n\nif hasattr(triton.language, "_experimental_make_tensor_descriptor"):\n    # For Triton 3.3.x\n    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor\nelif hasattr(triton.language, "make_tensor_descriptor"):\n    # For Triton 3.4.x and later\n    make_tensor_descriptor = triton.language.make_tensor_descriptor\nelse:\n    """\n    Fallback implementation when TMA is not supported.\n    Returns None to indicate TMA descriptors are unavailable.\n    Just make triton compiler happy.\n    """\n\n    @triton.jit\n    def make_tensor_descriptor(\n        base,\n        shape,\n        strides,\n        block_shape,\n        _builder=None,\n    ):\n        return None\n', 'solve_tril.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\n\nimport os\n\nimport torch\n\nimport triton\nimport triton.language as tl\n\nfrom .index import prepare_chunk_indices\nfrom .op import make_tensor_descriptor\nfrom .utils import input_guard, is_amd, is_tma_supported\n\nFLA_TRIL_PRECISION = os.environ.get("FLA_TRIL_PRECISION", "ieee")\nALLOWED_TRIL_PRECISIONS = ["ieee", "tf32"] if is_amd else ["ieee", "tf32", "tf32x3"]\nassert FLA_TRIL_PRECISION in ALLOWED_TRIL_PRECISIONS, (\n    f"FLA_TRIL_PRECISION must be one of {ALLOWED_TRIL_PRECISIONS}, but got {FLA_TRIL_PRECISION}"\n)\n\n\n@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})\n@triton.autotune(\n    configs=[\n        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n        for num_warps in [1, 2, 4, 8]\n        for num_stages in [2, 3, 4, 5]\n    ],\n    key=["BT"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef solve_tril_16x16_kernel(\n    A,\n    Ai,\n    cu_seqlens,\n    chunk_indices,\n    T,\n    H: tl.constexpr,\n    BT: tl.constexpr,\n    USE_TMA: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n    DOT_PRECISION: tl.constexpr,\n):\n    i_t, i_bh = tl.program_id(0), tl.program_id(1)\n    i_b, i_h = i_bh // H, i_bh % H\n    if IS_VARLEN:\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n    else:\n        bos, eos = i_b * T, i_b * T + T\n    o_i = tl.arange(0, 16)\n    m_A = o_i[:, None] > o_i[None, :]\n    m_I = o_i[:, None] == o_i[None, :]\n\n    A = A + (bos * H + i_h) * BT\n    Ai = Ai + (bos * H + i_h) * 16\n\n    offset = (i_t * 16) % BT\n    if not USE_TMA:\n        p_A = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * 16, offset), (16, 16), (1, 0)\n        )\n        # [16, 16]\n        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)\n    else:\n        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])\n        desc_o = make_tensor_descriptor(Ai, [T, 16], [H * 16, 1], [16, 16])\n        b_A = desc.load([i_t * 16, offset]).to(tl.float32)\n    b_A = -tl.where(m_A, b_A, 0)\n\n    for i in range(2, min(16, T - i_t * 16)):\n        # [16]\n        b_a = -tl.load(A + (i_t * 16 + i) * H * BT + o_i + offset)\n        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)\n        b_A = tl.where((o_i == i)[:, None], b_a, b_A)\n    b_A += m_I\n    if not USE_TMA:\n        p_Ai = tl.make_block_ptr(\n            Ai, (T, 16), (H * 16, 1), (i_t * 16, 0), (16, 16), (1, 0)\n        )\n        tl.store(\n            p_Ai,\n            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n    else:\n        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))\n\n\n@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})\n@triton.autotune(\n    configs=[\n        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n        for num_warps in [1, 2, 4, 8]\n        for num_stages in [2, 3, 4, 5]\n    ],\n    key=["H", "BT", "IS_VARLEN"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef merge_16x16_to_32x32_inverse_kernel(\n    A,\n    Ai,\n    cu_seqlens,\n    chunk_indices,\n    T,\n    H: tl.constexpr,\n    BT: tl.constexpr,\n    USE_TMA: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n    DOT_PRECISION: tl.constexpr,\n):\n    i_t, i_bh = tl.program_id(0), tl.program_id(1)\n    i_b, i_h = i_bh // H, i_bh % H\n    if IS_VARLEN:\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n    else:\n        bos, eos = i_b * T, i_b * T + T\n\n    o_i = tl.arange(0, 16)\n    m_A = o_i[:, None] > o_i[None, :]\n    m_I = o_i[:, None] == o_i[None, :]\n    A += (bos * H + i_h) * BT\n    Ai += (bos * H + i_h) * BT\n\n    if not USE_TMA:\n        p_A_11 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)\n        )\n        p_A_22 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)\n        )\n        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)\n        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)\n    else:\n        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])\n        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])\n        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)\n        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)\n\n    # [16, 16]\n    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)\n    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)\n\n    for i in range(2, min(16, T - i_t * BT)):\n        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)\n        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)\n        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)\n    for i in range(16 + 2, min(32, T - i_t * BT)):\n        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)\n        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)\n        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)\n\n    b_Ai_11 += m_I\n    b_Ai_22 += m_I\n\n    if not USE_TMA:\n        p_A_21 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)\n        )\n        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)\n    else:\n        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)\n\n    b_Ai_21 = -tl.dot(\n        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),\n        b_Ai_11,\n        input_precision=DOT_PRECISION,\n    )\n\n    if not USE_TMA:\n        p_Ai_11 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)\n        )\n        p_Ai_21 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)\n        )\n        p_Ai_22 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)\n        )\n        tl.store(\n            p_Ai_11,\n            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_22,\n            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_21,\n            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n    else:\n        desc_o.store(\n            [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n\n\n@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})\n@triton.autotune(\n    configs=[\n        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n        for num_warps in [2, 4, 8]\n        for num_stages in [2, 3, 4, 5]\n    ],\n    key=["H", "BT", "IS_VARLEN"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef merge_16x16_to_64x64_inverse_kernel(\n    A,\n    Ai,\n    cu_seqlens,\n    chunk_indices,\n    T,\n    H: tl.constexpr,\n    BT: tl.constexpr,\n    USE_TMA: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n    DOT_PRECISION: tl.constexpr,\n):\n    i_t, i_bh = tl.program_id(0), tl.program_id(1)\n    i_b, i_h = i_bh // H, i_bh % H\n    if IS_VARLEN:\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n    else:\n        bos, eos = i_b * T, i_b * T + T\n\n    o_i = tl.arange(0, 16)\n    m_A = o_i[:, None] > o_i[None, :]\n    m_I = o_i[:, None] == o_i[None, :]\n    A += (bos * H + i_h) * BT\n    Ai += (bos * H + i_h) * BT\n\n    if not USE_TMA:\n        p_A_11 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)\n        )\n        p_A_22 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)\n        )\n        p_A_33 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)\n        )\n        p_A_44 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)\n        )\n        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)\n        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)\n        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)\n        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)\n    else:\n        desc = make_tensor_descriptor(A, [T, BT], [H * BT, 1], [16, 16])\n        desc_o = make_tensor_descriptor(Ai, [T, BT], [H * BT, 1], [16, 16])\n        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)\n        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)\n        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)\n        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)\n\n    # [16, 16]\n    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)\n    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)\n    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)\n    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)\n\n    for i in range(2, min(16, T - i_t * BT)):\n        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)\n        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)\n        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)\n    for i in range(16 + 2, min(32, T - i_t * BT)):\n        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)\n        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)\n        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)\n    for i in range(32 + 2, min(48, T - i_t * BT)):\n        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)\n        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)\n        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)\n    for i in range(48 + 2, min(64, T - i_t * BT)):\n        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)\n        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)\n        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)\n    b_Ai_11 += m_I\n    b_Ai_22 += m_I\n    b_Ai_33 += m_I\n    b_Ai_44 += m_I\n\n    if not USE_TMA:\n        p_A_21 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)\n        )\n        p_A_31 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)\n        )\n        p_A_32 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)\n        )\n        p_A_41 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)\n        )\n        p_A_42 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)\n        )\n        p_A_43 = tl.make_block_ptr(\n            A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)\n        )\n        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)\n        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)\n        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)\n        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)\n        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)\n        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)\n    else:\n        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)\n        b_A_31 = desc.load([i_t * BT + 32, 0]).to(tl.float32)\n        b_A_32 = desc.load([i_t * BT + 32, 16]).to(tl.float32)\n        b_A_41 = desc.load([i_t * BT + 48, 0]).to(tl.float32)\n        b_A_42 = desc.load([i_t * BT + 48, 16]).to(tl.float32)\n        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)\n\n    b_Ai_21 = -tl.dot(\n        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),\n        b_Ai_11,\n        input_precision=DOT_PRECISION,\n    )\n    b_Ai_32 = -tl.dot(\n        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),\n        b_Ai_22,\n        input_precision=DOT_PRECISION,\n    )\n    b_Ai_43 = -tl.dot(\n        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),\n        b_Ai_33,\n        input_precision=DOT_PRECISION,\n    )\n\n    b_Ai_31 = -tl.dot(\n        b_Ai_33,\n        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)\n        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),\n        input_precision=DOT_PRECISION,\n    )\n    b_Ai_42 = -tl.dot(\n        b_Ai_44,\n        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)\n        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),\n        input_precision=DOT_PRECISION,\n    )\n    b_Ai_41 = -tl.dot(\n        b_Ai_44,\n        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)\n        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)\n        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),\n        input_precision=DOT_PRECISION,\n    )\n\n    if not USE_TMA:\n        p_Ai_11 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)\n        )\n        p_Ai_22 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)\n        )\n        p_Ai_33 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)\n        )\n        p_Ai_44 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)\n        )\n        p_Ai_21 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)\n        )\n        p_Ai_31 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)\n        )\n        p_Ai_32 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)\n        )\n        p_Ai_41 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)\n        )\n        p_Ai_42 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)\n        )\n        p_Ai_43 = tl.make_block_ptr(\n            Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)\n        )\n        tl.store(\n            p_Ai_11,\n            b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_22,\n            b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_33,\n            b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_44,\n            b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_21,\n            b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_31,\n            b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_32,\n            b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_41,\n            b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_42,\n            b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n        tl.store(\n            p_Ai_43,\n            b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),\n            boundary_check=(0, 1),\n        )\n    else:\n        desc_o.store(\n            [i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 32, 0], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 32, 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 48, 0], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 48, 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n        desc_o.store(\n            [i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne")\n        )\n\n\n@input_guard\ndef solve_tril(\n    A: torch.Tensor,\n    cu_seqlens: torch.Tensor | None = None,\n    chunk_indices: torch.Tensor | None = None,\n    output_dtype: torch.dtype = torch.float,\n) -> torch.Tensor:\n    """\n    Compute the inverse of the matrix I + A\n    A should be strictly lower triangular, i.e., A.triu() == 0.\n\n    Args:\n        A (torch.Tensor):\n            [B, T, H, BT], where BT should only be 16, 32, or 64.\n        cu_seqlens (torch.Tensor):\n            The cumulative sequence lengths of the input tensor. Default: `None`.\n        chunk_indices (torch.Tensor):\n            Pre-computed chunk indices. Default: `None`.\n        output_dtype (torch.dtype):\n            The dtype of the output tensor. Default: `torch.float`.\n            If `None`, the output dtype will be the same as the input dtype.\n\n    Returns:\n        (I + A)^-1 with the same shape as A\n    """\n    assert A.shape[-1] in [16, 32, 64]\n    output_dtype = A.dtype if output_dtype is None else output_dtype\n\n    B, T, H, BT = A.shape\n    if chunk_indices is None and cu_seqlens is not None:\n        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)\n    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)\n\n    Ai = torch.zeros_like(A, dtype=output_dtype)\n    if BT == 16:\n        merge_fn = solve_tril_16x16_kernel\n    elif BT == 32:\n        merge_fn = merge_16x16_to_32x32_inverse_kernel\n    elif BT == 64:\n        merge_fn = merge_16x16_to_64x64_inverse_kernel\n\n    merge_fn[NT, B * H](\n        A=A,\n        Ai=Ai,\n        cu_seqlens=cu_seqlens,\n        chunk_indices=chunk_indices,\n        T=T,\n        H=H,\n        BT=BT,\n        USE_TMA=is_tma_supported,\n        DOT_PRECISION=FLA_TRIL_PRECISION,\n    )\n    return Ai\n', 'utils.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# ruff: noqa: E501\nimport contextlib\nimport functools\nimport logging\nimport os\nfrom collections.abc import Callable\nfrom enum import Enum\nfrom typing import Any, Literal\n\nimport torch\n\nclass _P:\n    @staticmethod\n    def is_rocm():\n        return False\n    @staticmethod\n    def is_cuda_alike():\n        return True\n    @staticmethod\n    def is_arch_support_pdl():\n        return False\ncurrent_platform = _P()\nimport triton\n\nlogger = logging.getLogger(__name__)\n\nCOMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"\nFLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"\n\nSUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))\n\n# Default chunk size used across FLA triton kernels (kda, chunk, chunk_o, etc.)\nFLA_CHUNK_SIZE = 64\n\n\ndef tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:\n    """\n    A decorator that caches the most recent results of a function with tensor inputs.\n\n    This decorator will store the output of the decorated function for the most recent set of input tensors.\n    The cache is limited to a fixed size (default is 4). When the cache is full, the oldest entry will be removed.\n\n    Args:\n        fn (Callable[..., torch.Tensor]):\n            The function to be decorated. It should take tensor inputs and return tensor outputs.\n\n    Returns:\n        Callable[..., torch.Tensor]:\n            A wrapped version of the input function with single-entry caching.\n    """\n\n    cache_entries: tuple[tuple | None, dict | None, Any] = []\n    cache_size = 8\n\n    @functools.wraps(fn)\n    def wrapper(*args: Any, **kwargs: Any) -> Any:\n        nonlocal cache_entries, cache_size\n        for i, entry in enumerate(cache_entries):\n            last_args, last_kwargs, last_result = entry\n            if (\n                len(args) == len(last_args)\n                and len(kwargs) == len(last_kwargs)\n                and all(a is b for a, b in zip(args, last_args))\n                and all(\n                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()\n                )\n            ):\n                cache_entries = (\n                    cache_entries[:i]\n                    + cache_entries[i + 1 :]\n                    + [(args, kwargs, last_result)]\n                )\n                return last_result\n\n        result = fn(*args, **kwargs)\n\n        if len(cache_entries) >= cache_size:\n            cache_entries = cache_entries[1:]\n        cache_entries.append((args, kwargs, result))\n        return result\n\n    return wrapper\n\n\ndef input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:\n    """\n    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.\n    """\n\n    @functools.wraps(fn)\n    def wrapper(*args, **kwargs):\n        contiguous_args = (\n            i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args\n        )\n        contiguous_kwargs = {\n            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())\n            for k, v in kwargs.items()\n        }\n\n        tensor = None\n        for arg in args:\n            if isinstance(arg, torch.Tensor):\n                tensor = arg\n                break\n        if tensor is None:\n            for value in kwargs.values():\n                if isinstance(value, torch.Tensor):\n                    tensor = value\n                    break\n\n        if tensor is not None:\n            ctx = torch.accelerator.device_index(tensor.device.index)\n        else:\n            ctx = contextlib.nullcontext()\n\n        with ctx:\n            return fn(*contiguous_args, **contiguous_kwargs)\n\n    return wrapper\n\n\n@functools.cache\ndef get_available_device() -> str:\n    try:\n        return triton.runtime.driver.active.get_current_target().backend\n    except (RuntimeError, AttributeError):\n        return "cpu"\n\n\n@functools.cache\ndef _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:\n    device = get_available_device()\n    mapping = {\n        "cuda": "nvidia",\n        "hip": "amd",\n        "xpu": "intel",\n    }\n    # return the mapped value, or the original if not found\n    return mapping.get(device, device)\n\n\n# For AMD GPUs, the triton backend is \'hip\', while for Nvidia GPUs, the triton backend is \'cuda\'.\n# However, the torch backend is \'cuda\' for both Nvidia and AMD GPUs.\n# Therefore, we need to check the triton backend to determine the actual GPU vendor.\ndevice = "cuda" if current_platform.is_cuda_alike() else get_available_device()\ndevice_torch_lib = getattr(torch, device, None)\ndevice_platform = _check_platform()\n\nis_amd = device_platform == "amd"\nis_intel = device_platform == "intel"\nis_nvidia = device_platform == "nvidia"\nis_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)\nis_nvidia_hopper = is_nvidia and (\n    "NVIDIA H" in torch.cuda.get_device_name(0)\n    or torch.cuda.get_device_capability()[0] >= 9\n)\nuse_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"\nis_gather_supported = hasattr(triton.language, "gather")\nis_tma_supported = (\n    is_nvidia_hopper\n    and os.getenv("FLA_USE_TMA", "0") == "1"\n    and (\n        hasattr(triton.language, "_experimental_make_tensor_descriptor")\n        or hasattr(triton.language, "make_tensor_descriptor")\n    )\n)\n\n\ndef get_all_max_shared_mem():\n    try:\n        return [\n            triton.runtime.driver.active.utils.get_device_properties(i)[\n                "max_shared_mem"\n            ]\n            for i in range(device_torch_lib.device_count())\n        ]\n    except BaseException:\n        return [-1]\n\n\nclass Backend(Enum):\n    ADA = 101376  # RTX 4090\n    AMPERE = 166912  # A100\n    HOPPER = 232448  # H100\n    DEFAULT = 102400  # Default\n\n    @classmethod\n    def get_shared_memory(cls, arch: str) -> int:\n        try:\n            return cls[arch.upper()].value\n        except KeyError:\n            return cls.DEFAULT.value\n\n\n@functools.cache\ndef check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:\n    try:\n        device_shared_mem_list = get_all_max_shared_mem()\n        max_shared_memory = device_shared_mem_list[tensor_idx]\n        return max_shared_memory >= Backend.get_shared_memory(arch)\n    except Exception:\n        return False\n', 'wy_fast.py': '# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang\n#\n# This file contains code copied from the flash-linear-attention project.\n# The original source code was licensed under the MIT license and included\n# the following copyright notice:\n# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n\n# ruff: noqa: E501\n\nimport torch\n\nimport triton\nimport triton.language as tl\n\nfrom .index import prepare_chunk_indices\n\n\n@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})\n@triton.autotune(\n    configs=[\n        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n        for num_warps in [2, 4, 8]\n        for num_stages in [2, 3, 4]\n    ],\n    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],\n)\n@triton.jit(do_not_specialize=["T"])\ndef recompute_w_u_fwd_kernel(\n    k,\n    v,\n    beta,\n    w,\n    u,\n    A,\n    g,\n    cu_seqlens,\n    chunk_indices,\n    T,\n    H: tl.constexpr,\n    Hg: tl.constexpr,\n    K: tl.constexpr,\n    V: tl.constexpr,\n    BT: tl.constexpr,\n    BK: tl.constexpr,\n    BV: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n):\n    i_t, i_bh = tl.program_id(0), tl.program_id(1)\n    i_b, i_h = i_bh // H, i_bh % H\n    if IS_VARLEN:\n        i_n, i_t = (\n            tl.load(chunk_indices + i_t * 2).to(tl.int32),\n            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),\n        )\n        bos, eos = (\n            tl.load(cu_seqlens + i_n).to(tl.int32),\n            tl.load(cu_seqlens + i_n + 1).to(tl.int32),\n        )\n        T = eos - bos\n    else:\n        bos, eos = i_b * T, i_b * T + T\n    p_beta = tl.make_block_ptr(\n        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)\n    )\n    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))\n    p_A = tl.make_block_ptr(\n        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)\n    )\n    b_beta = tl.load(p_beta, boundary_check=(0,))\n    b_A = tl.load(p_A, boundary_check=(0, 1))\n    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))\n\n    for i_v in range(tl.cdiv(V, BV)):\n        p_v = tl.make_block_ptr(\n            v + (bos * H + i_h) * V,\n            (T, V),\n            (H * V, 1),\n            (i_t * BT, i_v * BV),\n            (BT, BV),\n            (1, 0),\n        )\n        p_u = tl.make_block_ptr(\n            u + (bos * H + i_h) * V,\n            (T, V),\n            (H * V, 1),\n            (i_t * BT, i_v * BV),\n            (BT, BV),\n            (1, 0),\n        )\n        b_v = tl.load(p_v, boundary_check=(0, 1))\n        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)\n        b_u = tl.dot(b_A, b_vb, allow_tf32=False)\n        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))\n\n    for i_k in range(tl.cdiv(K, BK)):\n        p_k = tl.make_block_ptr(\n            k + (bos * Hg + i_h // (H // Hg)) * K,\n            (T, K),\n            (Hg * K, 1),\n            (i_t * BT, i_k * BK),\n            (BT, BK),\n            (1, 0),\n        )\n        p_w = tl.make_block_ptr(\n            w + (bos * H + i_h) * K,\n            (T, K),\n            (H * K, 1),\n            (i_t * BT, i_k * BK),\n            (BT, BK),\n            (1, 0),\n        )\n        b_k = tl.load(p_k, boundary_check=(0, 1))\n        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)\n        b_w = tl.dot(b_A, b_kb)\n        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))\n\n\ndef recompute_w_u_fwd(\n    k: torch.Tensor,\n    v: torch.Tensor,\n    beta: torch.Tensor,\n    g_cumsum: torch.Tensor,\n    A: torch.Tensor,\n    cu_seqlens: torch.Tensor | None,\n    chunk_indices: torch.Tensor | None = None,\n) -> tuple[torch.Tensor, torch.Tensor]:\n    B, T, Hg, K, V = *k.shape, v.shape[-1]\n    H = v.shape[-2]\n    BT = A.shape[-1]\n\n    if chunk_indices is None and cu_seqlens is not None:\n        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)\n    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)\n    BK = 64\n    BV = 64\n    u = torch.empty_like(v)\n    w = k.new_empty(B, T, H, K)\n    recompute_w_u_fwd_kernel[(NT, B * H)](\n        k=k,\n        v=v,\n        beta=beta,\n        w=w,\n        u=u,\n        A=A,\n        g=g_cumsum,\n        cu_seqlens=cu_seqlens,\n        chunk_indices=chunk_indices,\n        T=T,\n        H=H,\n        Hg=Hg,\n        K=K,\n        V=V,\n        BT=BT,\n        BK=BK,\n        BV=BV,\n    )\n    return w, u\n'}

_VLA_OPS_CACHE = {}


def _load_vla_ops():
    """Materialize the vendored vLLM FLA chunk/norm ops into a real package dir
    and import them; triton.jit needs inspectable source files."""
    if "fns" in _VLA_OPS_CACHE:
        return _VLA_OPS_CACHE["fns"]
    import hashlib
    import importlib
    import importlib.util
    import tempfile

    try:
        import triton  # noqa: F401

        tag = hashlib.sha256("".join(
            _VLA_OPS_FILES[m] for m in sorted(_VLA_OPS_FILES)).encode()).hexdigest()[:10]
        root = os.path.join(tempfile.gettempdir(), f"verdict_vla_ops_{tag}")
        if not os.path.exists(os.path.join(root, "chunk.py")):
            os.makedirs(root, exist_ok=True)
            for name, src in _VLA_OPS_FILES.items():
                tmp = os.path.join(root, name + f".{os.getpid()}.tmp")
                with open(tmp, "w") as f:
                    f.write(src)
                os.replace(tmp, os.path.join(root, name))
            with open(os.path.join(root, "__init__.py"), "w") as f:
                f.write("")
        parent = os.path.dirname(root)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        chunk_fn = importlib.import_module(f"{os.path.basename(root)}.chunk").chunk_gated_delta_rule
        norm_mod = importlib.import_module(f"{os.path.basename(root)}.fused_norm_gate")
        fns = (chunk_fn, norm_mod.rms_norm_gated)
        print(f"[executor] vendored vLLM chunk/norm ops active ({tag})",
              file=sys.stderr, flush=True)
    except Exception as err:
        print(f"[executor] vendored vLLM chunk/norm ops unavailable "
              f"({type(err).__name__}: {err}); fla fallback", file=sys.stderr, flush=True)
        fns = (None, None)
    _VLA_OPS_CACHE["fns"] = fns
    return fns


_FUSED_PREP_SRC = r'''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused post-conv1d preparation for GDN prefill.

Replaces the chain:
    split → rearrange → contiguous * 3 → l2norm * 2 → gating
with a **single Triton kernel** that reads the conv'd mixed_qkv output
and writes directly to q/k/v/g/beta in the target contiguous layout.

"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _fused_post_conv_kernel(
    # ---- inputs ----
    mixed_qkv_ptr,  # [L, qkv_dim] conv'd output (contiguous)
    a_ptr,  # [L, HV]
    b_ptr,  # [L, HV]
    # ---- params ----
    A_log_ptr,  # [HV]
    dt_bias_ptr,  # [HV]
    # ---- outputs ----
    q_ptr,  # [L, H, K] contiguous
    k_ptr,  # [L, H, K] contiguous
    v_ptr,  # [L, HV, V] contiguous
    g_ptr,  # [L, HV] float32
    beta_ptr,  # [L, HV] float32
    # ---- strides ----
    stride_x_tok,  # qkv_dim
    stride_a_tok,  # HV
    stride_b_tok,  # HV
    stride_q_tok,  # H * K
    stride_k_tok,  # H * K
    stride_v_tok,  # HV * V
    # ---- dims ----
    L,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Single fused kernel for post-conv1d preparation.

    Grid: (ceil(L, BLOCK_T), H + HV)
      - program_id(1) in [0, H):    Q/K head processing + l2norm
      - program_id(1) in [H, H+HV): V head processing + gating
    """
    i_tb = tl.program_id(0)
    i_head = tl.program_id(1)

    HK: tl.constexpr = H * K

    offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    mask_t = offs_t < L

    if i_head < H:
        # ============ Q/K head processing ============
        i_h = i_head
        offs_k = tl.arange(0, BK)  # [BK]
        mask_k = offs_k < K
        mask_2d = mask_t[:, None] & mask_k[None, :]  # [BLOCK_T, BK]

        # Load Q features: mixed_qkv[t, i_h*K + k]
        q_offsets = offs_t[:, None] * stride_x_tok + i_h * K + offs_k[None, :]
        q_f32 = tl.load(mixed_qkv_ptr + q_offsets, mask=mask_2d, other=0).to(tl.float32)

        # Load K features: mixed_qkv[t, HK + i_h*K + k]
        k_offsets = offs_t[:, None] * stride_x_tok + HK + i_h * K + offs_k[None, :]
        k_f32 = tl.load(mixed_qkv_ptr + k_offsets, mask=mask_2d, other=0).to(tl.float32)

        if APPLY_L2NORM:
            q_sq_sum = tl.sum(q_f32 * q_f32, axis=1)  # [BLOCK_T]
            q_inv = 1.0 / tl.sqrt(q_sq_sum + L2NORM_EPS)
            q_f32 = q_f32 * q_inv[:, None]

            k_sq_sum = tl.sum(k_f32 * k_f32, axis=1)
            k_inv = 1.0 / tl.sqrt(k_sq_sum + L2NORM_EPS)
            k_f32 = k_f32 * k_inv[:, None]

        # Store Q
        q_out = offs_t[:, None] * stride_q_tok + i_h * K + offs_k[None, :]
        tl.store(
            q_ptr + q_out,
            q_f32.to(q_ptr.dtype.element_ty),
            mask=mask_2d,
        )

        # Store K
        k_out = offs_t[:, None] * stride_k_tok + i_h * K + offs_k[None, :]
        tl.store(
            k_ptr + k_out,
            k_f32.to(k_ptr.dtype.element_ty),
            mask=mask_2d,
        )
    else:
        # ============ V head + gating processing ============
        i_hv = i_head - H
        offs_v = tl.arange(0, BV)  # [BV]
        mask_v = offs_v < V
        mask_2d = mask_t[:, None] & mask_v[None, :]  # [BLOCK_T, BV]

        V_OFFSET: tl.constexpr = 2 * H * K

        # Load V features: mixed_qkv[t, 2*H*K + i_hv*V + v]
        v_offsets = (
            offs_t[:, None] * stride_x_tok + V_OFFSET + i_hv * V + offs_v[None, :]
        )
        v_vals = tl.load(mixed_qkv_ptr + v_offsets, mask=mask_2d, other=0)

        # Store V
        v_out = offs_t[:, None] * stride_v_tok + i_hv * V + offs_v[None, :]
        tl.store(v_ptr + v_out, v_vals, mask=mask_2d)

        # Gating: one scalar per (token, v-head)
        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

        a_offsets = offs_t * stride_a_tok + i_hv
        b_offsets = offs_t * stride_b_tok + i_hv
        a_vals = tl.load(a_ptr + a_offsets, mask=mask_t, other=0).to(tl.float32)
        b_vals = tl.load(b_ptr + b_offsets, mask=mask_t, other=0).to(tl.float32)

        # g = -exp(A_log) * softplus(a + dt_bias)
        x = a_vals + dt_bias_val
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)
        g_vals = -tl.exp(A_log_val) * sp

        if OUTPUT_G_EXP:
            g_vals = tl.exp(g_vals)

        beta_vals = tl.sigmoid(b_vals)

        gb_offsets = offs_t * HV + i_hv
        tl.store(g_ptr + gb_offsets, g_vals, mask=mask_t)
        tl.store(beta_ptr + gb_offsets, beta_vals, mask=mask_t)


def fused_post_conv_prep(
    conv_output: torch.Tensor,  # [L, qkv_dim] conv'd mixed_qkv
    a: torch.Tensor,  # [L, HV]
    b: torch.Tensor,  # [L, HV]
    A_log: torch.Tensor,  # [HV]
    dt_bias: torch.Tensor,  # [HV]
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused post-conv1d prep: split + l2norm + gating in one kernel.

    Args:
        conv_output: [L, qkv_dim] contiguous conv'd mixed_qkv
        a: [L, HV] gating input
        b: [L, HV] gating input
        A_log: [HV] log decay parameter
        dt_bias: [HV] dt bias parameter
        num_k_heads: number of K heads (H)
        head_k_dim: dimension per K head (K)
        head_v_dim: dimension per V head (V)
        apply_l2norm: whether to L2-normalize q and k
        output_g_exp: if True, output exp(g) instead of g (for FlashInfer)

    Returns:
        q: [L, H, K] contiguous, optionally l2-normalized
        k: [L, H, K] contiguous, optionally l2-normalized
        v: [L, HV, V] contiguous
        g: [L, HV] float32
        beta: [L, HV] float32
    """
    L = conv_output.shape[0]
    qkv_dim = conv_output.shape[1]
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]
    dtype = conv_output.dtype
    device = conv_output.device

    assert qkv_dim == 2 * H * K + HV * V, (
        f"qkv_dim={qkv_dim} != 2*H*K + HV*V = {2 * H * K + HV * V}"
    )

    # Allocate outputs in target contiguous layout
    q = torch.empty(L, H, K, dtype=dtype, device=device)
    k = torch.empty(L, H, K, dtype=dtype, device=device)
    v = torch.empty(L, HV, V, dtype=dtype, device=device)
    g = torch.empty(L, HV, dtype=torch.float32, device=device)
    beta = torch.empty(L, HV, dtype=torch.float32, device=device)

    if L == 0:
        return q, k, v, g, beta

    # ---- Kernel config ----
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    BLOCK_T = 16  # tokens per block

    # Single kernel: blocks [0,H) do Q/K, blocks [H, H+HV) do V+gating
    grid = (triton.cdiv(L, BLOCK_T), H + HV)
    _fused_post_conv_kernel[grid](
        mixed_qkv_ptr=conv_output,
        a_ptr=a,
        b_ptr=b,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        stride_x_tok=conv_output.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_q_tok=q.stride(0),
        stride_k_tok=k.stride(0),
        stride_v_tok=v.stride(0),
        L=L,
        H=H,
        HV=HV,
        K=K,
        V=V,
        APPLY_L2NORM=apply_l2norm,
        L2NORM_EPS=1e-6,
        OUTPUT_G_EXP=output_g_exp,
        SOFTPLUS_THRESHOLD=20.0,
        BLOCK_T=BLOCK_T,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )

    return q, k, v, g, beta
'''


_FUSED_PREP_CACHE = {}


def _load_fused_prep():
    """Materialize the vendored kernel into a real file and import it:
    triton.jit needs inspectable source, which python -c cannot provide."""
    if "fn" in _FUSED_PREP_CACHE:
        return _FUSED_PREP_CACHE["fn"]
    import hashlib
    import importlib.util
    import tempfile

    try:
        name = ("verdict_fused_prep_"
                + hashlib.sha256(_FUSED_PREP_SRC.encode()).hexdigest()[:10])
        path = os.path.join(tempfile.gettempdir(), name + ".py")
        if not os.path.exists(path):
            tmp = path + f".{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                f.write(_FUSED_PREP_SRC)
            os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.fused_post_conv_prep
    except Exception as err:
        print(f"[executor] fused GDN prep unavailable ({type(err).__name__}: {err}); "
              "stock layers", file=sys.stderr, flush=True)
        fn = None
    _FUSED_PREP_CACHE["fn"] = fn
    return fn


def _gdn_fast_forward(mod, vla_chunk=None, vla_norm=None):
    """Fused GDN layer forward: one qkvz GEMM, one ba GEMM, vendored
    fused_post_conv_prep with in-kernel l2norm (the delta kernels then run with
    use_qk_l2norm_in_kernel=False). Returns None whenever the layer does not
    match the shape this fast path knows, leaving the stock forward in place."""
    fused_post_conv_prep = _load_fused_prep()
    if fused_post_conv_prep is None:
        return None
    m = sys.modules.get(type(mod).__module__)
    if m is None:
        return None
    try:
        chunk_fn = m.torch_chunk_gated_delta_rule
        recurrent_fn = m.torch_recurrent_gated_delta_rule
        conv_fn = m.causal_conv1d_fn
        conv_update = m.causal_conv1d_update
        apply_mask = m.apply_mask_to_padding_states
    except AttributeError:
        return None
    if vla_chunk is None:
        vla_chunk = chunk_fn
        vla_norm = None
    lins = []
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
        lin = getattr(mod, name, None)
        if lin is None or getattr(lin, "bias", None) is not None:
            return None
        lins.append(lin.weight.data)
    w_qkvz = torch.cat([lins[0], lins[1]], 0)
    w_ba = torch.cat([lins[2], lins[3]], 0)
    conv_dim = w_qkvz.shape[0] - lins[1].shape[0]
    hv = lins[3].shape[0]
    n_heads, n_v_heads = mod.num_k_heads, mod.num_v_heads
    head_k, head_v = mod.head_k_dim, mod.head_v_dim
    ratio = n_v_heads // n_heads
    layer_idx = mod.layer_idx

    def forward(hidden_states, cache_params=None, attention_mask=None, **kwargs):
        import torch.nn.functional as F

        hidden_states = apply_mask(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_pre = (cache_params is not None
                   and cache_params.has_previous_state(layer_idx, state_idx=0))

        qkvz = F.linear(hidden_states, w_qkvz)
        ba = F.linear(hidden_states, w_ba)
        mixed = qkvz[..., :conv_dim].transpose(1, 2)
        z = qkvz[..., conv_dim:].reshape(batch_size, seq_len, -1, head_v)
        a = ba[..., :hv]
        b = ba[..., hv:]

        if use_pre and seq_len == 1 and not cache_params.layers[layer_idx].record_past:
            mixed = conv_update(
                mixed,
                cache_params.layers[layer_idx].conv_states[0],
                mod.conv1d.weight.squeeze(1),
                mod.conv1d.bias,
                mod.activation,
            )
        else:
            if cache_params is not None:
                mixed = cache_params.update_conv_state(
                    mixed, layer_idx, conv_kernel_size=mod.conv_kernel_size)
            mixed = conv_fn(
                mixed,
                mod.conv1d.weight.squeeze(1),
                mod.conv1d.bias,
                activation=mod.activation,
                **kwargs,
            )
            if cache_params is not None:
                mixed = mixed[:, :, -seq_len:]

        conv_out = mixed.transpose(1, 2)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv_out.reshape(-1, conv_out.shape[-1]),
            a=a.reshape(-1, hv),
            b=b.reshape(-1, hv),
            A_log=mod.A_log,
            dt_bias=mod.dt_bias,
            num_k_heads=n_heads,
            head_k_dim=head_k,
            head_v_dim=head_v,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.view(batch_size, seq_len, n_heads, head_k)
        k = k.view(batch_size, seq_len, n_heads, head_k)
        v = v.view(batch_size, seq_len, n_v_heads, head_v)
        g = g.view(batch_size, seq_len, n_v_heads)
        beta = beta.view(batch_size, seq_len, n_v_heads)
        if ratio > 1:
            q = q.repeat_interleave(ratio, dim=2)
            k = k.repeat_interleave(ratio, dim=2)

        recurrent_state = (cache_params.layers[layer_idx].recurrent_states[0]
                           if use_pre else None)
        cu_seqlens = kwargs.get("cu_seq_lens_q")
        for junk in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q",
                     "max_length_k", "seq_idx"):
            kwargs.pop(junk, None)
        core_kwargs = dict(
            g=g, beta=beta, initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=False)
        if cu_seqlens is not None:
            core_kwargs["cu_seqlens"] = cu_seqlens
        if use_pre and seq_len == 1:
            core_attn_out, last = recurrent_fn(q, k, v, **core_kwargs)
        else:
            core_attn_out, last = vla_chunk(q, k, v, **core_kwargs)
        if cache_params is not None:
            cache_params.update_recurrent_state(last, layer_idx)
        core_attn_out = core_attn_out.reshape(-1, head_v)
        z = z.reshape(-1, head_v)
        if vla_norm is not None:
            core_attn_out = vla_norm(core_attn_out, z.reshape(-1, z.shape[-1]),
                                     mod.norm.weight, None,
                                     activation=mod.norm.activation,
                                     eps=mod.norm.variance_epsilon)
        else:
            core_attn_out = mod.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return mod.out_proj(core_attn_out)

    return forward


def make_gdn_layers_opaque(model) -> int:
    """Keep gated-delta-net layers out of dynamo tracing (their cache-object
    attribute reads and per-shape branching recompile on every distinct fork
    shape) and install the fused fast path where the layer shape matches."""
    import torch._dynamo

    probes = ("A_log", "dt_bias", "num_v_heads", "head_k_dim", "conv_kernel_size")
    vla_chunk, vla_norm = _load_vla_ops()
    n = fast = 0
    for mod in model.modules():
        if not all(hasattr(mod, a) for a in probes):
            continue
        fwd = _gdn_fast_forward(mod, vla_chunk, vla_norm)
        if fwd is not None:
            fast += 1
        else:
            fwd = mod.forward
        mod.forward = torch._dynamo.disable(fwd)
        n += 1
    print(f"[executor] {n} GDN layers out of dynamo tracing "
          f"({fast} on the fused fast path)", file=sys.stderr, flush=True)
    return n

class GraphMemoryLow(RuntimeError):
    """Not enough free GPU memory to capture another graph, even after dropping the cached ones."""

GPU_MEMORY_UTILIZATION = float(os.environ.get("VERDICT_GPU_MEMORY_UTILIZATION", "0.92"))
TRANSIENT_PEAK = [0]

def _budget_bytes() -> int:
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return int(total * GPU_MEMORY_UTILIZATION)

def _over_budget(extra: int = 0) -> bool:
    return torch.cuda.memory_reserved() + TRANSIENT_PEAK[0] + extra >= _budget_bytes()

def _require_room() -> None:
    """Raise GraphMemoryLow once this process holds its declared gpu-memory-utilization share; capture runs only at boot, so returning cached blocks first is allowed here."""
    torch.cuda.empty_cache()
    if _over_budget():
        raise GraphMemoryLow(
            f"this instance holds its {GPU_MEMORY_UTILIZATION:.2f} gpu-memory-utilization share")

def _measure_transient_peak(model, pad_id, rows, length) -> int:
    """One eager forward at the largest served shape; the allocation peak above the live set is the headroom every capture must leave inside the budget."""
    ids = torch.full((rows, length), pad_id, dtype=torch.long, device=DEV)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated() - base)
    del ids
    torch.cuda.empty_cache()
    return peak

def _bucket(n: int, step: int) -> int:
    return -(-n // step) * step

def _capture_graph(run, pool=None):
    """Capture `run` into `pool`, after draining work already in flight: a capture that allocates or frees pool memory while a replay runs corrupts the context."""
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        out = run()
    torch.cuda.synchronize()
    return out, graph

class MediaGraphs:
    """Whole-prompt decoder graphs over media embeddings, one per (rows, length) bucket."""

    B_BUCKETS = GRAPH_ROW_LADDER
    PREWARM_MISS_LIMIT = 3

    def __init__(self, decoder, sizes):
        self.decoder = decoder
        self.sizes = tuple(sorted(sizes))
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()

    def plan(self, n, b):
        size = next((s for s in self.sizes if s >= n), None)
        rows = next((x for x in self.B_BUCKETS if x >= b), None)
        return (rows, size) if size is not None and rows is not None else None

    @torch.inference_mode()
    def _capture(self, rows, size):
        _require_room()
        w = self.decoder.get_input_embeddings().weight
        emb = torch.zeros((rows, size, w.shape[1]), dtype=w.dtype, device=DEV)
        pos = torch.zeros((3, rows, size), dtype=torch.long, device=DEV)

        def run():
            return self.decoder(inputs_embeds=emb, position_ids=pos, use_cache=False,
                                return_dict=True).last_hidden_state

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        out, graph = _capture_graph(run, self.pool)
        self.graphs[(rows, size)] = (emb, pos, out, graph)
        print(f"[executor] captured a media graph for {rows} rows x {size} tokens "
              f"(free {torch.cuda.mem_get_info()[0] >> 20} MiB)", file=sys.stderr, flush=True)
        return self.graphs[(rows, size)]

    @torch.inference_mode()
    def prewarm(self) -> None:
        """Capture every (rows, length) cell, largest first; a cell that does not fit stays eager."""
        keys = sorted(((r, s) for s in self.sizes for r in self.B_BUCKETS),
                      key=lambda k: k[0] * k[1], reverse=True)
        for key in keys:
            try:
                self._capture(*key)
            except (GraphMemoryLow, torch.OutOfMemoryError, RuntimeError) as err:
                if _fatal_cuda(err):
                    raise
                torch.cuda.empty_cache()
                print(f"[executor] media graph {key[0]}x{key[1]} not prewarmed "
                      f"({type(err).__name__}: {err}); eager for the life of the process",
                      file=sys.stderr, flush=True)
        print(f"[executor] media graphs prewarmed {len(self.graphs)}/{len(keys)}",
              file=sys.stderr, flush=True)

    def stats(self):
        return len(self.graphs), len(self.sizes) * len(self.B_BUCKETS)

    def why(self, n, b):
        """Why a batch has no media graph: past the ladder, or a cell the boot ladder could not fit."""
        if n > self.sizes[-1]:
            return "len"
        if b > self.B_BUCKETS[-1]:
            return "rows"
        return "degraded"

    def __call__(self, embeds, positions, ends):
        n = int(embeds.shape[1])
        b = int(embeds.shape[0])
        key = self.plan(n, b)
        entry = self.graphs.get(key) if key is not None else None
        if entry is None:
            return None
        emb, pos, out, g = entry
        rows, size = key
        emb.zero_()
        emb[:b, :n].copy_(embeds)
        pos.zero_()
        pos[:, :b, :n].copy_(positions)
        pos[:, :b, n:] = positions[:, :b, n - 1:n] + torch.arange(
            1, size - n + 1, device=pos.device)
        g.replay()
        return out[torch.arange(b, device=out.device), ends]

class ForkGraphs:
    """Prefix and batched-branch graphs over a forked cache.

    The prefix is left-padded with zero embeddings: no layer has a bias and a zero linear-attention state stays zero, so the real tokens' states match an unpadded run."""

    ROW_BUCKETS = tuple(x for x in _double_ladder(FORK_ROWS_CAP) if x >= 2)

    PREWARM_MISS_LIMIT = 3

    def __init__(self, decoder, fork, pad_id, step: int = GRAPH_PREFIX_STEP,
                 max_len: int = 16384, width_step: int = GRAPH_WIDTH_STEP,
                 max_width: int = FORK_GRAPH_MAX_WIDTH, max_rows: int = FORK_ROWS_CAP,
                 max_prefix: int = FORK_GRAPH_MAX_PREFIX,
                 compile_fwd: bool = False) -> None:
        self.decoder, self.fork, self.pad_id = decoder, fork, pad_id
        if compile_fwd:
            self.decoder = torch.compile(decoder, dynamic=True)
        self.step, self.max_len = step, max_len
        explicit = os.environ.get("VERDICT_GRAPH_PREFIX_BUCKETS", "")
        if explicit:
            cap = min(max_len, max_prefix)
            picked = sorted({int(x) for x in explicit.split(",") if x.strip()})
            self.caps = tuple(x for x in picked if 0 < x <= cap) or _double_ladder(cap, step)
        else:
            self.caps = _double_ladder(min(max_len, max_prefix), step)
        self.width_step, self.max_width, self.max_rows = width_step, max_width, max_rows
        self.prefills = collections.OrderedDict()
        self.branches = collections.OrderedDict()
        self.multi = collections.OrderedDict()
        self.src = {}
        self.batch_max = int(os.environ.get("VERDICT_FORK_BATCH_MAX", "4"))
        self.off = False
        self.pool = torch.cuda.graph_pool_handle()
        self.cache_budget = int(os.environ.get("VERDICT_PREFIX_CACHE_MB", "4096")) << 20
        self.cache_budget = max(0, min(
            self.cache_budget,
            _budget_bytes() - torch.cuda.memory_reserved() - TRANSIENT_PEAK[0]))
        self.prefix_cache = collections.OrderedDict()
        self.cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def prewarm_widths(self):
        return tuple(range(self.width_step, self.max_width + 1, self.width_step))

    def plan(self, prefix: int, rows: int, width: int):
        size, wide = next((x for x in self.caps if x >= prefix), None), _bucket(width, self.width_step)
        rows_b = next((x for x in self.ROW_BUCKETS if x >= rows), None)
        if (size is None or rows_b is None or rows_b > self.max_rows
                or wide > self.max_width):
            return None
        return (size, rows_b, wide)

    def stats(self):
        widths = sum(1 for x in self.prewarm_widths() if x <= self.max_width)
        return len(self.branches), len(self.caps) * len(self.ROW_BUCKETS) * widths

    def why(self, prefix: int, rows: int, width: int):
        """Why a group has no fork graph: past a declared ceiling, or a cell the boot ladder could not fit."""
        if prefix > self.caps[-1]:
            return "fork_prefix"
        if _bucket(width, self.width_step) > self.max_width:
            return "fork_width"
        if rows > self.max_rows:
            return "fork_rows"
        return "degraded" if self.plan(prefix, rows, width) is not None else "fork_rows"

    def __call__(self, embeds, positions, suffixes: list, delta: int, prefix_key=None,
                 cached=None):
        """embeds (1,P,hidden) and positions (3,1,P) of the prefix plus the branch token ids -> final-norm hidden state per branch end, or None past the limits. A `cached` snapshot from probe() supplies P and the rope delta and skips the prefix fill; embeds/positions may be None then."""
        n = len(suffixes)
        if cached is None:
            P = embeds.shape[1]
        else:
            _, P, delta = cached
        plan = self.plan(P, n, max(len(s) for s in suffixes))
        if plan is None:
            return None
        size, rows, wide = plan
        if size in self.prefills:
            self.prefills.move_to_end(size)
        entry = self.prefills.get(size)
        if entry is None:
            return None
        emb, pos, pad, cache, graph = entry
        key = (size, rows, wide)
        if key in self.branches:
            self.branches.move_to_end(key)
        bentry = self.branches.get(key)
        if bentry is None:
            return None
        a = size - P
        if cached is None:
            cached = self._cached_state(prefix_key, size)
        if cached is None:
            emb[:, :a].zero_()
            emb[:, a:].copy_(embeds)
            pos[..., :a].zero_()
            pos[..., a:].copy_(positions)
            pad.fill_(a)
            _tp0 = time.perf_counter()
            graph.replay()
            if MM_PROFILE:
                torch.cuda.synchronize()
            FORK_STATS["prefix"] = (time.perf_counter() - _tp0) * 1e3
            self._remember_state(prefix_key, size, cache, P, delta)
        else:
            try:
                _copy_state(cached[0], cache)
            except RuntimeError as err:
                self.prefix_cache.clear()
                self.cache_bytes = 0
                print(f"[executor] prefix cache cleared ({err})",
                      file=sys.stderr, flush=True)
                return None
            FORK_STATS["prefix"] = 0.0
        pad.fill_(a)
        ids, base, pad_b, out, bgraph = bentry
        host_np = np.full((rows, wide), self.pad_id, dtype="<i8")
        for r in range(rows):
            suf = suffixes[r % n]
            host_np[r, :len(suf)] = np.asarray(suf)
        ids.copy_(torch.from_numpy(host_np), non_blocking=True)
        base.fill_(P + delta)
        pad_b.fill_(a)
        _tb0 = time.perf_counter()
        bgraph.replay()
        if MM_PROFILE:
            torch.cuda.synchronize()
        FORK_STATS["branch"] = (time.perf_counter() - _tb0) * 1e3
        ends = torch.from_numpy(
            np.array([len(s) - 1 for s in suffixes], dtype="<i8")).to(DEV)
        return out[torch.arange(n, device=DEV), ends]

    def probe(self, pkey, prefix_len=None):
        """The cached (state, P, delta) for a prefix key at the graph size its length plans to, or None; a hit marks the entry recently used."""
        if pkey is None or self.off:
            return None
        size = None
        if prefix_len is not None:
            size = next((x for x in self.caps if x >= prefix_len), None)
            if size is None:
                return None
        return self._cached_state(pkey, size)

    def _cached_state(self, pkey, size=None):
        if pkey is None or self.cache_budget <= 0:
            return None
        key = (pkey, size)
        snap = self.prefix_cache.get(key)
        if snap is None:
            self.cache_misses += 1
            PREFIX_STATS["miss"] += 1
            return None
        self.prefix_cache.move_to_end(key)
        self.cache_hits += 1
        PREFIX_STATS["hit"] += 1
        return snap

    def _remember_state(self, pkey, size, cache, P: int = 0, delta: int = 0) -> None:
        if pkey is None or self.cache_budget <= 0:
            return
        try:
            held = sum(t.numel() * t.element_size() for t in _state_tensors(cache))
            if held > self.cache_budget or _over_budget(held):
                return
            snap = copy.deepcopy(cache)
            need = sum(t.numel() * t.element_size() for t in _state_tensors(snap))
            while self.prefix_cache and self.cache_bytes + need > self.cache_budget:
                _, old = self.prefix_cache.popitem(last=False)
                self.cache_bytes -= sum(t.numel() * t.element_size()
                                        for t in _state_tensors(old[0]))
                del old
            if need > self.cache_budget:
                return
            self.prefix_cache[(pkey, size)] = (snap, P, delta)
            self.cache_bytes += need
            PREFIX_STATS["bytes"] = self.cache_bytes
        except Exception as err:
            self.prefix_cache.clear()
            self.cache_bytes = 0
            PREFIX_STATS["bytes"] = 0
            print(f"[executor] prefix cache paused ({type(err).__name__}: {err}); "
                  f"serving misses", file=sys.stderr, flush=True)

    def clear(self) -> None:
        torch.cuda.synchronize()
        self.branches.clear()
        self.prefills.clear()

    def _side_warmup(self, run) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)

    @torch.inference_mode()
    def _capture_prefill(self, size: int, embeds, positions):
        dev = embeds.device
        emb = torch.zeros((1, size, embeds.shape[-1]), dtype=embeds.dtype, device=dev)
        pos = torch.zeros_like(positions)
        pad = torch.zeros((), dtype=torch.long, device=dev)

        def run():
            idx = torch.arange(size, device=dev)
            i, j = idx.view(-1, 1), idx.view(1, -1)
            mask = (j <= i) & ((i < pad) == (j < pad))
            return self.decoder(inputs_embeds=emb, position_ids=pos, use_cache=True,
                                return_dict=True,
                                attention_mask={"full_attention": mask.view(1, 1, size, size),
                                                "linear_attention": None}).past_key_values

        if _over_budget():
            return None
        _require_room()
        self._side_warmup(run)
        cache, graph = _capture_graph(run, self.pool)
        self.prefills[size] = (emb, pos, pad, cache, graph)
        print(f"[executor] captured a prefix graph for {size} tokens",
              file=sys.stderr, flush=True)
        return self.prefills[size]

    @torch.inference_mode()
    def _capture_branch(self, key, cache, pos_dtype, pos_lead):
        size, rows, wide = key
        dev = self.prefills[size][0].device
        ids = torch.full((rows, wide), self.pad_id, dtype=torch.long, device=dev)
        base = torch.zeros((), dtype=pos_dtype, device=dev)
        pad = torch.zeros((), dtype=torch.long, device=dev)

        def run():
            q = torch.arange(wide, device=dev)
            k = torch.arange(size + wide, device=dev)
            pos = (base + q).expand(*pos_lead[:-1], rows, wide)
            kk, qq = k.view(1, -1), q.view(-1, 1)
            mask = torch.where(kk < size, kk >= pad, kk - size <= qq)
            forked = self.fork(cache, rows)
            return self.decoder(input_ids=ids, position_ids=pos, past_key_values=forked,
                                use_cache=True, return_dict=True,
                                attention_mask={
                                    "full_attention": mask.view(1, 1, wide, size + wide)
                                    .expand(rows, 1, -1, -1),
                                    "linear_attention": None}).last_hidden_state

        if _over_budget():
            return None
        _require_room()
        self._side_warmup(run)
        out, graph = _capture_graph(run, self.pool)
        self.branches[key] = (ids, base, pad, out, graph)
        print(f"[executor] captured a branch graph for {rows} rows x {wide} tokens "
              f"over a {size}-token prefix (free {torch.cuda.mem_get_info()[0] >> 20} MiB)",
              file=sys.stderr, flush=True)
        return self.branches[key]

    @torch.inference_mode()
    def _capture_branch_multi(self, key, src, pos_dtype, pos_lead):
        size, rows, wide = key
        dev = self.prefills[size][0].device
        ids = torch.full((rows, wide), self.pad_id, dtype=torch.long, device=dev)
        bases = torch.zeros(rows, dtype=pos_dtype, device=dev)
        padv = torch.zeros(rows, dtype=torch.long, device=dev)
        idx = torch.zeros(rows, dtype=torch.long, device=dev)

        def run():
            q = torch.arange(wide, device=dev)
            k = torch.arange(size + wide, device=dev)
            pos = (bases[:, None] + q.view(1, -1)).expand(*pos_lead[:-1], rows, wide)
            kk, qq = k.view(1, -1), q.view(-1, 1)
            mask = torch.where(kk < size, kk >= padv[:, None, None], kk - size <= qq)
            forked = _gather_cache(src, idx)
            return self.decoder(input_ids=ids, position_ids=pos, past_key_values=forked,
                                use_cache=True, return_dict=True,
                                attention_mask={
                                    "full_attention": mask.view(rows, 1, wide, -1),
                                    "linear_attention": None}).last_hidden_state

        if _over_budget():
            return None
        _require_room()
        self._side_warmup(run)
        out, graph = _capture_graph(run, self.pool)
        self.multi[key] = (ids, bases, padv, idx, out, graph)
        print(f"[executor] captured a batched branch graph for {rows} rows x {wide} tokens "
              f"over {self.batch_max} prefixes of {size} tokens "
              f"(free {torch.cuda.mem_get_info()[0] >> 20} MiB)", file=sys.stderr, flush=True)
        return self.multi[key]

    @torch.inference_mode()
    def _run_prefill(self, size, embeds, positions):
        """Replay the prefix graph of one bucket with left padding; returns its cache buffer."""
        entry = self.prefills.get(size)
        if entry is None:
            return None
        emb, pos, pad, cache, graph = entry
        a = size - embeds.shape[1]
        emb[:, :a].zero_()
        emb[:, a:].copy_(embeds)
        pos[..., :a].zero_()
        pos[..., a:].copy_(positions)
        pad.fill_(a)
        graph.replay()
        return cache

    @torch.inference_mode()
    def run_groups(self, calls):
        """Branch every request of one batch in a single launch: calls carry
        (embeds, positions, suffixes, prefix_key, delta). Prefixes of different bucket sizes
        share the largest one while the extra padding stays under VERDICT_FORK_MERGE_PAD
        tokens; returns one hidden block per call, or None when the shape stays per-group."""
        if self.off or self.batch_max < 2 or len(calls) < 2 or len(calls) > self.batch_max:
            return None
        FORK_STATS["groups"] = float(len(calls))
        plans = []
        for embeds, positions, suffixes, pkey, delta in calls:
            n = len(suffixes)
            P = int(embeds.shape[1])
            size = next((x for x in self.caps if x >= P), None)
            rows = next((x for x in self.ROW_BUCKETS if x >= n), None)
            wide = _bucket(max(len(s) for s in suffixes), self.width_step)
            if size is None or rows is None or wide > self.max_width:
                return None
            plans.append((size, rows, wide, P, suffixes, pkey, embeds, positions, delta))
        size = max(p[0] for p in plans)
        if size * len(plans) - sum(p[3] for p in plans) > MERGE_PAD_LIMIT:
            FORK_STATS["multi"] = -4.0
            return None
        wide = max(p[2] for p in plans)
        total = sum(len(p[4]) for p in plans)
        rows_b = next((x for x in self.ROW_BUCKETS if x >= total), None)
        entry = self.multi.get((size, rows_b, wide)) if rows_b is not None else None
        src = self.src.get(size)
        if entry is None:
            FORK_STATS["multi"] = -2.0
            return None
        if src is None:
            FORK_STATS["multi"] = -3.0
            return None
        ids, bases, padv, idx, out, graph = entry
        t_pre, t_copy = 0.0, 0.0
        for k, p in enumerate(plans):
            snap = self._cached_state(p[5], size)
            if snap is None:
                _t0 = time.perf_counter()
                state = self._run_prefill(size, p[6], p[7])
                if state is None:
                    return None
                self._remember_state(p[5], size, state, p[3], p[8])
                t_pre += (time.perf_counter() - _t0) * 1e3
            else:
                state = snap[0]
            _t1 = time.perf_counter()
            _state_slot_copy(src, k, state)
            t_copy += (time.perf_counter() - _t1) * 1e3
        _t2 = time.perf_counter()
        host = np.full((rows_b, wide), self.pad_id, dtype="<i8")
        idx_np = np.zeros(rows_b, dtype="<i8")
        base_np = np.zeros(rows_b, dtype="<i8")
        pad_np = np.zeros(rows_b, dtype="<i8")
        ends = np.zeros(rows_b, dtype="<i8")
        r = 0
        for k, p in enumerate(plans):
            size_k, rows_k, wide_k, P, suffixes, pkey, embeds, positions, delta = p
            for s in suffixes:
                host[r, :len(s)] = np.asarray(s)
                idx_np[r] = k
                base_np[r] = P + delta
                pad_np[r] = size - P
                ends[r] = len(s) - 1
                r += 1
        ids.copy_(torch.from_numpy(host), non_blocking=True)
        idx.copy_(torch.from_numpy(idx_np), non_blocking=True)
        bases.copy_(torch.from_numpy(base_np), non_blocking=True)
        padv.copy_(torch.from_numpy(pad_np), non_blocking=True)
        t_asm = (time.perf_counter() - _t2) * 1e3
        _t3 = time.perf_counter()
        graph.replay()
        if MM_PROFILE:
            torch.cuda.synchronize()
        FORK_STATS["multi"] = 1.0
        FORK_STATS["prefix"] = t_pre
        FORK_STATS["copy"] = t_copy
        FORK_STATS["asm"] = t_asm
        FORK_STATS["branch"] = (time.perf_counter() - _t3) * 1e3
        rows_all = out[torch.arange(rows_b, device=DEV),
                       torch.from_numpy(ends).to(DEV)]
        blocks, start = [], 0
        for p in plans:
            k = len(p[4])
            blocks.append(rows_all[start:start + k])
            start += k
        return blocks

    @torch.inference_mode()
    def prewarm(self, pos_lead) -> None:
        """Capture the whole capacity x rows x width envelope, cheapest cells first so a device that runs out keeps the often-visited ones; a cell that does not fit stays eager."""
        w = self.decoder.get_input_embeddings().weight
        widths = sorted(x for x in self.prewarm_widths() if x <= self.max_width)
        keys = [(size, r, wide) for size in sorted(self.caps)
                for r in sorted(self.ROW_BUCKETS) for wide in widths]
        misses = 0
        for key in keys:
            if misses >= self.PREWARM_MISS_LIMIT:
                break
            size, r, wide = key
            try:
                if size not in self.prefills:
                    emb = torch.zeros((1, size, w.shape[1]), dtype=w.dtype, device=DEV)
                    pos = torch.zeros((*pos_lead, size), dtype=torch.long, device=DEV)
                    if self._capture_prefill(size, emb, pos) is None:
                        misses += 1
                        continue
                if self._capture_branch(key, self.prefills[size][3], torch.long,
                                        pos_lead) is None:
                    misses += 1
                    continue
                misses = 0
            except (GraphMemoryLow, torch.OutOfMemoryError, RuntimeError) as err:
                if _fatal_cuda(err):
                    raise
                misses += 1
                print(f"[executor] fork graph {size} x {r} rows x {wide} not prewarmed "
                      f"({type(err).__name__}: {err}); eager for the life of the process",
                      file=sys.stderr, flush=True)
        if self.batch_max >= 2:
            for size in sorted(self.prefills):
                if self.src.get(size) is None:
                    try:
                        self.src[size] = _fork(self.prefills[size][3], self.batch_max)
                    except (RuntimeError, torch.OutOfMemoryError) as err:
                        print(f"[executor] batched branch source {size} unavailable "
                              f"({type(err).__name__}: {err}); per-group branch stays",
                              file=sys.stderr, flush=True)
                        continue
                for r in sorted(x for x in self.ROW_BUCKETS if x >= 8):
                    for wide in widths:
                        key = (size, r, wide)
                        if key in self.multi:
                            continue
                        try:
                            if self._capture_branch_multi(key, self.src[size],
                                                          torch.long, pos_lead) is None:
                                break
                        except (GraphMemoryLow, torch.OutOfMemoryError, RuntimeError) as err:
                            if _fatal_cuda(err):
                                raise
                            print(f"[executor] batched branch {size} x {r} x {wide} not prewarmed "
                                  f"({type(err).__name__}: {err})",
                                  file=sys.stderr, flush=True)
            print(f"[executor] batched branch cells {len(self.multi)} "
                  f"(max {self.batch_max} requests per launch)",
                  file=sys.stderr, flush=True)
        print(f"[executor] fork graphs prewarmed prefixes={len(self.prefills)} "
              f"branches={len(self.branches)} of {len(keys)} cells",
              file=sys.stderr, flush=True)

    def run(self, embeds, positions, suffixes: list, delta: int, prefix_key=None,
            cached=None):
        """Graphs path: any failure turns the path off and the eager fork serves the rest."""
        if self.off:
            return None
        try:
            return self(embeds, positions, suffixes, delta, prefix_key, cached)
        except (GraphMemoryLow, torch.OutOfMemoryError) as err:
            print(f"[executor] fork graphs off after a replay {type(err).__name__}: "
                  f"{err!r}; this and later requests run the eager fork\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)
            self.off = True
            return None
        except Exception as err:
            print(f"[executor] fork graphs off after {type(err).__name__}: {err!r}\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)
            self.off = True
            return None

def _mm_vision(model, inputs):
    inner = model.model
    out = []
    for pix, grid, get in (
            ("pixel_values", "image_grid_thw", inner.get_image_features),
            ("pixel_values_videos", "video_grid_thw", inner.get_video_features)):
        out.append(torch.cat(get(inputs[pix], inputs[grid], return_dict=True).pooler_output, 0)
                   if pix in inputs else None)
    return out

def _mm_embed(model, inputs, vision=None):
    inner = model.model
    ids = inputs["input_ids"]
    _p0 = time.perf_counter()
    emb = inner.get_input_embeddings()(ids)
    feats_i, feats_v = vision if vision is not None else _mm_vision(model, inputs)
    for feats, key, which in ((feats_i, "image_features", 0), (feats_v, "video_features", 1)):
        if feats is not None:
            feats = feats.to(emb.device, emb.dtype)
            mask = inner.get_placeholder_mask(ids, inputs_embeds=emb, **{key: feats})[which]
            emb = emb.masked_scatter(mask, feats)
    if DEV == "cuda":
        torch.cuda.synchronize()
    _p1 = time.perf_counter()
    grids = {k: inputs.get(k) for k in ("image_grid_thw", "video_grid_thw")}
    if MM_PROFILE and grids.get("image_grid_thw") is not None:
        print(f"[mm-grid] image={grids['image_grid_thw'].tolist()} "
              f"pixels={int(inputs['pixel_values'].shape[0])}", file=sys.stderr, flush=True)
    delta = 0
    if any(v is not None for v in grids.values()):
        positions, deltas = inner.get_rope_index(
            ids, mm_token_type_ids=inputs["mm_token_type_ids"],
            attention_mask=inputs.get("attention_mask"), **grids)
        delta = int(deltas.view(-1)[0])
    else:
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, 1, -1)
    if DEV == "cuda":
        torch.cuda.synchronize()
    if MM_PROFILE:
        print(f"[mm-embed] vision={(_p1 - _p0) * 1e3:.1f}ms "
              f"rope={ (time.perf_counter() - _p1) * 1e3:.1f}ms", file=sys.stderr, flush=True)
    return emb, positions, delta
_FORK_REFUSED: set = set()

def _media_prefix_key(prefix_ids, media_key_parts):
    """blake2b over the unexpanded prefix ids plus each media item's content hash, every chunk length-framed so distinct inputs never share a key."""
    h = hashlib.blake2b(digest_size=16)
    raw = np.asarray(prefix_ids, dtype="<i8").tobytes()
    h.update(len(raw).to_bytes(8, "little"))
    h.update(raw)
    for part in media_key_parts:
        pb = part.encode("ascii")
        h.update(len(pb).to_bytes(8, "little"))
        h.update(pb)
    return h.digest()

def _refuse(reason: str, label: str = "media", fallback: str = "per-row"):
    if (label, reason) not in _FORK_REFUSED:
        _FORK_REFUSED.add((label, reason))
        print(f"[executor] {label} fork declined: {reason}; serving {fallback}",
              file=sys.stderr, flush=True)
    return None

def _state_tensors(cache):
    """Every tensor a decoder cache holds, attention K/V and conv/recurrent states alike."""
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if torch.is_tensor(value):
                yield value
        for name in ("conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            if isinstance(value, dict):
                for item in value.values():
                    if torch.is_tensor(item):
                        yield item

def _copy_state(src, dst) -> None:
    """Copy a cached prefix state into a live prefix buffer of the same shape."""
    left, right = list(_state_tensors(src)), list(_state_tensors(dst))
    if len(left) != len(right):
        raise RuntimeError(f"cache holds {len(left)} tensors, buffer {len(right)}")
    for a, b in zip(left, right):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(
                f"cache tensor {tuple(a.shape)} does not fit buffer {tuple(b.shape)}")
        b.copy_(a)

def _fork(cache, n: int):
    """Duplicate a batch-1 KV cache into n rows, conv/recurrent states included (DynamicCache.batch_repeat_interleave does not fork those in transformers 5.17)."""
    forked = copy.deepcopy(cache)
    layers = getattr(forked, "layers", None)
    if layers is None:
        raise RuntimeError(f"unsupported cache object {type(cache).__name__}")
    for layer in layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(n)
        elif hasattr(layer, "conv_states") and hasattr(layer, "recurrent_states"):
            for states in (layer.conv_states, layer.recurrent_states):
                for k, v in list(states.items()):
                    if torch.is_tensor(v):
                        states[k] = v.repeat_interleave(n, dim=0)
        else:
            raise RuntimeError(f"cannot fork cache layer {type(layer).__name__}")
    return forked

def _gather_cache(src, idx):
    """A cache of len(idx) rows, row i holding src[idx[i]]; rows of one request share that
    request's prefix state, so several requests can branch in one launch."""
    out = copy.deepcopy(src)
    layers = getattr(out, "layers", None)
    if layers is None:
        raise RuntimeError(f"unsupported cache object {type(out).__name__}")
    for layer in layers:
        if hasattr(layer, "conv_states") and hasattr(layer, "recurrent_states"):
            for states in (layer.conv_states, layer.recurrent_states):
                for k, v in list(states.items()):
                    if torch.is_tensor(v):
                        states[k] = v.index_select(0, idx)
        elif hasattr(layer, "batch_select_indices"):
            layer.batch_select_indices(idx)
        else:
            raise RuntimeError(f"cannot gather cache layer {type(layer).__name__}")
    return out


def _state_slot_copy(dst, slot: int, src) -> None:
    """Copy a single-row prefix state into one slot of a batched state buffer."""
    left, right = list(_state_tensors(dst)), list(_state_tensors(src))
    if len(left) != len(right):
        raise RuntimeError(f"cache holds {len(left)} tensors, buffer {len(right)}")
    for a, b in zip(left, right):
        if a.shape[1:] != b.shape[1:] or a.dtype != b.dtype:
            raise RuntimeError(f"cache tensor {tuple(b.shape)} does not fit slot {tuple(a.shape)}")
        a[slot].copy_(b[0])


def _mm_fork_forward(model, processor, tokenizer, letter_ids, head_w, pad_id,
                     ids, mask, texts, load_media, vision_ids, counts, fork_graphs=None,
                     row_media_keys=None, prefix_lens=None):
    """One processor pass and one prefix prefill, then every suffix as a batched branch off forked caches; None when the rows do not split.

    The prefix state depends on the media, not just the tokens: rows whose media differ never fork together, and the prefix-cache key mixes the media content hashes into the unexpanded prefix ids. A probe hit returns before load_media() runs: no decode, no processor, no tower, no prefill."""
    b = ids.shape[0]
    real_lens = mask.sum(1).tolist()
    rows_ids = [ids[i, :real_lens[i]].tolist() for i in range(b)]
    vs, v_pads, ve = vision_ids
    vision_set = {vs, ve, *v_pads}
    if row_media_keys is None:
        row_media_keys = [()] * b
    if any(k != row_media_keys[0] for k in row_media_keys[1:]):
        note_uncovered("fork_na", b)
        return _refuse("rows carry different media")

    if b == 1:
        if fork_graphs is None or fork_graphs.off or not prefix_lens:
            return None
        n = int(prefix_lens[0])
        if n <= 0 or n >= real_lens[0]:
            return None
        suffixes = [rows_ids[0][n:]]
        if any(t in vision_set for t in suffixes[0]):
            return _refuse("a vision token sits in a branch suffix")
        snap = fork_graphs.probe(
            _media_prefix_key(rows_ids[0][:n], row_media_keys[0]), n)
        if snap is None:
            return None
        with torch.inference_mode():
            last = fork_graphs.run(None, None, suffixes, 0, cached=snap)
        if last is None:
            note_uncovered("fork_degraded", 1)
            return None
        return (last.float() @ head_w.T), [snap[1] + len(suffixes[0])], snap[1]

    n = min(real_lens)
    for other in rows_ids[1:]:
        m = 0
        for a, c in zip(rows_ids[0], other):
            if a != c:
                break
            m += 1
        n = min(n, m)
    suffixes = [r[n:] for r in rows_ids]
    if n <= 0 or any(len(s) == 0 for s in suffixes):
        note_uncovered("fork_na", b)
        return _refuse("no shared token prefix across rows")
    if any(t in vision_set for s in suffixes for t in s):
        return _refuse("a vision token sits in a branch suffix")

    pkey = None
    if fork_graphs is not None and not fork_graphs.off:
        pkey = _media_prefix_key(rows_ids[0][:n], row_media_keys[0])
        snap = fork_graphs.probe(pkey, n)
        if snap is not None:
            with torch.inference_mode():
                last = fork_graphs.run(None, None, suffixes, 0, cached=snap)
            if last is not None:
                return ((last.float() @ head_w.T),
                        [snap[1] + len(s) for s in suffixes], snap[1])
            note_uncovered("fork_degraded", b)

    items0 = load_media()[: counts[0]]
    _tp = time.perf_counter()
    images = [x for x in items0 if not isinstance(x, tuple)]
    videos, metas = [], []
    for x in items0:
        if isinstance(x, tuple):
            videos.append(x[0])
            metas.append(x[1])
    out = processor(text=[texts[0]], return_tensors="pt",
                    **_processor_kwargs(images, videos, metas))
    expanded0 = out["input_ids"][0].tolist()
    P = n + len(expanded0) - len(rows_ids[0])
    if P <= 0 or expanded0[P:] != suffixes[0]:
        return _refuse("the expanded prompt does not split at row 0's suffix")
    FORK_STATS.clear()
    FORK_STATS["proc"] = (time.perf_counter() - _tp) * 1e3

    inputs = {"input_ids": out["input_ids"][:, :P],
              "attention_mask": out["attention_mask"][:, :P]}
    if "mm_token_type_ids" in out:
        inputs["mm_token_type_ids"] = out["mm_token_type_ids"][:, :P]
    for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if k in out:
            inputs[k] = out[k]
    inputs = {k: v.to(DEV) for k, v in inputs.items()}
    width = max(len(s) for s in suffixes)
    fork_ready = (fork_graphs is not None and not fork_graphs.off
                  and bool(inputs["attention_mask"].all()))
    if fork_ready and fork_graphs.plan(P, b, width) is not None:
        with torch.inference_mode():
            _te = time.perf_counter()
            emb, rope, rope_delta = _mm_embed(model, inputs)
            FORK_STATS["embed"] = (time.perf_counter() - _te) * 1e3
            last = fork_graphs.run(emb, rope, suffixes, rope_delta, pkey)
            logits = None if last is None else (last.float() @ head_w.T)
        if logits is not None:
            return logits, [P + len(s) for s in suffixes], P
        note_uncovered("fork_degraded", b)
    elif fork_ready:
        note_uncovered(fork_graphs.why(P, b, width), b)
    grids = {k: inputs.get(k) for k in ("image_grid_thw", "video_grid_thw")}
    if any(v is not None for v in grids.values()):
        positions, deltas = model.model.get_rope_index(
            inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
            attention_mask=inputs["attention_mask"], **grids)
        delta = int(deltas.view(-1)[0])
    else:
        positions = torch.arange(P, device=DEV).view(1, 1, -1).expand(3, 1, -1)
        delta = 0

    suf_np = np.full((b, width), pad_id, dtype="<i8")
    mask_np = np.zeros((b, P + width), dtype="<i8")
    ends = []
    for row, s in enumerate(suffixes):
        suf_np[row, : len(s)] = np.asarray(s)
        mask_np[row, : P + len(s)] = 1
        ends.append(len(s) - 1)
    suf_ids = torch.from_numpy(suf_np).to(DEV)
    full_mask = torch.from_numpy(mask_np).to(DEV)
    pos = (torch.arange(P, P + width, device=DEV) + delta).view(1, 1, -1).expand(3, b, -1)
    if bool(inputs["attention_mask"].all()):
        del inputs["attention_mask"]
    branch = {"input_ids": suf_ids, "position_ids": pos.to(positions.dtype),
              "use_cache": True, "return_dict": True}
    if not bool(full_mask.all()):
        branch["attention_mask"] = full_mask

    with torch.inference_mode():
        _t0 = time.perf_counter()
        base = model.model(**inputs, position_ids=positions,
                           use_cache=True, return_dict=True).past_key_values
        if DEV == "cuda":
            torch.cuda.synchronize()
        _t1 = time.perf_counter()
        try:
            cache = _fork(base, b)
            _t2 = time.perf_counter()
            o = model.model(past_key_values=cache, **branch)
            if DEV == "cuda":
                torch.cuda.synchronize()
            _t3 = time.perf_counter()
            FORK_STATS["prefill"] = (_t1 - _t0) * 1e3
            FORK_STATS["fork"] = (_t2 - _t1) * 1e3
            FORK_STATS["branch"] = (_t3 - _t2) * 1e3
            if MM_PROFILE:
                print(f"[mm-fork] prefill={(_t1 - _t0) * 1e3:.1f}ms "
                      f"fork={(_t2 - _t1) * 1e3:.1f}ms branch={(_t3 - _t2) * 1e3:.1f}ms "
                      f"P={P} width={width} rows={b}", file=sys.stderr, flush=True)
        finally:
            del base
    last = o.last_hidden_state[torch.arange(b, device=DEV), torch.tensor(ends, device=DEV)]
    logits = (last.float() @ head_w.T)
    expanded_lens = [P + len(s) for s in suffixes]
    return logits, expanded_lens, P

def _mm_fork_selfcheck(model, processor, tokenizer, letter_ids, head_w, pad_id,
                       tolerance=5e-2, fork_graphs=None):
    """Startup gate: the fork must match the whole-prompt softmax within tolerance or the path stays off."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 240), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 140, 100], fill=(200, 40, 40))
    d.ellipse([180, 120, 300, 220], fill=(40, 80, 200))
    text_a = ("<|im_start|>system\nApply the question to the state. Choose exactly one of the "
              "listed options. Respond with only its uppercase letter, with no explanation or "
              "reasoning.<|im_end|>\n<|im_start|>user\n<state>\n<|vision_start|><|image_pad|>"
              "<|vision_end|>\nred box\n</state>\n\nQuestion: is there a red box?\n\nOptions:\n"
              "A. yes\nB. no\n\nAnswer with one letter: A, B.<|im_end|>\n<|im_start|>assistant\n"
              "<think>\n\n</think>\n\n")
    text_b = text_a.replace(
        "Question: is there a red box?\n\nOptions:\nA. yes\nB. no",
        "Question: shape?\n\nOptions:\nA. circle\nB. box\nC. none")
    vids = [tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    if None in vids:
        return False, "tokenizer lacks vision tokens"

    def enc(t):
        return tokenizer(t, add_special_tokens=False).input_ids

    rows = [enc(text_a), enc(text_b)]
    L = max(len(r) for r in rows)
    ids_t = torch.full((2, L), pad_id, dtype=torch.long)
    mask = torch.zeros((2, L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids_t[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    loaded = [_load_image_from_pil(img)]
    vision_ids = (vids[0], (vids[1], vids[1]), vids[2])
    row_keys = [("selfcheck-image",)] * 2

    forked = _mm_fork_forward(model, processor, tokenizer, letter_ids, head_w,
                              pad_id, ids_t.to(DEV), mask.to(DEV), [text_a, text_b],
                              lambda: loaded, vision_ids, [1, 1], fork_graphs,
                              row_keys)
    if forked is None:
        return False, "fork refused the synthetic request"
    fork_logits, _, _ = forked
    if fork_graphs is not None and not fork_graphs.off:
        forked = _mm_fork_forward(model, processor, tokenizer, letter_ids, head_w,
                                  pad_id, ids_t.to(DEV), mask.to(DEV),
                                  [text_a, text_b], lambda: loaded, vision_ids,
                                  [1, 1], fork_graphs, row_keys)
        if forked is None:
            return False, "fork refused the cached synthetic request"
        hit_logits, _, _ = forked
        if not torch.equal(hit_logits, fork_logits):
            return False, "the prefix-cache hit diverged from the miss"
    whole = []
    with torch.no_grad():
        for text in (text_a, text_b):
            out = processor(text=[text], return_tensors="pt",
                            **_processor_kwargs(loaded, [], []))
            inputs = {k: v.to(DEV) for k, v in out.items()}
            o = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)
            whole.append(o.logits[0, -1][letter_ids].float())
    fa = torch.softmax(fork_logits[0, :2].double(), -1).tolist()
    wa = torch.softmax(whole[0][:2].double(), -1).tolist()
    fb = torch.softmax(fork_logits[1, :3].double(), -1).tolist()
    wb = torch.softmax(whole[1][:3].double(), -1).tolist()
    split = max(abs(x - y) for pair in ((fa, wa), (fb, wb)) for x, y in zip(*pair))
    ok = split <= tolerance
    return ok, f"max diff {split:.2e} < tol {tolerance:.0e}"

def _fork_one_group(model, head_w, pad_id, ids, n, suffixes, fork_graphs,
                    prefix_bytes=b""):
    """One prefix prefill, then the group's suffixes as a batched branch off forked caches; the flag says whether a fork graph answered."""
    b = len(suffixes)
    width = max(len(s) for s in suffixes)
    if fork_graphs is not None and not fork_graphs.off:
        if fork_graphs.plan(n, b, width) is not None:
            with torch.inference_mode():
                emb = model.get_input_embeddings()(ids[0, :n]).unsqueeze(0)
                pos = torch.arange(n, device=DEV).view(1, -1)
                pkey = hashlib.blake2b(prefix_bytes, digest_size=16).digest()
                last = fork_graphs.run(emb, pos, suffixes, 0, pkey)
                logits = None if last is None else (last.float() @ head_w.T)
            if logits is not None:
                return logits, True
            note_uncovered("fork_degraded", b)
        else:
            note_uncovered(fork_graphs.why(n, b, width), b)
    suf_np = np.full((b, width), pad_id, dtype="<i8")
    mask_np = np.zeros((b, n + width), dtype="<i8")
    for row, s in enumerate(suffixes):
        suf_np[row, : len(s)] = s
        mask_np[row, : n + len(s)] = 1
    suf_ids = torch.from_numpy(suf_np).to(DEV)
    full_mask = torch.from_numpy(mask_np).to(DEV)
    pos = torch.arange(n, n + width, device=DEV).view(1, -1).expand(b, -1)
    branch = {"input_ids": suf_ids, "position_ids": pos,
              "use_cache": True, "return_dict": True}
    if not bool(full_mask.all()):
        branch["attention_mask"] = full_mask
    with torch.inference_mode():
        base = model.model(input_ids=ids[0, :n].unsqueeze(0),
                           use_cache=True, return_dict=True).past_key_values
        try:
            cache = _fork(base, b)
            o = model.model(past_key_values=cache, **branch)
        finally:
            del base
    ends = torch.tensor([len(s) - 1 for s in suffixes], device=DEV)
    last = o.last_hidden_state[torch.arange(b, device=DEV), ends]
    return (last.float() @ head_w.T), False

def _text_fork_forward(model, runner, head_w, pad_id, ids, mask, slot_idx,
                       fork_graphs=None, prefix_lens=None,
                       ids_h=None, mask_h=None, slot_h=None):
    """Fork every row at its declared state prefix (or at the batch-wide LCP when none is declared), run the rest whole; None when nothing splits. Row bookkeeping runs on the host arrays so the batch never pays device syncs."""
    b = ids.shape[0]
    FORK_STATS["multi"] = 0.0
    if ids_h is None:
        ids_h, mask_h, slot_h = ids.cpu().numpy(), mask.cpu().numpy(), slot_idx.cpu().numpy()
    real_lens = mask_h.sum(1)
    floor = TEXT_FORK_MIN_PREFIX
    groups = {}
    rest = []
    splits = []
    for i in range(b):
        declared = prefix_lens[i] if prefix_lens is not None else 0
        if int(slot_h[i]) != real_lens[i] - 1:
            rest.append(i)
            continue
        if declared >= floor and declared < real_lens[i]:
            groups.setdefault(("declared", ids_h[i, :declared].tobytes(), declared), []).append(i)
            continue
        splits.append(i)
    for i in splits:
        if real_lens[i] <= floor:
            rest.append(i)
            continue
        key = ("lcp", ids_h[i, :floor].tobytes(), 0)
        groups.setdefault(key, []).append(i)
    out = [None] * b
    prepared = []
    for key, members in groups.items():
        n = key[2]
        if key[0] == "lcp":
            if len(members) < 2:
                rest.extend(members)
                continue
            mlen = min(real_lens[i] for i in members)
            eq = (ids_h[members, :mlen] == ids_h[members[0], :mlen]).all(0)
            nz = np.flatnonzero(~eq)
            n = int(nz[0]) if nz.size else int(mlen)
        suffixes = [ids_h[i, n:int(real_lens[i])] for i in members]
        if n < floor or any(len(s) == 0 for s in suffixes):
            rest.extend(members)
            continue
        prepared.append((members, n, suffixes))
    if fork_graphs is not None and not fork_graphs.off and len(prepared) > 1:
        calls = []
        for members, n, suffixes in prepared:
            emb = model.get_input_embeddings()(ids[members[0]:members[0] + 1, :n])
            pos = torch.arange(n, device=DEV).view(1, -1)
            pkey = hashlib.blake2b(ids_h[members[0], :n].tobytes(), digest_size=16).digest()
            calls.append((emb, pos, suffixes, pkey, 0))
        with torch.inference_mode():
            blocks = fork_graphs.run_groups(calls)
        if blocks is not None:
            for (members, n, suffixes), last in zip(prepared, blocks):
                logits = last.float() @ head_w.T
                for r, i in enumerate(members):
                    out[i] = logits[r]
            prepared = []
    for members, n, suffixes in prepared:
        logits, _ = _fork_one_group(model, head_w, pad_id,
                                    ids[members[0]:members[0] + 1], n, suffixes,
                                    fork_graphs,
                                    ids_h[members[0], :n].tobytes())
        for r, i in enumerate(members):
            out[i] = logits[r]
    if not any(x is not None for x in out):
        return _refuse("no shared token prefix across rows", "text", "whole-batch")
    if rest:
        note_uncovered("fork_na", len(rest))
        _rest0 = time.perf_counter()
        idx = torch.tensor(rest, device=DEV)
        sub_slot = slot_idx[idx]
        if runner is not None:
            sub_logits, why = runner.run(ids[idx], mask[idx],
                                         torch.arange(len(rest), device=DEV),
                                         sub_slot)
            note_uncovered(why, len(rest))
        else:
            kw = {} if bool(mask[idx].all()) else {"attention_mask": mask[idx]}
            with torch.inference_mode():
                hs = model.model(input_ids=ids[idx], use_cache=False,
                                 **kw).last_hidden_state
                sub_logits = hs[torch.arange(len(rest), device=DEV),
                                sub_slot].float() @ head_w.T
        for r, i in enumerate(rest):
            out[i] = sub_logits[r].float()
        PHASE_STATS["rest"] = (time.perf_counter() - _rest0) * 1e3
    return torch.stack(out)

def _text_fork_selfcheck(model, tokenizer, letter_ids, head_w, pad_id,
                         tolerance=5e-2, fork_graphs=None):
    """Startup gate: the fork must match the whole-prompt softmax within tolerance or the path stays off."""
    prefix = ("Context:\nThe weekly log covers meals, budget, weather, "
              "commute, and mood for seven days. " * 9)
    text_a = (prefix + "\n\nQuestion: Dinner tonight?\nOptions:"
                       "\n(A) delivery\n(B) pasta\n(C) skip\nAnswer: (")
    text_b = (prefix + "\n\nQuestion: Lunch tomorrow?\nOptions:"
                       "\n(A) salad\n(B) sandwich\nAnswer: (")

    def enc(t):
        return tokenizer(t, add_special_tokens=False).input_ids

    rows = [enc(text_a), enc(text_b)]
    L = max(len(r) for r in rows)
    ids = torch.full((2, L), pad_id, dtype=torch.long)
    mask = torch.zeros((2, L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    ids, mask = ids.to(DEV), mask.to(DEV)
    slot_idx = torch.tensor([len(r) - 1 for r in rows], device=DEV)
    fork_logits = _text_fork_forward(model, None, head_w, pad_id, ids, mask,
                                     slot_idx, fork_graphs)
    if fork_logits is None:
        return False, "fork refused the synthetic request"
    fork_logits = _text_fork_forward(model, None, head_w, pad_id, ids, mask,
                                     slot_idx, fork_graphs)
    if fork_logits is None:
        return False, "fork refused the cached synthetic request"
    whole = []
    with torch.no_grad():
        for r in rows:
            o = model(input_ids=torch.tensor([r], device=DEV), use_cache=False,
                      return_dict=True, logits_to_keep=1)
            whole.append(o.logits[0, -1][letter_ids].float())
    fa = torch.softmax(fork_logits[0, :3].double(), -1).tolist()
    wa = torch.softmax(whole[0][:3].double(), -1).tolist()
    fb = torch.softmax(fork_logits[1, :2].double(), -1).tolist()
    wb = torch.softmax(whole[1][:2].double(), -1).tolist()
    split = max(abs(x - y) for pair in ((fa, wa), (fb, wb)) for x, y in zip(*pair))
    ok = split <= tolerance
    return ok, f"max diff {split:.2e} < tol {tolerance:.0e}"

def _text_fork_multi_selfcheck(model, tokenizer, letter_ids, head_w, pad_id,
                               fork_graphs=None, tolerance=5e-2):
    """Startup gate for the batched branch: three prompts with distinct prefixes in one
    launch must match the whole-prompt softmax."""
    heads = [f"Scenario {k}: " * (8 + 3 * k) + "The weekly log covers meals, budget, "
             "weather, commute, and mood. " for k in range(3)]
    texts = [heads[k] + f"\nQuestion {k}.{j}? Options:\n(A) one\n(B) two\n(C) three\n"
             "Answer: (" for k in range(3) for j in range(3)]

    def enc(t):
        return tokenizer(t, add_special_tokens=False).input_ids

    rows = [enc(t) for t in texts]
    L = max(len(r) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    ids, mask = ids.to(DEV), mask.to(DEV)
    slot_idx = torch.tensor([len(r) - 1 for r in rows], device=DEV)
    declared = [len(enc(heads[k])) for k in range(3) for _ in range(3)]
    got = _text_fork_forward(model, None, head_w, pad_id, ids, mask, slot_idx,
                             fork_graphs, prefix_lens=declared)
    if got is None:
        return False, "batched fork refused the synthetic batch"
    whole = []
    with torch.no_grad():
        for r in rows:
            o = model(input_ids=torch.tensor([r], device=DEV), use_cache=False,
                      return_dict=True, logits_to_keep=1)
            whole.append(o.logits[0, -1][letter_ids].float())
    split = 0.0
    for i in range(len(rows)):
        a = torch.softmax(got[i, :3].double(), -1).tolist()
        b = torch.softmax(whole[i][:3].double(), -1).tolist()
        split = max(split, max(abs(x - y) for x, y in zip(a, b)))
    return split <= tolerance, f"max diff {split:.2e} < tol {tolerance:.0e}"

def _install_tower_hook(model):
    tower = getattr(model.model, "visual", None)
    if tower is None:
        return None
    acc = [0.0, 0.0]
    state = {}

    def pre(_module, _args):
        if DEV == "cuda":
            torch.cuda.synchronize()
        state["t"] = time.perf_counter()

    def post(_module, _args, _out):
        if DEV == "cuda":
            torch.cuda.synchronize()
        acc[1] += (time.perf_counter() - state.get("t", time.perf_counter())) * 1000.0

    tower.register_forward_pre_hook(pre)
    tower.register_forward_hook(post)
    return acc

def _i64h(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<i8").copy()


def _i64(b64):
    """base64 little-endian int64 vector -> device tensor (.copy() makes the buffer writable for torch)."""
    return torch.from_numpy(
        np.frombuffer(base64.b64decode(b64), dtype="<i8").copy()
    ).to(DEV)

def _check_expanded(rust_ids, expanded, vs_id, pad_ids, ve_id, decode):
    """Compare the engine's unexpanded ids with the processor's expansion; a video expands into several blocks."""
    segs, kinds, cur = [], [], []
    i = 0
    while i < len(rust_ids):
        if (
            rust_ids[i] == vs_id
            and i + 2 < len(rust_ids)
            and rust_ids[i + 1] in pad_ids
            and rust_ids[i + 2] == ve_id
        ):
            segs.append(cur)
            kinds.append(rust_ids[i + 1])
            cur = []
            i += 3
        else:
            cur.append(rust_ids[i])
            i += 1
    segs.append(cur)

    def locate(haystack, needle, start):
        """First index >= start where the subsequence matches, else None."""
        if not needle:
            return start
        for p in range(start, len(haystack) - len(needle) + 1):
            if haystack[p:p + len(needle)] == needle:
                return p
        return None

    specials = {vs_id, ve_id, *pad_ids}
    pos = 0
    for idx, (seg, kind) in enumerate(zip(segs, kinds)):
        if expanded[pos:pos + len(seg)] != seg:
            raise ValueError("expansion diverges before a media block")
        pos += len(seg)
        nxt = segs[idx + 1]
        if not nxt:
            raise ValueError("adjacent media blocks without interleaved text")
        end = locate(expanded, nxt, pos)
        if end is None:
            raise ValueError("the text after a media block is missing")
        middle = expanded[pos:end]
        stamps = [t for t in middle if t not in specials]
        text = decode(stamps).strip() if stamps else ""
        if text and not re.fullmatch(r"(<\d+(\.\d+)? seconds>\s*)+", text):
            raise ValueError(f"unexpected text inside a media block: {text[:60]!r}")
        if not any(t == kind for t in middle):
            raise ValueError("a media block expanded without its placeholder pads")
        if any(t in middle for t in pad_ids if t != kind):
            raise ValueError("a media block expanded with mixed placeholder kinds")
        pos = end
    if expanded[pos:] != segs[-1]:
        raise ValueError("expansion diverges after the last media block")

def _load_image(item):
    """One RGB PIL image from a {format, data_b64} asset, decoded once per batch item."""
    from PIL import Image

    raw = base64.b64decode(item["data_b64"])
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB")

def _load_image_from_pil(img):
    """Passthrough loader for already-decoded PIL images (self-check path)."""
    return img.convert("RGB")

def _load_video(item):
    """(frames, VideoMetadata) from caller-extracted frames at the caller's fps (default 1.0)."""
    from transformers.video_utils import VideoMetadata

    items = item["frames"]
    if len(items) == 1:
        frames = [_load_image(items[0])]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
            frames = list(pool.map(_load_image, items))
    n = len(frames)
    fps = float(item.get("fps") or 1.0)
    return frames, VideoMetadata(
        total_num_frames=n, fps=fps, frames_indices=list(range(n)), duration=n / fps
    )

def _processor_kwargs(images, videos, metas):
    """The official media.processor_kwargs: video metadata plus the sampling switches."""
    kw = {}
    if images:
        kw["images"] = images
    if videos:
        kw["videos"] = videos
        kw["video_metadata"] = metas
        kw["videos_kwargs"] = {"do_sample_frames": False, "cap_pixels_per_frame": True}
    return kw

def _normalize_on_gpu(processor):
    """Rescale and normalize on the GPU; resize stays on the CPU because the GPU kernel rounds differently."""
    for proc in (getattr(processor, "image_processor", None),
                 getattr(processor, "video_processor", None)):
        if proc is None or not hasattr(proc, "rescale_and_normalize"):
            continue
        original = proc.rescale_and_normalize

        def on_gpu(images, *args, _original=original, **kwargs):
            return _original(images.to(DEV, non_blocking=True), *args, **kwargs)

        proc.rescale_and_normalize = on_gpu

def _f32_bytes(t):
    """f32 tensor -> little-endian bytes for the wire."""
    return t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()

def _f32_b64(t):
    """f32 tensor -> base64 little-endian bytes for the wire."""
    return base64.b64encode(_f32_bytes(t)).decode("ascii")

def derive_letter_ids(tok):
    """The letter-readout label table; must match src/tokenize.rs derive_letter_table."""
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

BRANCH_NOT_ONE_TOKEN = {
    "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV",
    "FZ", "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN",
    "JQ",
}
BRANCH_MAX_SLOTS = 256

def derive_branch_ids(tok):
    """The letter-branches slot table; must match the Rust rule."""
    def one(label):
        enc = tok.encode(label, add_special_tokens=False)
        if len(enc) == 1 and tok.decode(enc) == label:
            return enc[0]
        return None

    letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    table = []
    for label in letters:
        i = one(label)
        if i is None:
            raise RuntimeError(
                f"answer slot {label!r} is not a single round-trip token; the "
                f"letter-branches family requires all 26 letters"
            )
        table.append(i)
    pairs = []
    for a in letters:
        for b in letters:
            if len(pairs) >= BRANCH_MAX_SLOTS - len(letters):
                break
            pair = a + b
            if pair not in BRANCH_NOT_ONE_TOKEN:
                pairs.append(pair)
    extra = [one(p) for p in pairs]
    if all(i is not None for i in extra):
        table += extra
    if len(set(table)) != len(table):
        raise RuntimeError("two option labels tokenize to the same id")
    return table

def load_manifest_arg(ckpt):
    """Family/pad_token source: argv[3] (the normalized manifest) wins, else the raw model.json (v1 shape only)."""
    if len(sys.argv) > 3:
        return json.loads(sys.argv[3])
    path = os.path.join(ckpt, "model.json")
    if os.path.isfile(path):
        manifest = json.load(open(path))
        if manifest.get("manifest_version") == 2:
            raise RuntimeError(
                "torch_exec got a v2 model.json without the normalized manifest "
                "argument (argv[3]); run through `verdict serve` or pass the "
                "flattened manifest JSON"
            )
        return manifest
    return {}

def main():
    device = sys.argv[1] if len(sys.argv) > 2 else "cuda"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]
    if device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({
            "error": "the torch runtime was started with device cuda: "
                     "torch.cuda.is_available() is False on this host"
        }), flush=True)
        sys.exit(2)
    global DEV
    DEV = dev = "cuda" if device == "cuda" else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    t0 = time.time()
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(ckpt)
    cls = AutoModelForCausalLM
    text_only = getattr(config, "vision_config", None) is None
    if not text_only:
        from transformers import AutoModelForImageTextToText
        cls = AutoModelForImageTextToText
    attn_impl = "sdpa"
    if dev == "cuda" and text_only and os.environ.get("VERDICT_PACKED", "0") == "1":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            print("[executor] VERDICT_PACKED=1 but flash_attn is not installed; "
                  "staying on the padded path", file=sys.stderr, flush=True)
    model = cls.from_pretrained(
        ckpt, dtype=dtype, attn_implementation=attn_impl
    ).to(dev)
    if dev == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    model.eval()
    tok_dir = os.path.join(ckpt, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_dir if os.path.isfile(os.path.join(tok_dir, "tokenizer.json")) else ckpt
    )
    manifest = load_manifest_arg(ckpt)
    family = manifest.get("family", "letter_readout")
    if family == "letter_branches":
        table = derive_branch_ids(tokenizer)
    else:
        table = derive_letter_ids(tokenizer)
    letter_ids = torch.tensor(table, dtype=torch.long, device=dev)
    head_w = model.get_output_embeddings().weight[letter_ids].float()

    pad_name = manifest.get("pad_token", "<|endoftext|>")
    pad_id = tokenizer.convert_tokens_to_ids(pad_name)
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    warmup_ids = torch.ones(1, 4, dtype=torch.long, device=dev)
    warmup_mask = torch.ones(1, 4, dtype=torch.long, device=dev)
    if dev == "cuda":
        make_gdn_layers_opaque(model)
    with torch.no_grad():
        _ = model.model(input_ids=warmup_ids, attention_mask=warmup_mask)

    processor = None
    vision_ids = ()
    TOWER_ACC = None
    if getattr(config, "vision_config", None) is not None:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(ckpt, tokenizer=tokenizer)
        names = {
            "vs": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "image_pad": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            "video_pad": tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            "ve": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        }
        if None in names.values():
            raise RuntimeError(
                "the vision checkpoint's tokenizer lacks the vision special tokens"
            )
        vision_ids = (names["vs"], (names["image_pad"], names["video_pad"]), names["ve"])
        if dev == "cuda":
            _normalize_on_gpu(processor)
        if MM_PROFILE:
            TOWER_ACC = _install_tower_hook(model)
            print(f"[mm-profile] tower hook: {'on' if TOWER_ACC else 'NOT FOUND'}",
                  file=sys.stderr, flush=True)

    graphs_on = dev == "cuda" and os.environ.get("VERDICT_TORCH_GRAPHS", "1") != "0"
    compile_on = dev == "cuda" and os.environ.get("VERDICT_TORCH_COMPILE", "1") != "0"
    if compile_on:
        from torch import _inductor

        _inductor.config.combo_kernels = True
        _inductor.config.benchmark_combo_kernel = True
    cc_dir = compile_cache_dir(manifest, ckpt) if compile_on else None
    cc_hit = bool(cc_dir) and any(
        n != "key.json" for n in os.listdir(cc_dir))
    if not cc_dir and dev == "cuda" and not compile_on:
        print("[executor] torch.compile off (VERDICT_TORCH_COMPILE=0)",
              file=sys.stderr, flush=True)
    model_max_len = int(manifest.get("max_len", 32768))
    length_cap = min(GRAPH_MAX_LEN, max(384, model_max_len // 8))
    media_sizes = [s for s in GraphRunner._l_buckets(model_max_len, GRAPH_MAX_LEN, 5, False)
                   if s >= 128]
    t_warm = time.time()

    def build_runner(compiled: bool):
        return GraphRunner(
            model,
            head_w,
            pad_id,
            max_len=model_max_len,
            compile_fwd=compiled,
        )

    runner = None
    if graphs_on:
        try:
            runner = build_runner(compile_on)
        except Exception as err:
            if not compile_on:
                raise
            print(f"[executor] the compiled forward failed to build "
                  f"({type(err).__name__}: {err!r}); wiping the cache and "
                  "recapturing uncompiled", file=sys.stderr, flush=True)
            if cc_dir:
                shutil.rmtree(cc_dir, ignore_errors=True)
            cc_dir = None
            runner = build_runner(False)
        else:
            if compile_on:
                ok, note = runner.compile_selfcheck()
                if not ok:
                    print(f"[executor] compile self-check FAILED ({note}); "
                          "wiping the cache and recapturing uncompiled",
                          file=sys.stderr, flush=True)
                    if cc_dir:
                        shutil.rmtree(cc_dir, ignore_errors=True)
                    cc_dir = None
                    runner = None
                    torch.cuda.empty_cache()
                    runner = build_runner(False)

    media_graphs = None
    if processor is not None and dev == "cuda" and MM_GRAPHS[0]:
        media_graphs = MediaGraphs(model.model, media_sizes)

    fork_graphs = None
    if (processor is not None and dev == "cuda" and MM_GRAPHS[0]
            and os.environ.get("VERDICT_FORK_GRAPHS", "1") != "0"):
        fork_graphs = ForkGraphs(model.model.language_model, _fork, pad_id,
                                 compile_fwd=compile_on)

    text_fork_graphs = None
    if (processor is None and graphs_on
            and os.environ.get("VERDICT_FORK_GRAPHS", "1") != "0"):
        text_fork_graphs = ForkGraphs(model.model, _fork, pad_id,
                                      compile_fwd=compile_on)

    packed = None
    if processor is None and attn_impl == "flash_attention_2":
        packed = PackedRunner(model, head_w, pad_id)
        rng = torch.Generator(device=dev).manual_seed(20261028)
        ids = torch.randint(1, 8192, (3, 96), generator=rng, device=dev)
        mask = torch.zeros(3, 96, dtype=torch.long, device=dev)
        for r, n in enumerate((96, 40, 71)):
            mask[r, :n] = 1
        slots = mask.sum(1) - 1
        with torch.no_grad():
            ref = torch.softmax(
                runner._eager_core(ids)[torch.arange(3, device=dev), slots].float()
                if runner is not None else
                model.model(input_ids=ids, attention_mask=mask,
                            use_cache=False).last_hidden_state[
                    torch.arange(3, device=dev), slots].float() @ head_w.T, dim=-1)
        with torch.inference_mode():
            got = torch.softmax(packed.run(ids, mask, torch.arange(3, device=dev),
                                           slots)[0].float(), dim=-1)
        diff = float((ref - got).abs().max())
        if diff > 5e-2:
            print(f"[executor] packed self-check FAILED (max diff {diff:.2e}); "
                  "packed path disabled", file=sys.stderr, flush=True)
            packed = None
        else:
            print(f"[executor] packed varlen forward on "
                  f"(self-check max diff {diff:.2e})", file=sys.stderr, flush=True)

    if graphs_on:
        _mrows = GRAPH_ROW_LADDER[-1]
        _mlen = max(runner.L_BUCKETS[-1] if runner is not None else 0,
                    media_sizes[-1] if media_sizes else 0)
        TRANSIENT_PEAK[0] = _measure_transient_peak(model, pad_id, _mrows, _mlen)
        print(f"[executor] gpu-memory-utilization {GPU_MEMORY_UTILIZATION:.2f} = "
              f"{_budget_bytes() >> 20} MiB budget (held "
              f"{torch.cuda.memory_reserved() >> 20} MiB, transient peak "
              f"{TRANSIENT_PEAK[0] >> 20} MiB)", file=sys.stderr, flush=True)
        print(f"[executor] graph ladder: --graph-max-len requested {GRAPH_MAX_LEN}, "
              f"effective length cap {length_cap}, rows up to {GRAPH_ROW_LADDER[-1]}",
              file=sys.stderr, flush=True)
        if runner is not None:
            runner.prewarm()
        if media_graphs is not None:
            media_graphs.prewarm()
        if fork_graphs is not None:
            fork_graphs.prewarm((3, 1))
        if text_fork_graphs is not None:
            text_fork_graphs.prewarm((1,))
        torch.cuda.empty_cache()
        if compile_on:
            from torch import _dynamo

            _dynamo.config.suppress_errors = True

    if processor is not None and MM_FORK[0]:
        ok, note = _mm_fork_selfcheck(
            model, processor, tokenizer, letter_ids, head_w, pad_id,
            fork_graphs=fork_graphs)
        degraded = False
        if not ok and fork_graphs is not None:
            fork_graphs.off = True
            degraded = True
            ok, note = _mm_fork_selfcheck(
                model, processor, tokenizer, letter_ids, head_w, pad_id)
        MM_FORK[0] = ok
        if degraded:
            print(f"[executor] fork graphs failed the self-check; "
                  f"serving fork eager ({note})", file=sys.stderr, flush=True)
        elif not ok:
            print(f"[executor] fork self-check FAILED ({note})",
                  file=sys.stderr, flush=True)

    if processor is None and TEXT_FORK[0]:
        ok, note = _text_fork_selfcheck(model, tokenizer, letter_ids, head_w,
                                        pad_id, fork_graphs=text_fork_graphs)
        degraded = False
        if not ok and text_fork_graphs is not None:
            text_fork_graphs.off = True
            degraded = True
            ok, note = _text_fork_selfcheck(model, tokenizer, letter_ids,
                                            head_w, pad_id)
        TEXT_FORK[0] = ok
        if degraded:
            print(f"[executor] text fork graphs failed the self-check; "
                  f"serving fork eager ({note})", file=sys.stderr, flush=True)
        elif not ok:
            print(f"[executor] text fork self-check FAILED ({note})",
                  file=sys.stderr, flush=True)

    if (processor is None and TEXT_FORK[0] and text_fork_graphs is not None
            and text_fork_graphs.batch_max >= 2 and text_fork_graphs.multi):
        ok, note = _text_fork_multi_selfcheck(
            model, tokenizer, letter_ids, head_w, pad_id,
            fork_graphs=text_fork_graphs)
        if ok:
            print(f"[executor] batched branch on ({note})",
                  file=sys.stderr, flush=True)
        else:
            print(f"[executor] batched branch self-check FAILED ({note}); "
                  "serving per-group branch", file=sys.stderr, flush=True)
            text_fork_graphs.batch_max = 0

    if graphs_on:
        if cc_dir:
            if cc_hit:
                print(f"[executor] compile cache HIT at {cc_dir}",
                      file=sys.stderr, flush=True)
            else:
                print(f"[executor] compile cache MISS -- compiled fresh, stored at {cc_dir}",
                      file=sys.stderr, flush=True)
        print(f"[executor] warmup done in {time.time() - t_warm:.1f}s",
              file=sys.stderr, flush=True)

    fd = sys.stdin.fileno()

    fork_used = text_fork_graphs if text_fork_graphs is not None else fork_graphs
    cells = {
        "text": list(runner.stats()) if runner is not None else [0, 0],
        "media": list(media_graphs.stats()) if media_graphs is not None else [0, 0],
        "fork": list(fork_used.stats()) if fork_used is not None else [0, 0],
    }

    print(json.dumps({"ready": True, "load_s": time.time() - t0,
                      "device": dev, "dtype": str(dtype).replace("torch.", ""),
                      "family": family, "letters": len(table),
                      "multimodal": processor is not None,
                      "fork": MM_FORK[0],
                      "text_fork": processor is None and TEXT_FORK[0],
                      "graphs": len(runner.graphs) if runner else 0,
                      "fork_graphs": len(text_fork_graphs.branches) if text_fork_graphs else 0,
                      "l_buckets": list(runner.L_BUCKETS) if runner else [],
                      "graph_cells": cells,
                      "graph_envelope": {
                          "max_len": runner.L_BUCKETS[-1] if runner is not None
                          else (media_sizes[-1] if media_sizes else 0),
                          "max_rows": GRAPH_ROW_LADDER[-1],
                          "fork_max_prefix": fork_used.caps[-1] if fork_used is not None else 0,
                          "fork_max_width": fork_used.max_width if fork_used is not None else 0,
                          "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                          "budget_bytes": _budget_bytes(),
                          "reserved_bytes": torch.cuda.memory_reserved(),
                          "transient_bytes": TRANSIENT_PEAK[0]}}),
          flush=True)

    import select as _select

    def emit(meta, logits):
        _o0 = time.perf_counter()
        SERVED_ROWS[0] += int(meta.get("n_rows", 0))
        reply = {"logits_b64": _f32_b64(logits),
                 "served_rows": SERVED_ROWS[0],
                 "uncovered": dict(COVERAGE)}
        reply.update(meta)
        print(json.dumps(reply), flush=True)
        EMIT_LAST[0] = (time.perf_counter() - _o0) * 1000.0
        return EMIT_LAST[0]

    while True:
        _tr = time.perf_counter()
        ready, _, _ = _select.select([fd], [], [])
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as err:
            print(json.dumps({"error": f"bad request json: {err}"}), flush=True)
            continue
        READ_MS[0] = (time.perf_counter() - _tr) * 1e3
        if req.get("cmd") == "quit":
            break
        try:
            t = time.time()
            _ipc0 = time.perf_counter()
            slot_batch = _i64(req["slot_batch_b64"])
            b = slot_batch.shape[0]
            ids_h = _i64h(req["input_ids_b64"])
            mask_h = _i64h(req["attention_mask_b64"])
            slot_h = _i64h(req["slot_idx_b64"])
            ids = torch.from_numpy(ids_h).to(DEV).view(b, -1)
            mask = torch.from_numpy(mask_h).to(DEV).view(b, -1)
            slot_idx = torch.from_numpy(slot_h).to(DEV)
            _prefix_b64 = req.get("prefix_len_b64")
            prefix_lens = (np.frombuffer(base64.b64decode(_prefix_b64), dtype="<i8")
                           .tolist() if _prefix_b64 else None)
            _ipc_in_ms = (time.perf_counter() - _ipc0) * 1000.0
            IPC_LAST[0] = _ipc_in_ms
            media = req.get("media")
            if media is not None and processor is None:
                raise RuntimeError("this checkpoint has no vision tower; media was rejected too late")
            if media is not None:
                counts = req["media_counts"]
                texts = req["row_texts"]
                if len(counts) != b or len(texts) != b or sum(counts) != len(media):
                    raise RuntimeError("media_counts / row_texts do not match the batch")
                _dec_cache = {}
                _dks = [hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()
                        for m in media]
                row_media_keys = []
                _koff = 0
                for _c in counts:
                    row_media_keys.append(tuple(_dks[_koff:_koff + _c]))
                    _koff += _c
                _dec_ms = [0.0]

                def _load_all():
                    _d0 = time.perf_counter()
                    out = []
                    for m, _dk in zip(media, _dks):
                        if _dk not in _dec_cache:
                            _dec_cache[_dk] = (_load_video(m) if m.get("kind", "image") == "video"
                                               else _load_image(m))
                        out.append(_dec_cache[_dk])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _dec_ms[0] = (time.perf_counter() - _d0) * 1000.0
                    return out

                if TOWER_ACC is not None:
                    TOWER_ACC[1] = 0.0
                forked = None
                if (MM_FORK[0] and fork_graphs is not None
                        and not fork_graphs.off and b >= 1):
                    _f0 = time.perf_counter()
                    try:
                        forked = _mm_fork_forward(
                            model, processor, tokenizer, letter_ids, head_w,
                            pad_id, ids, mask, texts, _load_all, vision_ids, counts,
                            fork_graphs, row_media_keys, prefix_lens)
                    except Exception as err:
                        print(f"[executor] fork refused ({err}); serving per-row",
                              file=sys.stderr, flush=True)
                    if DEV == "cuda":
                        torch.cuda.synchronize()
                    _fwd_ms = (time.perf_counter() - _f0) * 1000.0
                    _proc_ms = _drift_ms = 0.0
                if forked is not None:
                    logits, expanded_lens, _ = forked
                    if MM_PROFILE:
                        _tower = TOWER_ACC[1] if TOWER_ACC is not None else -1.0
                        _st = FORK_STATS
                        print(f"[mm-profile] wall={(time.time() - t) * 1000.0:.1f}ms "
                              f"ipc_in={_ipc_in_ms:.1f} decode={_dec_ms[0]:.1f} "
                              f"processor={_st.get('proc', -1.0):.1f} tower={_tower:.1f} "
                              f"embed={_st.get('embed', -1.0):.1f} "
                              f"prefix={_st.get('prefix', -1.0):.1f} "
                              f"branch={_st.get('branch', -1.0):.1f} "
                              f"fwd={_fwd_ms:.1f} drift=0.0 ipc_out=0.0 rows={b} fork=1",
                              file=sys.stderr, flush=True)
                    emit({"n_rows": int(logits.shape[0]),
                          "width": int(logits.shape[1]),
                          "expanded_lens": expanded_lens,
                          "fwd_ms": (time.time() - t) * 1000.0}, logits)
                    continue
                loaded = _load_all()
                real_lens = mask.sum(1).tolist()
                rows = []
                expanded_lens = []
                offset = 0
                woffset = 0
                vision_cache = {}
                proc_cache = {}
                _proc_ms = _drift_ms = _fwd_ms = 0.0
                if TOWER_ACC is not None:
                    TOWER_ACC[1] = 0.0
                row_inputs = []
                for i in range(b):
                    items = loaded[offset:offset + counts[i]]
                    offset += counts[i]
                    wire_items = media[woffset:woffset + counts[i]]
                    woffset += counts[i]
                    images = [x for x in items if not isinstance(x, tuple)]
                    videos, metas = [], []
                    for x in items:
                        if isinstance(x, tuple):
                            videos.append(x[0])
                            metas.append(x[1])
                    rust_ids = ids[i, :real_lens[i]].tolist()
                    _vk = hashlib.sha1(
                        json.dumps(wire_items, sort_keys=True).encode()).hexdigest()
                    out = None
                    cached = proc_cache.get(_vk)
                    if cached is None:
                        _p0 = time.perf_counter()
                        out = processor(text=[texts[i]], return_tensors="pt",
                                        **_processor_kwargs(images, videos, metas))
                        _proc_ms += (time.perf_counter() - _p0) * 1000.0
                        expanded = out["input_ids"][0].tolist()
                        proc_cache[_vk] = (
                            expanded, rust_ids,
                            {k: v.to(DEV) for k, v in out.items()
                             if k in ("pixel_values", "image_grid_thw",
                                      "pixel_values_videos", "video_grid_thw")},
                            out["mm_token_type_ids"][0].tolist()
                            if "mm_token_type_ids" in out else None)
                    else:
                        expanded0, base_ids, _base, mm0 = cached
                        n = 0
                        for a, c in zip(base_ids, rust_ids):
                            if a != c:
                                break
                            n += 1
                        P = n + len(expanded0) - len(base_ids)
                        expanded = expanded0[:P] + rust_ids[n:]
                        if len(expanded) != len(expanded0) - len(base_ids) + len(rust_ids):
                            raise RuntimeError("processor cache: expansion mismatch")
                        out = {"__derived__": True, "mm0": mm0, "P": P, "base": _base}
                    _d0 = time.perf_counter()
                    if MM_DRIFT_CHECK:
                        try:
                            _check_expanded(
                                rust_ids, expanded, *vision_ids,
                                lambda t: tokenizer.decode(t),
                            )
                        except ValueError as err:
                            raise RuntimeError(f"row {i}: expansion check: {err}")
                    _drift_ms += (time.perf_counter() - _d0) * 1000.0
                    if out.get("__derived__"):
                        inputs = dict(out["base"])
                        inputs["input_ids"] = torch.tensor([expanded], dtype=torch.long,
                                                           device=DEV)
                        if out["mm0"] is not None:
                            _mm = out["mm0"][:out["P"]] + [0] * (len(expanded) - out["P"])
                            inputs["mm_token_type_ids"] = torch.tensor(
                                [_mm], dtype=torch.long, device=DEV)
                    else:
                        inputs = {k: v.to(DEV) for k, v in out.items()
                                  if k in ("input_ids", "attention_mask", "pixel_values",
                                           "image_grid_thw", "pixel_values_videos",
                                           "video_grid_thw", "mm_token_type_ids")}
                    rows.append(None)
                    row_inputs.append((_vk, inputs, i))
                    expanded_lens.append(len(expanded))
                _f0 = time.perf_counter()
                with torch.inference_mode():
                    groups = {}
                    for _vk, _inp, _i in row_inputs:
                        groups.setdefault(_vk, []).append((_inp, _i))
                    for _vk, members in groups.items():
                        _first = members[0][0]
                        _done = False
                        _media_plan = None
                        if len(members) > 1 and media_graphs is not None and MM_BATCH[0]:
                            _media_plan = media_graphs.plan(
                                max(m[0]["input_ids"].shape[1] for m in members),
                                len(members))
                        if _media_plan is not None:
                          try:
                            _L = max(m[0]["input_ids"].shape[1] for m in members)
                            _ids = torch.full((len(members), _L), pad_id, dtype=torch.long,
                                              device=DEV)
                            _mmt = torch.zeros((len(members), _L), dtype=torch.long,
                                               device=DEV)
                            _ends = []
                            for _r, (_inp, _i) in enumerate(members):
                                _n = _inp["input_ids"].shape[1]
                                _ids[_r, :_n] = _inp["input_ids"][0]
                                if "mm_token_type_ids" in _inp:
                                    _mmt[_r, :_n] = _inp["mm_token_type_ids"][0]
                                _ends.append(_n - 1)
                            _b = {"input_ids": _ids, "mm_token_type_ids": _mmt}
                            for _k in ("pixel_values", "image_grid_thw",
                                       "pixel_values_videos", "video_grid_thw"):
                                if _k in _first:
                                    _b[_k] = _first[_k].repeat(len(members), 1)
                            _z = vision_cache.get(_vk)
                            if _z is None:
                                _z = _mm_vision(model, _first)
                                vision_cache[_vk] = _z
                            _z = tuple(None if f is None else f.repeat(len(members), 1)
                                       for f in _z)
                            _emb, _pos, _ = _mm_embed(model, _b, _z)
                            _last = media_graphs(
                                _emb, _pos, torch.tensor(_ends, device=DEV))
                            if _last is not None:
                                _lg = (_last.float() @ head_w.T)
                                for _r, _inp_i in enumerate(members):
                                    rows[_inp_i[1]] = _lg[_r].tolist()
                                _done = True
                          except Exception as err:
                            print(f"[executor] media batch path failed: {err!r}\n"
                                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)
                        if _done:
                            continue
                        for _inp, _i in members:
                            _fwd = dict(_inp)
                            _am = _fwd.pop("attention_mask", None)
                            if _am is not None and not bool(_am.all()):
                                _fwd["attention_mask"] = _am
                            _last = None
                            if media_graphs is not None:
                                _z = vision_cache.get(_vk)
                                if _z is None:
                                    _z = _mm_vision(model, _inp)
                                    vision_cache[_vk] = _z
                                _emb, _pos, _ = _mm_embed(model, _inp, _z)
                                _last = media_graphs(
                                    _emb, _pos,
                                    torch.tensor([_inp["input_ids"].shape[1] - 1],
                                                 device=DEV))
                            if _last is None:
                                if media_graphs is not None:
                                    note_uncovered(
                                        media_graphs.why(_inp["input_ids"].shape[1], 1), 1)
                                _last = model(**_fwd, use_cache=False, return_dict=True,
                                              logits_to_keep=1).logits[0, -1][letter_ids].float()
                            else:
                                _last = (_last.float() @ head_w.T)
                            rows[_i] = _last.tolist()
                if DEV == "cuda":
                    torch.cuda.synchronize()
                _fwd_ms += (time.perf_counter() - _f0) * 1000.0
                logits = torch.tensor(rows, dtype=torch.float32, device=DEV)
                _ipc_out_ms = emit({"n_rows": int(logits.shape[0]),
                                    "width": int(logits.shape[1]),
                                    "expanded_lens": expanded_lens,
                                    "fwd_ms": (time.time() - t) * 1000.0}, logits)
                if MM_PROFILE:
                    _tower = TOWER_ACC[1] if TOWER_ACC is not None else -1.0
                    print(f"[mm-profile] wall={(time.time() - t) * 1000.0:.1f}ms "
                          f"ipc_in={_ipc_in_ms:.1f} decode={_dec_ms[0]:.1f} "
                          f"processor={_proc_ms:.1f} tower={_tower:.1f} "
                          f"fwd={_fwd_ms:.1f} drift={_drift_ms:.1f} "
                          f"ipc_out={_ipc_out_ms:.1f} rows={b}",
                          file=sys.stderr, flush=True)
                continue
            if _TORCH_PROFILE:
                _tprof_tick()
            if processor is None and TEXT_FORK[0] and b >= 1:
                _fc0 = time.perf_counter()
                PHASE_STATS["rest"] = -1.0
                try:
                    forked = _text_fork_forward(model, runner, head_w, pad_id,
                                                ids, mask, slot_idx,
                                                text_fork_graphs, prefix_lens)
                except Exception as err:
                    print(f"[executor] text fork refused ({err}); "
                          "serving whole rows", file=sys.stderr, flush=True)
                    forked = None
                PHASE_STATS["forkcall"] = (time.perf_counter() - _fc0) * 1e3
                if forked is not None:
                    _batch_prof(b, ids.shape[1], mask, "fork", t)
                    emit({"n_rows": int(forked.shape[0]),
                          "width": int(forked.shape[1]),
                          "fwd_ms": (time.time() - t) * 1000.0}, forked)
                    continue
            if packed is not None:
                try:
                    logits, _ = packed.run(ids, mask, slot_batch, slot_idx,
                                           host=(ids_h, mask_h))
                except Exception as err:
                    print(f"[executor] packed forward failed ({err!r}); "
                          "packed path disabled, serving padded",
                          file=sys.stderr, flush=True)
                    packed = None
                else:
                    _batch_prof(b, ids.shape[1], mask, "packed", t)
                    emit({"n_rows": int(logits.shape[0]),
                          "width": int(logits.shape[1]),
                          "fwd_ms": (time.time() - t) * 1000.0}, logits)
                    continue
            why = None
            if runner is not None:
                logits, why = runner.run(ids, mask, slot_batch, slot_idx)
                note_uncovered(why, b)
            else:
                kw = {} if bool(mask.all()) else {"attention_mask": mask}
                with torch.inference_mode():
                    hs = model.model(input_ids=ids, use_cache=False, **kw).last_hidden_state
                    logits = hs[slot_batch, slot_idx].float() @ head_w.T
            reply = {"n_rows": int(logits.shape[0]),
                     "width": int(logits.shape[1]),
                     "fwd_ms": (time.time() - t) * 1000.0}
            _batch_prof(b, ids.shape[1], mask, "eager" if why else "graph", t)
            emit(reply, logits)
        except Exception as err:
            print(f"[executor] request failed: {err!r}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
            if _fatal_cuda(err):
                print("[executor] fatal CUDA fault: exiting so the server restarts the executor",
                      file=sys.stderr, flush=True)
                sys.exit(1)
            print(json.dumps({"error": f"forward failed: {err!r}"}), flush=True)

if __name__ == "__main__":
    main()
