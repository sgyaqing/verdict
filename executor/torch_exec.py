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

def _row_ladder(cap: int) -> tuple:
    """Bucket ladder: every row count maps up to one of these sizes."""
    out, v = [], 1
    while v < cap:
        out.append(v)
        v *= 2
    out.append(cap)
    return tuple(out)

GRAPH_ROWS_CAP = int(os.environ.get("VERDICT_GRAPH_ROWS_CAP", 8))
GRAPH_ROW_LADDER = _row_ladder(GRAPH_ROWS_CAP)
GRAPH_WIDTH_STEP = int(os.environ.get("VERDICT_GRAPH_WIDTH_STEP", 16))
GRAPH_PREFIX_STEP = int(os.environ.get("VERDICT_GRAPH_PREFIX_STEP", 128))

class GraphRunner:
    """CUDA graphs over the letter forward, one per (rows, length) bucket.

    The captured region carries no attention mask: the model is causal so right padding is never attended, and a mask built inside the forward allocates a tensor per call, which capture forbids."""

    B_BUCKETS = GRAPH_ROW_LADDER
    MAX_GRAPHS = 48
    WARM_L_CAP = 512

    def __init__(self, model, head_w, pad_id, max_len=32768, compile_fwd=False,
                 max_graphs: int = MAX_GRAPHS):
        self.model = model
        self.head_w = head_w
        self.pad_id = pad_id
        self.dev = "cuda"
        self.graphs = collections.OrderedDict()
        self.failed = set()
        self.max_graphs = max_graphs
        self.L_BUCKETS = self._l_buckets(max_len)
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

    @classmethod
    def _l_buckets(cls, max_len: int) -> tuple:
        cap = min(cls.WARM_L_CAP, max(384, max_len // 8))
        ratio = (cap / 64) ** (1.0 / 5)
        out, v = [64], 64.0
        for _ in range(5):
            v = min(round(v * ratio), cap)
            if v > out[-1]:
                out.append(v)
        if out[-1] < cap:
            out[-1] = cap
        return tuple(out)

    def _evict(self) -> bool:
        if not self.graphs:
            return False
        self.graphs.popitem(last=False)
        return True

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

    def warm_compiled(self) -> None:
        """Compile the forward at the shapes a capture will use."""
        if not self.compiled:
            return
        with torch.no_grad():
            for rows in (1, self.B_BUCKETS[-1]):
                self._fwd(torch.full((rows, 64), self.pad_id,
                                     dtype=torch.long, device=self.dev))
        torch.cuda.synchronize()

    def _eager_core(self, ids):
        """The eager forward body: backbone (no cache) + letter readout."""
        hs = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        return (hs.float() @ self.head_w.T)

    def _fwd(self, ids):
        return self._fwd_impl(ids)

    @torch.no_grad()
    def capture(self, b, l):
        t0 = time.perf_counter()
        s_ids = torch.full((b, l), self.pad_id, dtype=torch.long, device=self.dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._fwd(s_ids)
        torch.cuda.current_stream().wait_stream(st)

        def run():
            return self._fwd(s_ids)

        while len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False)
        _make_room(self._evict)
        s_out, g = _capture_graph(run)
        self.graphs[(b, l)] = (s_ids, s_out, g)
        print(f"[executor] captured a text graph for {b} rows x {l} tokens "
              f"in {(time.perf_counter() - t0) * 1e3:.0f} ms "
              f"(free {torch.cuda.mem_get_info()[0] >> 20} MiB)",
              file=sys.stderr, flush=True)
        return self.graphs[(b, l)]

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

    @torch.no_grad()
    def run(self, ids, mask, slot_batch, slot_idx):
        """Graphs path: OOM clears every cached graph and serves this request eagerly; any other failure turns the path off."""
        b, l = ids.shape
        key = self.plan(b, l)
        if key is None or key in self.failed:
            return self._eager_readout(ids, mask, slot_batch, slot_idx), True
        entry = self.graphs.get(key)
        if entry is None:
            try:
                entry = self.capture(*key)
            except (GraphMemoryLow, RuntimeError) as err:
                self.failed.add(key)
                torch.cuda.empty_cache()
                print(f"[executor] text graph {key[0]}x{key[1]} unusable "
                      f"({type(err).__name__}); eager there from now on",
                      file=sys.stderr, flush=True)
                return self._eager_readout(ids, mask, slot_batch, slot_idx), True
        else:
            self.graphs.move_to_end(key)
        bb, ll = key
        s_ids, s_out, g = entry
        s_ids.fill_(self.pad_id)
        s_ids[:b, :l].copy_(ids)
        g.replay()
        return s_out[slot_batch, slot_idx].float().clone(), False

DEV = "cuda"

MM_PROFILE = os.environ.get("VERDICT_MM_PROFILE") == "1"

MM_DRIFT_CHECK = os.environ.get("VERDICT_MM_DRIFT_CHECK", "0") == "1"

MM_FORK = [os.environ.get("VERDICT_MM_FORK", "1") != "0"]
MM_GRAPHS = [os.environ.get("VERDICT_MM_GRAPHS", "1") != "0"]
MM_BATCH = [os.environ.get("VERDICT_MM_BATCH", "1") != "0"]
FORK_STATS = {}
FORK_MIN_ROWS = 6

def _host_available() -> int:
    """MemAvailable in bytes; on shared-memory GPUs page cache is free for CUDA while mem_get_info counts it as used."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0

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

class GraphMemoryLow(RuntimeError):
    """Not enough free GPU memory to capture another graph, even after dropping the cached ones."""

def _make_room(evict, reserve_fraction: float = 0.1, reserve_bytes: int = 8 << 30) -> None:
    """Evict LRU graphs until the device holds its reserve, else raise GraphMemoryLow."""

    def low() -> bool:
        free, total = torch.cuda.mem_get_info()
        if torch.cuda.get_device_properties(torch.cuda.current_device()).is_integrated:
            free = max(free, _host_available())
        return free < max(reserve_bytes, reserve_fraction * total)

    if low():
        torch.cuda.empty_cache()
    while low() and evict():
        torch.cuda.empty_cache()
    if low():
        raise GraphMemoryLow("less than the reserve free on the device")

def _bucket(n: int, step: int) -> int:
    return -(-n // step) * step

def _capture_graph(run, pool=None):
    """Capture `run` with its own memory pool unless one is passed."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        out = run()
    return out, graph

class MediaGraphs:
    """Whole-prompt decoder graphs over media embeddings, one per (rows, length) bucket."""

    B_BUCKETS = GRAPH_ROW_LADDER
    MAX_GRAPHS = 32

    def __init__(self, decoder, sizes, max_graphs: int = MAX_GRAPHS):
        self.decoder = decoder
        self.sizes = tuple(sorted(sizes))
        self.max_graphs = max_graphs
        self.graphs = collections.OrderedDict()
        self.failed = set()

    def plan(self, n, b):
        size = next((s for s in self.sizes if s >= n), None)
        rows = next((x for x in self.B_BUCKETS if x >= b), None)
        return (rows, size) if size is not None and rows is not None else None

    def _evict(self) -> bool:
        if not self.graphs:
            return False
        self.graphs.popitem(last=False)
        return True

    def _capture(self, rows, size):
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
        while len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False)
        _make_room(self._evict)
        out, graph = _capture_graph(run)
        self.graphs[(rows, size)] = (emb, pos, out, graph)
        print(f"[executor] captured a media graph for {rows} rows x {size} tokens "
              f"(free {torch.cuda.mem_get_info()[0] >> 20} MiB)", file=sys.stderr, flush=True)
        return self.graphs[(rows, size)]

    def __call__(self, embeds, positions, ends):
        n = int(embeds.shape[1])
        b = int(embeds.shape[0])
        key = self.plan(n, b)
        if key is None or key in self.failed:
            return None
        entry = self.graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(*key)
            except (GraphMemoryLow, torch.OutOfMemoryError) as err:
                self.failed.add(key)
                torch.cuda.empty_cache()
                print(f"[executor] media graph {key[0]}x{key[1]} unusable "
                      f"({type(err).__name__}); eager there from now on",
                      file=sys.stderr, flush=True)
                return None
        self.graphs.move_to_end(key)
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

    ROW_BUCKETS = tuple(x for x in GRAPH_ROW_LADDER if x >= 2)

    def __init__(self, decoder, fork, pad_id, step: int = GRAPH_PREFIX_STEP,
                 max_len: int = 16384, width_step: int = GRAPH_WIDTH_STEP,
                 max_width: int = 512, max_rows: int = GRAPH_ROWS_CAP,
                 max_graphs: int = 8) -> None:
        self.decoder, self.fork, self.pad_id = decoder, fork, pad_id
        self.step, self.max_len = step, max_len
        self.width_step, self.max_width, self.max_rows = width_step, max_width, max_rows
        self.max_graphs = max_graphs
        self.prefills = collections.OrderedDict()
        self.branches = collections.OrderedDict()
        self.failed = set()
        self.off = False

    def plan(self, prefix: int, rows: int, width: int):
        size, wide = _bucket(prefix, self.step), _bucket(width, self.width_step)
        rows_b = next((x for x in self.ROW_BUCKETS if x >= rows), None)
        if size > self.max_len or rows_b is None or wide > self.max_width:
            return None
        return (size, rows_b, wide)

    def __call__(self, embeds, positions, suffixes: list, delta: int):
        """embeds (1,P,hidden) and positions (3,1,P) of the prefix plus the branch token ids -> final-norm hidden state per branch end, or None past the limits."""
        n, P = len(suffixes), embeds.shape[1]
        wide = _bucket(max(len(s) for s in suffixes), self.width_step)
        plan = self.plan(P, n, max(len(s) for s in suffixes))
        if plan is None:
            return None
        size, rows, wide = plan
        if size in self.prefills:
            self.prefills.move_to_end(size)
        entry = self.prefills.get(size)
        if entry is None:
            if ("prefix", size) in self.failed:
                return None
            try:
                emb = torch.zeros((1, size, embeds.shape[-1]), dtype=embeds.dtype, device=DEV)
                pos = torch.zeros((3, 1, size), dtype=positions.dtype, device=DEV)
                entry = self._capture_prefill(size, emb, pos)
            except (GraphMemoryLow, torch.OutOfMemoryError) as err:
                self.failed.add(("prefix", size))
                torch.cuda.empty_cache()
                print(f"[executor] fork prefix {size} tokens unusable "
                      f"({type(err).__name__}); the eager fork serves it from now on",
                      file=sys.stderr, flush=True)
                return None
        emb, pos, pad, cache, graph = entry
        key = (size, rows, wide)
        if key in self.branches:
            self.branches.move_to_end(key)
        bentry = self.branches.get(key)
        if bentry is None:
            if key in self.failed:
                return None
            try:
                bentry = self._capture_branch(key, cache, positions.dtype)
            except (GraphMemoryLow, torch.OutOfMemoryError) as err:
                self.failed.add(key)
                torch.cuda.empty_cache()
                print(f"[executor] fork branch {rows}x{wide} over {size} unusable "
                      f"({type(err).__name__}); the eager fork serves it from now on",
                      file=sys.stderr, flush=True)
                return None
        a = size - P
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
        ids, base, pad_b, out, bgraph = bentry
        host = torch.full((rows, wide), self.pad_id, dtype=torch.long)
        for r in range(rows):
            s = suffixes[r % n]
            host[r, :len(s)] = torch.tensor(s, dtype=torch.long)
        ids.copy_(host, non_blocking=True)
        base.fill_(P + delta)
        pad_b.fill_(a)
        _tb0 = time.perf_counter()
        bgraph.replay()
        if MM_PROFILE:
            torch.cuda.synchronize()
        FORK_STATS["branch"] = (time.perf_counter() - _tb0) * 1e3
        ends = torch.tensor([len(s) - 1 for s in suffixes], device=DEV)
        return out[torch.arange(n, device=DEV), ends]

    def _drop_prefix(self, size: int) -> None:
        self.prefills.pop(size, None)
        for key in [k for k in self.branches if k[0] == size]:
            del self.branches[key]

    def _evict(self, keep: int | None = None) -> bool:
        """Drop the oldest branch graph, else the oldest prefix graph with its branches; never `keep`."""
        for key in self.branches:
            self.branches.pop(key)
            return True
        for size in self.prefills:
            if size != keep:
                self._drop_prefix(size)
                return True
        return False

    def clear(self) -> None:
        self.branches.clear()
        self.prefills.clear()

    def _side_warmup(self, run) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(side)

    def _capture_prefill(self, size: int, embeds, positions):
        dev = embeds.device
        emb = torch.zeros((1, size, embeds.shape[-1]), dtype=embeds.dtype, device=dev)
        pos = torch.zeros((3, 1, size), dtype=positions.dtype, device=dev)
        pad = torch.zeros((), dtype=torch.long, device=dev)

        def run():
            idx = torch.arange(size, device=dev)
            i, j = idx.view(-1, 1), idx.view(1, -1)
            mask = (j <= i) & ((i < pad) == (j < pad))
            return self.decoder(inputs_embeds=emb, position_ids=pos, use_cache=True,
                                return_dict=True,
                                attention_mask={"full_attention": mask.view(1, 1, size, size),
                                                "linear_attention": None}).past_key_values

        self._side_warmup(run)
        while len(self.prefills) >= self.max_graphs:
            self._drop_prefix(next(iter(self.prefills)))
        _make_room(self._evict)
        cache, graph = _capture_graph(run)
        self.prefills[size] = (emb, pos, pad, cache, graph)
        print(f"[executor] captured a prefix graph for {size} tokens",
              file=sys.stderr, flush=True)
        return self.prefills[size]

    def _capture_branch(self, key, cache, pos_dtype):
        size, rows, wide = key
        dev = self.prefills[size][0].device
        ids = torch.full((rows, wide), self.pad_id, dtype=torch.long, device=dev)
        base = torch.zeros((), dtype=pos_dtype, device=dev)
        pad = torch.zeros((), dtype=torch.long, device=dev)

        def run():
            q = torch.arange(wide, device=dev)
            k = torch.arange(size + wide, device=dev)
            pos = (base + q).view(1, 1, -1).expand(3, rows, -1)
            kk, qq = k.view(1, -1), q.view(-1, 1)
            mask = torch.where(kk < size, kk >= pad, kk - size <= qq)
            forked = self.fork(cache, rows)
            return self.decoder(input_ids=ids, position_ids=pos, past_key_values=forked,
                                use_cache=True, return_dict=True,
                                attention_mask={
                                    "full_attention": mask.view(1, 1, wide, size + wide)
                                    .expand(rows, 1, -1, -1),
                                    "linear_attention": None}).last_hidden_state

        self._side_warmup(run)
        while len(self.branches) >= 8 * self.max_graphs:
            self.branches.popitem(last=False)
        _make_room(lambda: self._evict(keep=size))
        out, graph = _capture_graph(run)
        self.branches[key] = (ids, base, pad, out, graph)
        print(f"[executor] captured a branch graph for {rows} rows x {wide} tokens "
              f"over a {size}-token prefix (free {torch.cuda.mem_get_info()[0] >> 20} MiB)",
              file=sys.stderr, flush=True)
        return self.branches[key]

    def run(self, embeds, positions, suffixes: list, delta: int):
        """Graphs path: OOM clears every cached graph and serves this request eagerly; any other failure turns the path off."""
        if self.off:
            return None
        try:
            return self(embeds, positions, suffixes, delta)
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

def _refuse(reason: str):
    if reason not in _FORK_REFUSED:
        _FORK_REFUSED.add(reason)
        print(f"[executor] media fork declined: {reason}; serving per-row",
              file=sys.stderr, flush=True)
    return None

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

def _mm_fork_forward(model, processor, tokenizer, letter_ids, head_w, pad_id,
                     ids, mask, texts, loaded, vision_ids, counts, fork_graphs=None):
    """One processor pass and one prefix prefill, then every suffix as a batched branch off forked caches; None when the rows do not split."""
    b = ids.shape[0]
    real_lens = mask.sum(1).tolist()
    rows_ids = [ids[i, :real_lens[i]].tolist() for i in range(b)]
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
        return _refuse("no shared token prefix across rows")
    vs, v_pads, ve = vision_ids
    vision_set = {vs, ve, *v_pads}
    if any(t in vision_set for s in suffixes for t in s):
        return _refuse("a vision token sits in a branch suffix")

    items0 = loaded[: counts[0]]
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
    if (fork_graphs is not None and not fork_graphs.off
            and bool(inputs["attention_mask"].all())
            and fork_graphs.plan(P, b, width) is not None):
        with torch.inference_mode():
            _te = time.perf_counter()
            emb, rope, rope_delta = _mm_embed(model, inputs)
            FORK_STATS["embed"] = (time.perf_counter() - _te) * 1e3
            last = fork_graphs.run(emb, rope, suffixes, rope_delta)
            logits = None if last is None else (last.float() @ head_w.T)
        if logits is not None:
            return logits, [P + len(s) for s in suffixes], P
    grids = {k: inputs.get(k) for k in ("image_grid_thw", "video_grid_thw")}
    if any(v is not None for v in grids.values()):
        positions, deltas = model.model.get_rope_index(
            inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
            attention_mask=inputs["attention_mask"], **grids)
        delta = int(deltas.view(-1)[0])
    else:
        positions = torch.arange(P, device=DEV).view(1, 1, -1).expand(3, 1, -1)
        delta = 0

    suf_ids = torch.full((b, width), pad_id, dtype=torch.long, device=DEV)
    full_mask = torch.zeros((b, P + width), dtype=torch.long, device=DEV)
    ends = []
    for row, s in enumerate(suffixes):
        suf_ids[row, : len(s)] = torch.tensor(s, device=DEV)
        full_mask[row, : P + len(s)] = 1
        ends.append(len(s) - 1)
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

    forked = _mm_fork_forward(model, processor, tokenizer, letter_ids, head_w,
                              pad_id, ids_t.to(DEV), mask.to(DEV), [text_a, text_b],
                              loaded, vision_ids, [1, 1], fork_graphs)
    if forked is None:
        return False, "fork refused the synthetic request"
    fork_logits, _, _ = forked
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
    if getattr(config, "vision_config", None) is not None:
        from transformers import AutoModelForImageTextToText
        cls = AutoModelForImageTextToText
    model = cls.from_pretrained(
        ckpt, dtype=dtype, attn_implementation="sdpa"
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
    cc_dir = compile_cache_dir(manifest, ckpt) if compile_on else None
    cc_hit = bool(cc_dir) and any(
        n != "key.json" for n in os.listdir(cc_dir))
    if not cc_dir and dev == "cuda" and not compile_on:
        print("[executor] torch.compile off (VERDICT_TORCH_COMPILE=0)",
              file=sys.stderr, flush=True)
    media_sizes = [s for s in GraphRunner._l_buckets(int(manifest.get("max_len", 32768)))
                   if s >= 128]
    t_warm = time.time()

    def build_runner(compiled: bool):
        return GraphRunner(
            model,
            head_w,
            pad_id,
            max_len=int(manifest.get("max_len", 32768)),
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

    if runner is not None:
        runner.warm_compiled()

    media_graphs = None
    if processor is not None and dev == "cuda" and MM_GRAPHS[0]:
        media_graphs = MediaGraphs(model.model, media_sizes)

    fork_graphs = None
    if (processor is not None and dev == "cuda" and MM_GRAPHS[0]
            and os.environ.get("VERDICT_FORK_GRAPHS", "1") != "0"):
        fork_graphs = ForkGraphs(model.model.language_model, _fork, pad_id)

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

    print(json.dumps({"ready": True, "load_s": time.time() - t0,
                      "device": dev, "dtype": str(dtype).replace("torch.", ""),
                      "family": family, "letters": len(table),
                      "multimodal": processor is not None,
                      "fork": MM_FORK[0],
                      "graphs": len(runner.graphs) if runner else 0}), flush=True)

    import select as _select

    def emit(meta, logits):
        _o0 = time.perf_counter()
        reply = {"logits_b64": _f32_b64(logits)}
        reply.update(meta)
        print(json.dumps(reply), flush=True)
        return (time.perf_counter() - _o0) * 1000.0

    while True:
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
        if req.get("cmd") == "quit":
            break
        try:
            t = time.time()
            _ipc0 = time.perf_counter()
            slot_batch = _i64(req["slot_batch_b64"])
            b = slot_batch.shape[0]
            ids = _i64(req["input_ids_b64"]).view(b, -1)
            mask = _i64(req["attention_mask_b64"]).view(b, -1)
            slot_idx = _i64(req["slot_idx_b64"])
            _ipc_in_ms = (time.perf_counter() - _ipc0) * 1000.0
            media = req.get("media")
            if media is not None and processor is None:
                raise RuntimeError("this checkpoint has no vision tower; media was rejected too late")
            if media is not None:
                counts = req["media_counts"]
                texts = req["row_texts"]
                if len(counts) != b or len(texts) != b or sum(counts) != len(media):
                    raise RuntimeError("media_counts / row_texts do not match the batch")
                loaded = []
                _dec0 = time.perf_counter()
                _dec_cache = {}
                for m in media:
                    _dk = hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()
                    if _dk not in _dec_cache:
                        _dec_cache[_dk] = (_load_video(m) if m.get("kind", "image") == "video"
                                           else _load_image(m))
                    loaded.append(_dec_cache[_dk])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _dec_ms = (time.perf_counter() - _dec0) * 1000.0
                if TOWER_ACC is not None:
                    TOWER_ACC[1] = 0.0
                forked = None
                if (MM_FORK[0] and fork_graphs is not None
                        and not fork_graphs.off and b >= 2):
                    _f0 = time.perf_counter()
                    try:
                        forked = _mm_fork_forward(
                            model, processor, tokenizer, letter_ids, head_w,
                            pad_id, ids, mask, texts, loaded, vision_ids, counts,
                            fork_graphs)
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
                              f"ipc_in={_ipc_in_ms:.1f} decode={_dec_ms:.1f} "
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
                          f"ipc_in={_ipc_in_ms:.1f} decode={_dec_ms:.1f} "
                          f"processor={_proc_ms:.1f} tower={_tower:.1f} "
                          f"fwd={_fwd_ms:.1f} drift={_drift_ms:.1f} "
                          f"ipc_out={_ipc_out_ms:.1f} rows={b}",
                          file=sys.stderr, flush=True)
                continue
            if runner is not None:
                logits, eager = runner.run(ids, mask, slot_batch, slot_idx)
            else:
                kw = {} if bool(mask.all()) else {"attention_mask": mask}
                with torch.inference_mode():
                    hs = model.model(input_ids=ids, use_cache=False, **kw).last_hidden_state
                    logits = hs[slot_batch, slot_idx].float() @ head_w.T
                eager = True
            reply = {"n_rows": int(logits.shape[0]),
                     "width": int(logits.shape[1]),
                     "fwd_ms": (time.time() - t) * 1000.0}
            if runner is not None:
                reply["eager"] = eager
            emit(reply, logits)
        except Exception as err:
            print(f"[executor] request failed: {err!r}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
            print(json.dumps({"error": f"forward failed: {err!r}"}), flush=True)

if __name__ == "__main__":
    main()
