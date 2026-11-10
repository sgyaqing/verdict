"""Probe cases for the marker_head converter.

Covers the dynamic-axis minimums, padding presence, marker counts (all, partial,
one) and every qtype value.
"""

import numpy as np
import torch

from compare import deviation

SENTINEL = -1e4


def _batch(rows, pad):
    b = len(rows)
    seq = max(len(row[0]) for row in rows)
    k = max(len(row[1]) for row in rows)
    ids = np.full((b, seq), pad, dtype=np.int64)
    amask = np.zeros((b, seq), dtype=np.int64)
    pos = np.zeros((b, k), dtype=np.int64)
    pmask = np.zeros((b, k), dtype=bool)
    qtype = np.zeros(b, dtype=np.int64)
    for i, (tokens, markers, valid, q) in enumerate(rows):
        ids[i, : len(tokens)] = tokens
        amask[i, : len(tokens)] = 1
        pos[i, : len(markers)] = markers
        pmask[i, : len(valid)] = valid
        qtype[i] = q
    return ids, amask, pos, pmask, qtype


def cases(n_act, vocab, pad):
    rng = np.random.default_rng(0)

    def tokens(n):
        return rng.integers(5, vocab, size=n).tolist()

    return [
        _batch(
            [
                (tokens(40), [3, 9, 15, 21], [True, True, True, True], 0),
                (tokens(30), [3, 9, 0, 0], [True, True, False, False], 1),
            ],
            pad,
        ),
        _batch([(tokens(8), [2, 5], [True, True], n_act - 1)], pad),
        _batch([(tokens(40), [3, 0, 0, 0], [True, False, False, False], 0)], pad),
    ]


def run(wrapper, session, n_act, vocab, pad):
    worst = 0.0
    for ids, amask, pos, pmask, qtype in cases(n_act, vocab, pad):
        feed = {
            "input_ids": ids,
            "attention_mask": amask,
            "marker_pos": pos,
            "marker_mask": pmask,
            "qtype": qtype,
        }
        with torch.no_grad():
            ref = wrapper(
                torch.from_numpy(ids),
                torch.from_numpy(amask),
                torch.from_numpy(pos),
                torch.from_numpy(pmask),
                torch.from_numpy(qtype),
            )
        got = session.run(None, feed)
        worst = max(
            worst,
            deviation(ref[0].numpy(), got[0], SENTINEL),
            deviation(ref[1].numpy(), got[1]),
        )
    return worst
