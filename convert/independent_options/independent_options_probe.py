"""Probe cases for the independent_options converter.

Covers the dynamic-axis minimums, padding presence, an option count below K and
adjacent option spans.
"""

import numpy as np
import torch

from compare import deviation

SENTINEL = -1e4


def _feed(rows, pad, sliding):
    from von.backends.option_marker_backend import _bool_to_additive
    from von.models.option_marker import (
        build_independent_option_masks,
        build_option_invariant_position_ids,
    )

    b = len(rows)
    seq = max(len(row[0]) for row in rows)
    k = max(len(row[1]) for row in rows)
    ids = np.full((b, seq), pad, dtype=np.int64)
    amask = np.zeros((b, seq), dtype=np.int64)
    pos = np.zeros((b, k), dtype=np.int64)
    pmask = np.zeros((b, k), dtype=bool)
    for i, (tokens, markers) in enumerate(rows):
        ids[i, : len(tokens)] = tokens
        amask[i, : len(tokens)] = 1
        pos[i, : len(markers)] = markers
        pmask[i, : len(markers)] = True
    t_ids = torch.from_numpy(ids)
    t_mask = torch.from_numpy(amask)
    positions = [row[1] for row in rows]
    position_ids = build_option_invariant_position_ids(t_ids, t_mask, positions)
    masks = build_independent_option_masks(t_ids, t_mask, positions, position_ids, sliding)
    return {
        "input_ids": ids,
        "full_mask": _bool_to_additive(masks["full_attention"]).numpy(),
        "sliding_mask": _bool_to_additive(masks["sliding_attention"]).numpy(),
        "position_ids": position_ids.numpy(),
        "marker_pos": pos,
        "marker_mask": pmask,
    }


def cases(vocab, pad, sliding):
    rng = np.random.default_rng(0)

    def tokens(n):
        return rng.integers(5, vocab, size=n).tolist()

    return [
        _feed(
            [(tokens(64), [10, 30, 50]), (tokens(45), [10, 30])],
            pad,
            sliding,
        ),
        _feed([(tokens(8), [2, 5])], pad, sliding),
        _feed([(tokens(64), [10, 11])], pad, sliding),
    ]


def run(wrapper, session, vocab, pad, sliding):
    worst = 0.0
    for feed in cases(vocab, pad, sliding):
        with torch.no_grad():
            ref = wrapper(
                torch.from_numpy(feed["input_ids"]),
                torch.from_numpy(feed["full_mask"]),
                torch.from_numpy(feed["sliding_mask"]),
                torch.from_numpy(feed["position_ids"]),
                torch.from_numpy(feed["marker_pos"]),
                torch.from_numpy(feed["marker_mask"]),
            )
        got = session.run(None, feed)[0]
        worst = max(worst, deviation(ref.numpy(), got, SENTINEL))
    return worst
