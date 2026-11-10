"""Shared export check: how far the exported graph strays from the torch reference.

The converter prints one line for the Rust side to turn into the user-facing
report; nothing here decides whether a conversion succeeds.
"""

import numpy as np

THRESHOLD = 1e-3


def deviation(ref, got, sentinel=None):
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    if ref.shape != got.shape:
        return float("inf")
    scale_source = ref if sentinel is None else ref[ref != sentinel]
    if scale_source.size == 0:
        scale_source = ref
    scale = max(1.0, float(np.abs(scale_source).max()))
    return float(np.abs(ref - got).max()) / scale


def report(worst):
    print(f"[verdict] deviation={worst:.1e} large={'yes' if worst > THRESHOLD else 'no'}")
