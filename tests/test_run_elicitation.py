#!/usr/bin/env python3
"""Integration test for the elicitation driver (numpy-only, CPU).

Builds synthetic activation .npz files matching extract_activations.py's schema — with one
*signal* layer and one *noise* layer, plus planted CCS contrast pairs — then runs the driver and
asserts it: selects the signal layer (via train-pool validation, not the test set), recovers the
signal with the k-shot probe, and elicits truth unsupervised with CCS while the random-direction
control stays near chance.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "elicitation"))
import run_elicitation as R  # noqa: E402

SIGNAL_LAYER, NOISE_LAYER, D = 12, 6, 16


def _dirs(seed=100):
    """Planted directions are a property of the 'model' — shared across train/test splits."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, D); u /= np.linalg.norm(u)
    t = rng.normal(0, 1, D); t /= np.linalg.norm(t)
    posp = rng.normal(0, 1, D); negp = rng.normal(0, 1, D)
    return u, t, posp, negp


def _make_split(n, noise_seed, dirs):
    u, t, posp, negp = dirs
    rng = np.random.default_rng(noise_seed)   # only labels + noise vary across splits
    y = rng.integers(0, 2, n)
    return dict(
        hard_label=y.astype(np.int64),
        txt=np.array([f"ex{i}" for i in range(n)], dtype=object),
        layers=np.array([NOISE_LAYER, SIGNAL_LAYER]),
        **{f"acts_L{SIGNAL_LAYER}": ((2 * y - 1)[:, None] * 3.0 * u + rng.normal(0, 1.2, (n, D))).astype(np.float32),
           f"acts_L{NOISE_LAYER}": rng.normal(0, 1, (n, D)).astype(np.float32),
           f"pos_L{SIGNAL_LAYER}": (y[:, None] * 4.0 * t + posp + rng.normal(0, 1.0, (n, D))).astype(np.float32),
           f"neg_L{SIGNAL_LAYER}": ((1 - y)[:, None] * 4.0 * t + negp + rng.normal(0, 1.0, (n, D))).astype(np.float32),
           f"pos_L{NOISE_LAYER}": rng.normal(0, 1, (n, D)).astype(np.float32),
           f"neg_L{NOISE_LAYER}": rng.normal(0, 1, (n, D)).astype(np.float32)},
    )


def test_driver_end_to_end():
    dirs = _dirs()
    with tempfile.TemporaryDirectory() as d:
        np.savez(Path(d) / "boolq_gpt2-xl_s0_train.npz", **_make_split(800, 1, dirs))
        np.savez(Path(d) / "boolq_gpt2-xl_s0_test.npz", **_make_split(500, 2, dirs))
        res = R.run(d, "boolq", "gpt2-xl", 0)

    assert res["layer"] == SIGNAL_LAYER, res["layer"]              # picked the signal layer
    assert res["m1_full_supervised"] > 0.9, res["m1_full_supervised"]
    assert res["m1_probe"][256] >= res["m1_probe"][8]             # more labels never hurt
    assert res["m1_probe"][8] > 0.7 and res["m1_probe"][256] > 0.85
    assert "m2_ccs" in res
    assert res["m2_ccs"][32] > 0.8, res["m2_ccs"][32]            # CCS elicits truth (oriented w/ 32)
    # random direction is a weak baseline; in d=16 it catches partial chance overlap (~0.63), in the
    # real 768-1600-dim models it sits near chance. The meaningful check: CCS clearly beats it.
    assert res["control_random_direction"] < 0.75
    assert res["m2_ccs"][32] > res["control_random_direction"] + 0.15


if __name__ == "__main__":
    test_driver_end_to_end()
    print("  ok  test_driver_end_to_end")
    print("test_run_elicitation: ALL PASS")
