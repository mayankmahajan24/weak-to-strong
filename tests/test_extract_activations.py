#!/usr/bin/env python3
"""Unit tests for the activation-selection core (no torch/GPU needed).

Verifies `select_last_token_states` reproduces the model's last-non-pad-token indexing and that
`resolve_layers` maps anchors correctly. The torch forward (`extract_states`) is integration-tested
on the GPU box, not here.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "elicitation"))
import extract_activations as ea  # noqa: E402


def test_select_basic():
    # B=2, S=4, H=3. ex0 len=2 -> position 1; ex1 len=4 -> position 3.
    hidden = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(float)
    lens = np.array([2, 4])
    out = ea.select_last_token_states(hidden, lens)
    assert out.shape == (2, 3)
    assert np.array_equal(out[0], hidden[0, 1])
    assert np.array_equal(out[1], hidden[1, 3])


def test_select_clamp():
    hidden = np.random.default_rng(0).normal(size=(3, 5, 2))
    lens = np.array([0, 1, 5])  # 0 -> idx 0 (clamped); 1 -> idx 0; 5 -> idx 4 (clamped to S-1)
    out = ea.select_last_token_states(hidden, lens)
    assert np.array_equal(out[0], hidden[0, 0])
    assert np.array_equal(out[1], hidden[1, 0])
    assert np.array_equal(out[2], hidden[2, 4])


def test_select_matches_reference_loop():
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(7, 9, 4))
    lens = rng.integers(1, 10, size=7)
    out = ea.select_last_token_states(hidden, lens)
    ref = np.stack([hidden[i, min(lens[i] - 1, 8)] for i in range(7)])
    assert np.array_equal(out, ref)


def test_resolve_layers():
    assert ea.resolve_layers("last,half", 12) == [6, 12]
    assert ea.resolve_layers("first,quarter,threequarter,last", 12) == [1, 3, 9, 12]
    assert ea.resolve_layers("5,2,5", 12) == [2, 5]          # dedup + sort
    assert ea.resolve_layers("last", 48) == [48]             # gpt2-xl has 48 layers


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_extract_activations: ALL PASS")
