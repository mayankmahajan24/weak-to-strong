#!/usr/bin/env python3
"""Unit tests for Method 2 — CCS + GT-orient (numpy-only, CPU).

Checks: the consistency+confidence loss formula; analytic gradient matches finite differences;
confidence penalizes uncertain (0.5) solutions; CCS recovers a planted truth direction unsupervised
and the GT-orient fixes the sign; orient_sign logic.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "elicitation"))
import ccs as C  # noqa: E402


def test_loss_formula_at_half():
    # w=0,b=0 -> pp=pn=0.5: consistency 0, confidence min(.5,.5)^2 = .25
    w = np.zeros(1)
    xp = np.zeros((3, 1)); xn = np.zeros((3, 1))
    tot, con, conf = C.ccs_loss(w, 0.0, xp, xn)
    assert abs(con - 0.0) < 1e-12
    assert abs(conf - 0.25) < 1e-12
    assert abs(tot - 0.25) < 1e-12


def test_confidence_penalizes_uncertain():
    xp = np.array([[1.0], [1.0]]); xn = np.array([[-1.0], [-1.0]])
    l_uncertain = C.ccs_loss(np.zeros(1), 0.0, xp, xn)[0]      # all 0.5
    l_confident = C.ccs_loss(np.array([3.0]), 0.0, xp, xn)[0]  # pp high, pn low: consistent+confident
    assert l_confident < l_uncertain


def test_grad_matches_finite_difference():
    rng = np.random.default_rng(0)
    xp = rng.normal(0, 1, (25, 5)); xn = rng.normal(0, 1, (25, 5))
    w = rng.normal(0, 1, 5); b = 0.3
    gw, gb = C._grad(w, b, xp, xn)
    eps = 1e-6
    for i in range(5):
        wp = w.copy(); wp[i] += eps
        wm = w.copy(); wm[i] -= eps
        num = (C.ccs_loss(wp, b, xp, xn)[0] - C.ccs_loss(wm, b, xp, xn)[0]) / (2 * eps)
        assert abs(num - gw[i]) < 1e-5, (i, num, gw[i])
    numb = (C.ccs_loss(w, b + eps, xp, xn)[0] - C.ccs_loss(w, b - eps, xp, xn)[0]) / (2 * eps)
    assert abs(numb - gb) < 1e-5


def _planted_contrast(n=400, d=16, sig=4.0, noise=1.0, seed=1):
    """phi_pos encodes truth of 'answer is true' (= y); phi_neg encodes truth of 'answer is false'
    (= 1-y), each along a shared truth axis t, plus distinct per-framing prompt offsets + noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = rng.normal(0, 1, d); t /= np.linalg.norm(t)
    pos_prompt = rng.normal(0, 1, d); neg_prompt = rng.normal(0, 1, d)
    phi_pos = y[:, None] * sig * t[None, :] + pos_prompt[None, :] + rng.normal(0, noise, (n, d))
    phi_neg = (1 - y)[:, None] * sig * t[None, :] + neg_prompt[None, :] + rng.normal(0, noise, (n, d))
    return phi_pos, phi_neg, y


def test_ccs_recovers_truth_unsupervised():
    phi_pos, phi_neg, y = _planted_contrast()
    idx = np.random.default_rng(2).choice(len(y), 16, replace=False)   # 16 GT labels to orient sign
    acc, preds, probe = C.ccs_eval(phi_pos, phi_neg, y, idx, n_restarts=10, seed=0)
    assert acc > 0.85, acc


def test_orient_sign():
    probs = np.array([0.9, 0.8, 0.1, 0.2])
    idx = np.array([0, 1, 2, 3])
    assert C.orient_sign(probs, np.array([1, 1, 0, 0]), idx) is False   # already aligned
    assert C.orient_sign(probs, np.array([0, 0, 1, 1]), idx) is True    # inverted -> flip


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_ccs: ALL PASS")
