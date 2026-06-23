#!/usr/bin/env python3
"""Phase 2 M4 unit tests — teacher-reliability weighting (pure core, no torch/numpy)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weak_to_strong.reliability import (  # noqa: E402
    _fit_logistic_1d, _predict, compute_reliability_weights)

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


# --- logistic fit recovers a clean confidence→correctness relationship ---
# high confidence => correct, low confidence => wrong
xs = [0.95, 0.9, 0.92, 0.88, 0.55, 0.52, 0.6, 0.58]
ys = [1, 1, 1, 1, 0, 0, 0, 0]
m = _fit_logistic_1d(xs, ys)
check("fit: high-confidence row predicted reliable (>0.7)", _predict(m, 0.95) > 0.7)
check("fit: low-confidence row predicted unreliable (<0.3)", _predict(m, 0.52) < 0.3)
check("fit: monotone (higher conf ⇒ higher reliability)", _predict(m, 0.9) > _predict(m, 0.55))

# --- degenerate GT subsets fall back to a constant rate ---
mc = _fit_logistic_1d([0.8, 0.9, 0.7], [1, 1, 1])      # all correct
check("degenerate all-correct ⇒ constant ≈1", _predict(mc, 0.5) > 0.9)
mz = _fit_logistic_1d([0.8, 0.9, 0.7], [0, 0, 0])      # all wrong
check("degenerate all-wrong ⇒ constant ≈0", _predict(mz, 0.5) < 0.1)
check("empty GT subset ⇒ 0.5", _predict(_fit_logistic_1d([], []), 0.5) == 0.5)

# --- compute_reliability_weights: GT rows weight 1.0, weak rows = predicted reliability ---
# correctness tracks confidence (high-conf ⇒ correct). GT subset spans BOTH so the fit can learn.
conf =       [0.95, 0.90, 0.55, 0.50, 0.92, 0.52, 0.88, 0.60]
gt_labels =  [1,    0,    1,    0,    1,    0,    1,    0]
correct =    [1,    1,    0,    0,    1,    0,    1,    0]   # = (conf >= 0.8)
hard = [g if c else 1 - g for g, c in zip(gt_labels, correct)]
gt_idx = {0, 2, 4, 5}                  # revealed subset: high-correct (0,4) + low-wrong (2,5)
weak_idx = [1, 3, 6, 7]                 # held-out weak rows: conf 0.90, 0.50, 0.88, 0.60
w = compute_reliability_weights(gt_labels, hard, conf, gt_idx)
check("weights: GT rows == 1.0", all(w[i] == 1.0 for i in gt_idx))
check("weights: weak rows in [0,1]", all(0.0 <= w[i] <= 1.0 for i in weak_idx))
check("weights: high-confidence weak rows up-weighted (>0.5)", w[1] > 0.5 and w[6] > 0.5)
check("weights: low-confidence weak rows down-weighted (<0.5)", w[3] < 0.5 and w[7] < 0.5)
check("weights: length matches", len(w) == len(gt_labels))

print(f"\nALL {PASS} CHECKS PASSED")
