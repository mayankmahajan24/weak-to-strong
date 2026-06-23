#!/usr/bin/env python3
"""Phase 2A unit tests — loss plumbing + WeightedXentLoss.

Covers: A1 (existing losses accept-and-ignore the aux kwargs the loop now passes) and the
WeightedXentLoss reference consumer, including INVARIANT 5 (gt_weight=1 / no sample_weight
reduces EXACTLY to xent_loss). Run with the scratch venv that has CPU torch:
  <venv>/bin/python tests/test_losses.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weak_to_strong.loss import (  # noqa: E402
    xent_loss, logconf_loss_fn, product_loss_fn, WeightedXentLoss)

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


torch.manual_seed(0)
N = 8
logits = torch.randn(N, 2)
hard = (torch.rand(N) < 0.5).long()
labels = torch.nn.functional.one_hot(hard, 2).float()       # soft labels (probabilities)
gt_mask = torch.tensor([True, False] * (N // 2))
sw = torch.rand(N) + 0.5
sf = 0.5

# --- A1: existing losses ignore the aux kwargs the loop always passes ---
for name, fn in [("xent", xent_loss()), ("logconf", logconf_loss_fn()), ("product", product_loss_fn())]:
    base = fn(logits, labels, step_frac=sf)
    withkw = fn(logits, labels, step_frac=sf, gt_mask=gt_mask, sample_weight=sw)
    check(f"A1 {name} ignores gt_mask/sample_weight", torch.allclose(base, withkw))

# --- INVARIANT 5: WeightedXentLoss with gt_weight=1 and no sample_weight == xent_loss ---
xent = xent_loss()(logits, labels, sf)
check("Inv5 WX(gt_weight=1, no sw) == xent",
      torch.allclose(WeightedXentLoss(1.0)(logits, labels, sf, gt_mask=gt_mask), xent, atol=1e-6))
check("Inv5 WX(gt_weight=1, sw=ones) == xent",
      torch.allclose(WeightedXentLoss(1.0)(logits, labels, sf, sample_weight=torch.ones(N)), xent, atol=1e-6))

# --- all-GT batch: uniform weight cancels regardless of gt_weight ---
check("WX all-GT cancels (gt_weight=4)",
      torch.allclose(WeightedXentLoss(4.0)(logits, labels, sf, gt_mask=torch.ones(N, dtype=torch.bool)),
                     xent, atol=1e-6))

# --- hand-computed closed forms ---
ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
wvec = 1.0 + (4.0 - 1.0) * gt_mask.float()
check("WX gt-weighted mean matches closed form",
      torch.allclose(WeightedXentLoss(4.0)(logits, labels, sf, gt_mask=gt_mask),
                     (wvec * ce).sum() / wvec.sum(), atol=1e-6))
check("WX sample_weight matches closed form",
      torch.allclose(WeightedXentLoss(1.0)(logits, labels, sf, sample_weight=sw),
                     (sw * ce).sum() / sw.sum(), atol=1e-6))

# --- gt_weight>1 actually upweights GT rows (sanity direction) ---
# make GT rows the high-loss ones; increasing gt_weight should raise the weighted loss
check("WX gt_weight>1 increases weight on GT rows",
      WeightedXentLoss(8.0)(logits, labels, sf, gt_mask=gt_mask).item()
      != WeightedXentLoss(1.0)(logits, labels, sf, gt_mask=gt_mask).item())

print(f"\nALL {PASS} CHECKS PASSED")
