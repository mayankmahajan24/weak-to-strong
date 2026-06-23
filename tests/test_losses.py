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
    xent_loss, logconf_loss_fn, product_loss_fn, WeightedXentLoss, GTAnchoredLogconfLoss)

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

# --- M3: GTAnchoredLogconfLoss ---
# (a) gt_mask all-False == stock logconf_loss_fn (regression invariant)
none_mask = torch.zeros(N, dtype=torch.bool)
check("M3 all-weak == logconf_loss_fn",
      torch.allclose(GTAnchoredLogconfLoss()(logits, labels, sf, gt_mask=none_mask),
                     logconf_loss_fn()(logits, labels, sf), atol=1e-6))
check("M3 (no gt_mask) == logconf_loss_fn",
      torch.allclose(GTAnchoredLogconfLoss()(logits, labels, sf),
                     logconf_loss_fn()(logits, labels, sf), atol=1e-6))
# (b) gt_mask all-True ⇒ pure CE on the (hard) labels (coef→0, no self-prediction blend)
all_mask = torch.ones(N, dtype=torch.bool)
check("M3 all-GT == cross_entropy(logits, labels)",
      torch.allclose(GTAnchoredLogconfLoss()(logits, labels, sf, gt_mask=all_mask),
                     torch.nn.functional.cross_entropy(logits, labels), atol=1e-6))
# (c) mixed mask: matches a manual reconstruction (GT rows → hard labels, weak rows → blend)
_l = logits.float(); _lab = labels.float()
_coef = 0.5  # step_frac 0.5 > warmup 0.1 -> coef = 1.0 * aux_coef(0.5)
_preds = torch.softmax(_l, dim=-1)
_thr = torch.quantile(_preds[:, 0], torch.mean(_lab, dim=0)[1])
_sp = torch.cat([(_preds[:, 0] >= _thr)[:, None], (_preds[:, 0] < _thr)[:, None]], dim=1).float()
_cr = (_coef * (~gt_mask).float())[:, None]
_target = _lab * (1 - _cr) + _sp * _cr
_expected = torch.nn.functional.cross_entropy(_l, _target, reduction="none").mean()
check("M3 mixed-mask matches manual reconstruction",
      torch.allclose(GTAnchoredLogconfLoss()(logits, labels, sf, gt_mask=gt_mask), _expected, atol=1e-6))

print(f"\nALL {PASS} CHECKS PASSED")
