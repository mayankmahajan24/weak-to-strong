#!/usr/bin/env python3
"""Phase 2 M2 unit test — soft-GT targets in apply_label_mixing (needs `datasets`).

Verifies: eps=0 reproduces one-hot GT exactly (no-op invariant); eps>0 label-smooths only the
GT-selected rows; non-GT rows and the GT selection set are unchanged.
"""
import sys
from pathlib import Path

import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weak_to_strong.label_mixing import apply_label_mixing, select_gt_indices  # noqa: E402

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


# synthetic weak-label dataset: weak soft labels distinct from one-hot so we can tell them apart
n = 20
ds = datasets.Dataset.from_dict({
    "gt_label": [i % 2 for i in range(n)],
    "hard_label": [(i + 1) % 2 for i in range(n)],          # weak disagrees with gt everywhere
    "soft_label": [[0.3, 0.7] for _ in range(n)],            # arbitrary weak soft label
})
frac, seed = 0.25, 0
gt_idx = select_gt_indices(n, ds["gt_label"], ds["hard_label"], frac, seed, "naive")

# eps = 0 ⇒ one-hot GT exactly (no-op invariant)
m0 = apply_label_mixing(ds, frac, seed, strategy="naive", soft_gt_eps=0.0)
ok_onehot = all(
    m0[i]["soft_label"] == [1.0 - m0[i]["gt_label"], float(m0[i]["gt_label"])]
    for i in gt_idx
)
check("eps=0 ⇒ GT rows one-hot (no-op)", ok_onehot)

# eps > 0 ⇒ GT rows label-smoothed to [eps, 1-eps] toward the GT class; hard_label unchanged
eps = 0.1
me = apply_label_mixing(ds, frac, seed, strategy="naive", soft_gt_eps=eps)
def smoothed(gt):
    p1 = (1 - eps) if gt == 1 else eps
    return [1 - p1, p1]
check("eps>0 ⇒ GT rows label-smoothed",
      all(max(abs(a - b) for a, b in zip(me[i]["soft_label"], smoothed(me[i]["gt_label"]))) < 1e-9
          for i in gt_idx))
check("eps>0 ⇒ GT hard_label still argmax-correct",
      all(me[i]["hard_label"] == me[i]["gt_label"] for i in gt_idx))
check("GT-row soft_label peaks on the GT class",
      all((me[i]["soft_label"][1] > 0.5) == (me[i]["gt_label"] == 1) for i in gt_idx))

# non-GT rows untouched (keep their weak soft label) and selection set identical to naive
weak_rows = [i for i in range(n) if i not in gt_idx]
check("non-GT rows keep weak soft_label", all(me[i]["soft_label"] == [0.3, 0.7] for i in weak_rows))
check("non-GT rows tagged weak", all(me[i]["label_source"] == "weak" for i in weak_rows))
check("selection set unchanged by eps",
      {i for i in range(n) if me[i]["label_source"] == "gt"} == set(gt_idx))

print(f"\nALL {PASS} CHECKS PASSED")
