"""
Label mixing: replace a fraction of weak labels with ground-truth labels.

The only strategy implemented now is "naive" (uniform random selection).
Adding a new strategy is a one-function change — add a branch here.
"""

import random


def apply_label_mixing(ds, gt_fraction, gt_seed, strategy="naive"):
    """
    Mix ground-truth labels into a weak-label dataset.

    For selected rows, soft_label becomes [1 - gt_label, gt_label] and
    hard_label becomes gt_label.  Every row gets a `label_source` field
    ("gt" or "weak") for provenance.

    Returns a new dataset (does not mutate the input).
    """
    assert strategy == "naive", f"Unknown mixing strategy: {strategy}"
    assert 0.0 <= gt_fraction <= 1.0

    if gt_fraction == 0.0:
        return ds.map(lambda ex: {"label_source": "weak"})

    n = len(ds)
    k = round(gt_fraction * n)
    gt_indices = set(random.Random(gt_seed).sample(range(n), k=k))

    def _mix(ex, idx):
        if idx in gt_indices:
            gt = ex["gt_label"]
            return {
                "soft_label": [1.0 - gt, float(gt)],
                "hard_label": gt,
                "label_source": "gt",
            }
        return {"label_source": "weak"}

    return ds.map(_mix, with_indices=True)
