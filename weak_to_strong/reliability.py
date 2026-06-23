"""Phase 2 M4 — teacher-reliability weighting.

Reframes the strong-label budget as a *calibration resource*: on the GT-revealed subset we
observe whether the weak teacher was correct (`hard_label == gt_label`); fit P(teacher correct |
feature) there (1-D logistic on weak confidence `max(soft_label)`), predict reliability for the
remaining weak rows, and downweight low-reliability weak labels in the loss (via the
`sample_weight` column that `WeightedXentLoss` consumes). GT rows keep weight 1.0.

Pure-Python logistic fit (no numpy/sklearn) so the core is testable anywhere. Must be applied to
the RAW weak-label dataset *before* `apply_label_mixing` overwrites the GT rows' weak predictions.
"""
import math

from .label_mixing import select_gt_indices


def _fit_logistic_1d(xs, ys, steps=1500, lr=0.3):
    """Fit P(y=1|x)=sigmoid(w·z+b) on standardized x via gradient descent. Returns a model
    tuple. Degenerate (n=0 or single-class y) → a constant predictor at the empirical rate."""
    n = len(xs)
    if n == 0 or len(set(ys)) < 2:
        p = (sum(ys) / n) if n else 0.5
        return ("const", min(max(p, 1e-6), 1 - 1e-6))
    mx = sum(xs) / n
    var = sum((x - mx) ** 2 for x in xs) / n
    sx = math.sqrt(var) or 1.0
    xz = [(x - mx) / sx for x in xs]
    w, b = 0.0, 0.0
    for _ in range(steps):
        gw = gb = 0.0
        for xi, yi in zip(xz, ys):
            p = 1.0 / (1.0 + math.exp(-(w * xi + b)))
            gw += (p - yi) * xi
            gb += (p - yi)
        w -= lr * gw / n
        b -= lr * gb / n
    return ("lr", (w, b, mx, sx))


def _predict(model, x):
    kind, params = model
    if kind == "const":
        return params
    w, b, mx, sx = params
    return 1.0 / (1.0 + math.exp(-(w * (x - mx) / sx + b)))


def compute_reliability_weights(gt_labels, hard_labels, confidences, gt_indices):
    """Pure core. Fit reliability on the GT subset, predict for all rows.
    Returns per-row weights: 1.0 on GT rows, predicted P(teacher correct) on weak rows."""
    gt_indices = set(gt_indices)
    xs = [confidences[i] for i in sorted(gt_indices)]
    ys = [int(hard_labels[i] == gt_labels[i]) for i in sorted(gt_indices)]
    model = _fit_logistic_1d(xs, ys)
    out = []
    for i in range(len(gt_labels)):
        out.append(1.0 if i in gt_indices else float(_predict(model, confidences[i])))
    return out


def add_reliability_weights(ds, gt_fraction, gt_seed, strategy="naive"):
    """Attach a `sample_weight` column to a RAW weak-label dataset (before label mixing).
    `ds` must have gt_label, hard_label, soft_label."""
    n = len(ds)
    gt_indices = select_gt_indices(n, ds["gt_label"], ds["hard_label"], gt_fraction, gt_seed, strategy)
    confidences = [max(sl) for sl in ds["soft_label"]]
    weights = compute_reliability_weights(ds["gt_label"], ds["hard_label"], confidences, gt_indices)
    return ds.map(lambda ex, idx: {"sample_weight": weights[idx]}, with_indices=True)
