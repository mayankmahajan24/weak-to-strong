"""
Label mixing: replace a fraction of weak labels with ground-truth labels.

Strategies (which rows get GT, and what the non-GT rows carry):
  - "naive"         : GT on a uniform-random subset; non-GT rows keep their weak labels.
  - "oracle"        : GT on the rows the weak teacher got WRONG first (gt != hard), then
                      fill with random correct rows if the budget exceeds the error count.
                      Non-GT rows keep their weak labels. Upper bound on allocation value.
  - "random_labels" : GT on the SAME subset as naive, but non-GT rows get a random
                      Bernoulli(0.5) label instead of the weak label. De-confounds
                      "mixing > GT-only": holds row/step count fixed, removes weak-label
                      information. (Phase 1b Component A.)

Index selection is factored into `select_gt_indices` (pure, no `datasets` dependency) so
the allocation logic is unit-testable on plain lists / real weak_labels columns.
"""

import random

STRATEGIES = ("naive", "oracle", "random_labels")


def select_gt_indices(n, gt_labels, hard_labels, gt_fraction, gt_seed, strategy="naive"):
    """Return the set of row indices that receive ground-truth labels.

    Pure function — takes plain sequences, returns a set[int]. Deterministic in gt_seed.
    `hard_labels` (weak predictions) is only consulted by the "oracle" strategy.
    """
    assert strategy in STRATEGIES, f"Unknown strategy: {strategy}"
    assert 0.0 <= gt_fraction <= 1.0
    k = round(gt_fraction * n)
    if k == 0:
        return set()
    if k >= n:
        return set(range(n))

    rng = random.Random(gt_seed)
    if strategy in ("naive", "random_labels"):
        return set(rng.sample(range(n), k=k))

    if strategy == "oracle":
        wrong = [i for i in range(n) if gt_labels[i] != hard_labels[i]]
        if len(wrong) >= k:
            return set(rng.sample(sorted(wrong), k=k))
        # all wrong rows + fill the remainder from the correct rows
        correct = [i for i in range(n) if gt_labels[i] == hard_labels[i]]
        fill = rng.sample(sorted(correct), k=k - len(wrong))
        return set(wrong) | set(fill)

    raise AssertionError(strategy)  # unreachable


def apply_label_mixing(ds, gt_fraction, gt_seed, strategy="naive"):
    """
    Mix ground-truth labels into a weak-label dataset.

    For GT-selected rows, soft_label becomes [1 - gt_label, gt_label] and hard_label
    becomes gt_label. For "random_labels", non-GT rows get a deterministic Bernoulli(0.5)
    label; for other strategies non-GT rows keep their weak labels. Every row gets a
    `label_source` field ("gt", "weak", or "random") for provenance.

    Returns a new dataset (does not mutate the input).
    """
    assert strategy in STRATEGIES, f"Unknown mixing strategy: {strategy}"
    assert 0.0 <= gt_fraction <= 1.0

    if gt_fraction == 0.0:
        return ds.map(lambda ex: {"label_source": "weak"})

    n = len(ds)
    gt_indices = select_gt_indices(
        n, ds["gt_label"], ds["hard_label"], gt_fraction, gt_seed, strategy
    )

    # Deterministic random labels for the non-GT rows (random_labels strategy only),
    # seeded independently of the selection RNG and assigned in index order.
    rand_label = {}
    if strategy == "random_labels":
        rng = random.Random(gt_seed * 1_000_003 + 1)
        for i in range(n):
            if i not in gt_indices:
                rand_label[i] = int(rng.random() < 0.5)

    def _mix(ex, idx):
        if idx in gt_indices:
            gt = ex["gt_label"]
            return {"soft_label": [1.0 - gt, float(gt)], "hard_label": gt, "label_source": "gt"}
        if strategy == "random_labels":
            r = rand_label[idx]
            return {"soft_label": [1.0 - r, float(r)], "hard_label": r, "label_source": "random"}
        return {"label_source": "weak"}

    return ds.map(_mix, with_indices=True)
