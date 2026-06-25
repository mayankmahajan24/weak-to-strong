#!/usr/bin/env python3
"""Elicitation driver — turn extracted activations into the accuracy grid (CPU, numpy-only).

For one (ds, model, seed): load the train/test .npz from extract_activations.py, then
  - select a layer using a *train-pool* validation split (no test-set selection),
  - M1 k-shot probe: test accuracy vs k (mean over N_KSEED label samples),
  - M1 full-supervised probe: the linear-readout upper bound,
  - M2 CCS (if contrast activations present): fit unsupervised on train contrasts, orient with k
    train labels, evaluate on test; plus a random-direction control,
and write results/elicitation/runs/<ds>_<model>_s<seed>.json.

GPU is only used by extract_activations.py; this step is pure numpy and runs anywhere.
Usage: python run_elicitation.py --ds=boolq --model_size=gpt2-xl --seed=0 --acts=results/elicitation/acts
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ccs as C
import probe as P

KSWEEP = [8, 16, 32, 64, 128, 256]
N_KSEED = 5
VAL_FRAC = 0.25  # of the train pool, for layer selection (kept away from test)


def _load(acts_dir, ds, model, seed, split):
    fn = os.path.join(acts_dir, f"{ds}_{model}_s{seed}_{split}.npz")
    return np.load(fn, allow_pickle=True)


def select_layer(train, layers, seed):
    """Pick the layer whose full-supervised probe generalizes best on a held-out slice of the
    train pool. Uses only train activations/labels — never the test set."""
    y = train["hard_label"]
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(50, int(VAL_FRAC * n))
    val, fit = perm[:n_val], perm[n_val:]
    best, best_acc = layers[0], -1.0
    val_accs = {}
    for L in layers:
        X = train[f"acts_L{L}"]
        pr = P.fit_logistic(X[fit], y[fit])
        a = P.accuracy(pr, X[val], y[val])
        val_accs[int(L)] = a
        if a > best_acc:
            best_acc, best = a, int(L)
    return best, val_accs


def run(acts_dir, ds, model, seed):
    train = _load(acts_dir, ds, model, seed, "train")
    test = _load(acts_dir, ds, model, seed, "test")
    layers = [int(L) for L in train["layers"]]
    ytr, yte = train["hard_label"], test["hard_label"]

    layer, val_accs = select_layer(train, layers, seed)
    Xtr, Xte = train[f"acts_L{layer}"], test[f"acts_L{layer}"]

    # M1 k-shot probe (mean over N_KSEED label draws) + full-supervised upper bound
    m1 = {}
    for k in KSWEEP:
        accs = [P.kshot_eval(Xtr, ytr, Xte, yte, k, seed=s)[0] for s in range(N_KSEED)]
        m1[k] = float(np.mean(accs))
    m1_full = P.accuracy(P.fit_logistic(Xtr, ytr), Xte, yte)

    out = {"ds": ds, "model": model, "seed": int(seed), "layer": layer,
           "layer_val_accs": val_accs, "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "chance": float(max(yte.mean(), 1 - yte.mean())),
           "m1_probe": m1, "m1_full_supervised": float(m1_full)}

    # M2 CCS + controls (only if contrast activations were extracted)
    if f"pos_L{layer}" in train.files:
        pos_tr, neg_tr = train[f"pos_L{layer}"], train[f"neg_L{layer}"]
        pos_te, neg_te = test[f"pos_L{layer}"], test[f"neg_L{layer}"]
        ccs_probe = C.fit_ccs(pos_tr, neg_tr, n_restarts=10, seed=seed)
        p_tr = C.ccs_predict(ccs_probe, pos_tr, neg_tr)
        p_te = C.ccs_predict(ccs_probe, pos_te, neg_te)
        ccs_acc = {}
        for k in KSWEEP:
            idx_k = P.sample_k(ytr, k, seed=0)
            flip = C.orient_sign(p_tr, ytr, idx_k)
            preds = (p_te > 0.5).astype(int) ^ int(flip)
            ccs_acc[k] = float((preds == yte).mean())
        # random-direction control: a random probe, oriented with the largest budget
        rng = np.random.default_rng(seed + 999)
        w = rng.normal(0, 1, pos_tr.shape[1])
        rand_probe = {"w": w, "b": 0.0,
                      "mu_p": pos_tr.mean(0), "sd_p": pos_tr.std(0) + 1e-6,
                      "mu_n": neg_tr.mean(0), "sd_n": neg_tr.std(0) + 1e-6}
        pr_tr = C.ccs_predict(rand_probe, pos_tr, neg_tr)
        pr_te = C.ccs_predict(rand_probe, pos_te, neg_te)
        flip = C.orient_sign(pr_tr, ytr, P.sample_k(ytr, max(KSWEEP), seed=0))
        rand_acc = float((((pr_te > 0.5).astype(int) ^ int(flip)) == yte).mean())
        out["m2_ccs"] = ccs_acc
        out["m2_ccs_unsup_loss"] = float(ccs_probe["loss"])
        out["control_random_direction"] = rand_acc

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--model_size", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ROOT = Path(__file__).resolve().parents[2]
    ap.add_argument("--acts", default=str(ROOT / "results/elicitation/acts"))
    ap.add_argument("--out", default=str(ROOT / "results/elicitation/runs"))
    a = ap.parse_args()
    res = run(a.acts, a.ds, a.model_size, a.seed)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{a.ds}_{a.model_size}_s{a.seed}.json")
    json.dump(res, open(fn, "w"), indent=2)
    print(f"wrote {fn}")
    print(f"  layer={res['layer']}  chance={res['chance']:.3f}  "
          f"m1@256={res['m1_probe'][256]:.3f}  full={res['m1_full_supervised']:.3f}"
          + (f"  ccs@32={res['m2_ccs'][32]:.3f}  rand={res['control_random_direction']:.3f}"
             if 'm2_ccs' in res else ""))


if __name__ == "__main__":
    main()
