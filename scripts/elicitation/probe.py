#!/usr/bin/env python3
"""Method 1 — k-shot supervised linear probe (elicitation readout).

Fit a logistic probe on a frozen strong model's activations using only `k` ground-truth labels,
then read out predictions on the test split. Pure numpy (no torch/sklearn) so it is unit-tested on
CPU. Standardization statistics are fit on the *labeled training* activations only (no test leakage).

Core API:
  probe = fit_logistic(X_train_k, y_train_k)         # train on k labeled examples
  p     = predict_proba(probe, X_test)               # test probabilities
  acc   = (p > 0.5).astype(int) == y_test
  kshot_eval(...) wraps sample-k -> fit -> eval and returns test accuracy + probs.
"""
import numpy as np


def fit_logistic(X, y, l2=1.0, lr=0.5, steps=600, seed=0):
    """Logistic regression by full-batch gradient descent on standardized features.

    X: [n, d] train activations; y: [n] in {0,1}. Returns a probe dict carrying the standardization
    stats so prediction applies the *train* mean/std (no leakage). l2 regularizes weights (not bias).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    assert X.shape[0] == y.shape[0] and X.ndim == 2
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gw = Xs.T @ (p - y) / n + (l2 / n) * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def predict_proba(probe, X):
    """P(y=1 | x) using the train-fit standardization. X: [m, d] -> [m]."""
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - probe["mu"]) / probe["sd"]
    z = Xs @ probe["w"] + probe["b"]
    return 1.0 / (1.0 + np.exp(-z))


def accuracy(probe, X, y):
    return float(((predict_proba(probe, X) > 0.5).astype(int) == np.asarray(y).reshape(-1)).mean())


def sample_k(y, k, seed=0, balanced=True):
    """Pick k labeled-example indices. If balanced, take k//2 from each class when possible."""
    y = np.asarray(y).reshape(-1)
    rng = np.random.default_rng(seed)
    n = len(y)
    if k >= n:
        return np.arange(n)
    if balanced:
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        kp = min(k // 2, len(pos))
        kn = min(k - kp, len(neg))
        kp = min(k - kn, len(pos))  # backfill if one class is short
        idx = np.concatenate([rng.choice(pos, kp, replace=False),
                              rng.choice(neg, kn, replace=False)])
        if len(idx) < k:  # top up from the remainder
            rest = np.setdiff1d(np.arange(n), idx)
            idx = np.concatenate([idx, rng.choice(rest, k - len(idx), replace=False)])
        return np.sort(idx)
    return np.sort(rng.choice(n, k, replace=False))


def kshot_eval(X_train, y_train, X_test, y_test, k, seed=0, l2=1.0, balanced=True):
    """Sample k labeled train examples, fit the probe, evaluate on test. Returns (acc, test_probs)."""
    idx = sample_k(y_train, k, seed=seed, balanced=balanced)
    probe = fit_logistic(X_train[idx], np.asarray(y_train)[idx], l2=l2, seed=seed)
    p = predict_proba(probe, X_test)
    acc = float(((p > 0.5).astype(int) == np.asarray(y_test).reshape(-1)).mean())
    return acc, p
