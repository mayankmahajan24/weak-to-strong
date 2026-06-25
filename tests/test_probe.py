#!/usr/bin/env python3
"""Unit tests for Method 1 — k-shot supervised probe (numpy-only, CPU).

Checks: recovers a planted linear signal; standardization uses train stats only (no test leakage);
more labels never hurt and beat chance; determinism; balanced k-shot sampling.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "elicitation"))
import probe as P  # noqa: E402


def make_data(n, d, noise, seed, scale=2.0):
    """Plant a linear truth direction u: x = (2y-1)*scale*u + gaussian noise."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, d)
    u /= np.linalg.norm(u)
    y = rng.integers(0, 2, n)
    X = (2 * y - 1)[:, None] * scale * u[None, :] + rng.normal(0, noise, (n, d))
    return X, y


def test_fit_separable():
    X, y = make_data(400, 20, 1.0, 0)
    pr = P.fit_logistic(X[:200], y[:200])
    assert P.accuracy(pr, X[200:], y[200:]) > 0.9


def test_standardization_uses_train_stats_only():
    X, y = make_data(200, 10, 1.0, 1)
    pr = P.fit_logistic(X[:100], y[:100])
    assert np.allclose(pr["mu"], X[:100].mean(0))            # train mean, not pooled/test
    assert np.allclose(pr["sd"], X[:100].std(0) + 1e-6)
    # a constant shift of the test inputs must change predictions (probe applies fixed train mu)
    p1 = P.predict_proba(pr, X[100:])
    p2 = P.predict_proba(pr, X[100:] + 5.0)
    assert not np.allclose(p1, p2)


def test_kshot_more_labels_help_and_beat_chance():
    # signal strong enough that a large k is clearly good, but d=30 noise dims make small k hard
    X, y = make_data(2400, 30, 1.8, 2, scale=3.0)
    Xtr, ytr, Xte, yte = X[:1200], y[:1200], X[1200:], y[1200:]
    a8 = np.mean([P.kshot_eval(Xtr, ytr, Xte, yte, 8, seed=s)[0] for s in range(5)])
    a256 = np.mean([P.kshot_eval(Xtr, ytr, Xte, yte, 256, seed=s)[0] for s in range(5)])
    assert a8 > 0.5, a8                          # even tiny k beats chance on a real signal
    assert a256 > a8 + 0.03, (a8, a256)          # more labels help (clear margin)
    assert a256 > 0.8, a256


def test_determinism():
    X, y = make_data(500, 10, 2.0, 3)
    a, p = P.kshot_eval(X[:250], y[:250], X[250:], y[250:], 32, seed=7)
    b, q = P.kshot_eval(X[:250], y[:250], X[250:], y[250:], 32, seed=7)
    assert a == b and np.allclose(p, q)


def test_sample_k_balanced():
    y = np.array([0] * 80 + [1] * 20)
    idx = P.sample_k(y, 20, seed=0, balanced=True)
    assert len(idx) == 20 and len(set(idx)) == 20      # exactly k unique indices
    assert (y[idx] == 1).sum() == 10                   # balanced: 10/10 (class 1 has enough)
    # if k exceeds n, return all
    assert len(P.sample_k(y, 1000, seed=0)) == len(y)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_probe: ALL PASS")
