#!/usr/bin/env python3
"""Method 2 — Contrastive Consistency Search (CCS) + GT-orient (unsupervised elicitation).

CCS (Burns et al. 2022, "Discovering Latent Knowledge") learns a direction in the frozen model's
activation space that behaves like a truth predictor, *without labels*, from contrast pairs
(phi_pos = activations of "the answer is true", phi_neg = "...false"). It minimizes:
    consistency = mean[(p_pos - (1 - p_neg))^2]      # the two framings should disagree in prob
    confidence  = mean[min(p_pos, p_neg)^2]          # predictions should be confident (not 0.5)
The learned probe is sign-ambiguous; we use a few GT labels only to fix the sign (`orient_sign`),
plus pick the layer/restart. Pure numpy → unit-tested on CPU.

API:
  probe = fit_ccs(phi_pos, phi_neg, n_restarts=10)
  p     = ccs_predict(probe, phi_pos, phi_neg)       # truth probability per example (pre-orient)
  flip  = orient_sign(p, gt_labels, idx_labeled)     # True if the direction is inverted
  preds = (p > 0.5) ^ flip
"""
import numpy as np


def _norm_stats(phi):
    phi = np.asarray(phi, dtype=np.float64)
    mu = phi.mean(0)
    sd = phi.std(0) + 1e-6
    return mu, sd


def _apply(phi, mu, sd):
    return (np.asarray(phi, dtype=np.float64) - mu) / sd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ccs_loss(w, b, xp, xn):
    """Return (total, consistency, confidence) for normalized contrast activations xp, xn."""
    pp = _sigmoid(xp @ w + b)
    pn = _sigmoid(xn @ w + b)
    consistency = np.mean((pp - (1.0 - pn)) ** 2)
    confidence = np.mean(np.minimum(pp, pn) ** 2)
    return consistency + confidence, consistency, confidence


def _grad(w, b, xp, xn):
    """Analytic gradient of ccs_loss wrt (w, b)."""
    zp = xp @ w + b
    zn = xn @ w + b
    pp = _sigmoid(zp)
    pn = _sigmoid(zn)
    n = xp.shape[0]
    # consistency term c = (pp - 1 + pn); dL_c/dpp = 2c, dL_c/dpn = 2c
    c = pp - 1.0 + pn
    dpp = 2.0 * c
    dpn = 2.0 * c
    # confidence term: min(pp,pn)^2 -> derivative flows only to the smaller one
    mp = (pp <= pn).astype(np.float64)
    dpp += 2.0 * np.minimum(pp, pn) * mp
    dpn += 2.0 * np.minimum(pp, pn) * (1.0 - mp)
    # chain through sigmoid: dp/dz = p(1-p); dz/dw = x, dz/db = 1
    gp = dpp * pp * (1.0 - pp)
    gn = dpn * pn * (1.0 - pn)
    gw = (xp.T @ gp + xn.T @ gn) / n
    gb = float((gp + gn).mean())
    return gw, gb


def fit_ccs(phi_pos, phi_neg, n_restarts=10, lr=1.0, steps=1000, seed=0):
    """Fit CCS over `n_restarts` random inits; keep the lowest unsupervised loss."""
    mu_p, sd_p = _norm_stats(phi_pos)
    mu_n, sd_n = _norm_stats(phi_neg)
    xp = _apply(phi_pos, mu_p, sd_p)
    xn = _apply(phi_neg, mu_n, sd_n)
    d = xp.shape[1]
    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        w = rng.normal(0, 1, d)
        w /= np.linalg.norm(w) + 1e-8
        b = 0.0
        for _ in range(steps):
            gw, gb = _grad(w, b, xp, xn)
            w -= lr * gw
            b -= lr * gb
        loss = ccs_loss(w, b, xp, xn)[0]
        if best is None or loss < best["loss"]:
            best = {"w": w.copy(), "b": b, "loss": float(loss),
                    "mu_p": mu_p, "sd_p": sd_p, "mu_n": mu_n, "sd_n": sd_n}
    return best


def ccs_predict(probe, phi_pos, phi_neg):
    """Truth probability per example: average the two consistent estimates. Pre-orientation."""
    xp = _apply(phi_pos, probe["mu_p"], probe["sd_p"])
    xn = _apply(phi_neg, probe["mu_n"], probe["sd_n"])
    pp = _sigmoid(xp @ probe["w"] + probe["b"])
    pn = _sigmoid(xn @ probe["w"] + probe["b"])
    return 0.5 * (pp + (1.0 - pn))


def orient_sign(probs, gt_labels, idx_labeled):
    """Resolve CCS sign ambiguity using a few GT labels. Returns True if predictions should flip
    (i.e. the learned direction points to 'false'). Decided only on the labeled indices."""
    preds = (np.asarray(probs)[idx_labeled] > 0.5).astype(int)
    gt = np.asarray(gt_labels)[idx_labeled].astype(int)
    acc = float((preds == gt).mean())
    return acc < 0.5


def ccs_eval(phi_pos, phi_neg, gt_labels, idx_labeled, n_restarts=10, seed=0):
    """Fit CCS, orient with the labeled subset, return (test_accuracy_on_all, oriented_preds)."""
    probe = fit_ccs(phi_pos, phi_neg, n_restarts=n_restarts, seed=seed)
    p = ccs_predict(probe, phi_pos, phi_neg)
    flip = orient_sign(p, gt_labels, idx_labeled)
    preds = (p > 0.5).astype(int) ^ int(flip)
    acc = float((preds == np.asarray(gt_labels).astype(int)).mean())
    return acc, preds, probe
