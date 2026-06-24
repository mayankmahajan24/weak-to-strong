#!/usr/bin/env python3
"""Phase 1 extra plots — PGR-vs-fraction, the no-knee diagnostic, mixing-vs-gt_only gap.

Complements plot_phase1.py. BoolQ, seeds 0,1,2, EXCLUDE={(1,'gpt2-large')}. Ceilings from baseline.
  phase1_pgr_vs_fraction.png   median PGR vs GT fraction (xent vs logconf) — no knee, logconf negative
  phase1_knee_diagnostic.png   acc vs fraction with the linear frac0→frac1 reference (curve below=back-loaded, not concave)
  phase1_mixing_vs_gtonly.png  mixing − gt_only gap vs fraction (the Phase-1 confound, resolved in 1b)
"""
import csv, glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1])); import plot_style; plot_style.setup()
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "results/plots"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]; RANK = {m: i for i, m in enumerate(ORDER)}
EXCLUDE = {(1, "gpt2-large")}; SEEDS = [0, 1, 2]
strict = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]
def exc(seed, m): return (seed, m) in EXCLUDE

# All Phase-1 data from the canonical csv (one consistent source — avoids the phase-0
# baseline-dir vs phase-1-sweep frac0 discrepancy of ~0.018). The csv 'baseline' condition holds
# both the GT ceilings (weak_model='') and the frac0 transfers (weak_model set).
mixing, gtonly, gt = {}, {}, {}
for r in csv.DictReader((ROOT / "results/phase1/phase1_results.csv").open()):
    if r["ds_name"] != "boolq": continue
    f, s, loss, acc = float(r["gt_fraction_requested"]), int(r["seed"]), r["loss"], float(r["accuracy"])
    w, sg = r["weak_model"], r["strong_model"]
    if r["condition"] == "mixing": mixing[(loss, w, sg, f, s)] = acc
    elif r["condition"] == "gt_only": gtonly[(loss, sg, f, s)] = acc
    elif r["condition"] == "baseline":
        if w == "":                              # GT ceiling (loss-agnostic; take xent)
            if loss == "xent": gt[(sg, s)] = acc
        elif w != sg:                            # frac0 weak→strong transfer = the f=0 point
            mixing[(loss, w, sg, 0.0, s)] = acc
def med(v): return st.median(v) if v else None
FRACS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

# ===== PGR vs fraction =====
fig, ax = plt.subplots(figsize=(8, 5))
for loss, col in [("xent", "tab:blue"), ("logconf", "tab:red")]:
    ys = []
    for f in FRACS:
        pgrs = []
        for sg, w in strict:
            for s in SEEDS:
                if exc(s, w) or exc(s, sg): continue
                if (loss, w, sg, f, s) in mixing and (w, s) in gt and (sg, s) in gt and gt[(sg, s)] - gt[(w, s)] > 0:
                    pgrs.append((mixing[(loss, w, sg, f, s)] - gt[(w, s)]) / (gt[(sg, s)] - gt[(w, s)]))
        ys.append(med(pgrs))
    ax.plot(FRACS, ys, "o-", color=col, label=loss)
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("GT fraction"); ax.set_ylabel("median PGR over strict pairs")
ax.set_title("Phase 1 — PGR vs GT fraction (BoolQ): gradual/back-loaded, no knee; logconf stays negative")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "phase1_pgr_vs_fraction.png", dpi=140); plt.close(fig)
print("wrote phase1_pgr_vs_fraction.png")

# ===== knee diagnostic — PAIRED Δacc(mixing(f) − own frac0) vs fraction =====
# Paired per-(pair,seed) contrast (each pair differenced against ITS OWN frac0 transfer) cancels
# per-pair baseline level → the metric the Phase-1 'within noise' claim was based on.
fig, ax = plt.subplots(figsize=(8.5, 5))
FLOOR = 0.014
def paired_delta(loss, f):
    ds = []
    for sg, w in strict:
        for s in SEEDS:
            if exc(s, w) or exc(s, sg): continue
            if (loss, w, sg, f, s) in mixing and (loss, w, sg, 0.0, s) in mixing:
                ds.append(mixing[(loss, w, sg, f, s)] - mixing[(loss, w, sg, 0.0, s)])
    return med(ds)
fx = [f for f in FRACS]
ys = [paired_delta("xent", f) for f in fx]
fx = [f for f, y in zip(fx, ys) if y is not None]; ys = [y for y in ys if y is not None]
ax.plot(fx, ys, "o-", color="tab:blue", lw=2, label="paired Δacc(mixing − frac0), xent")
y1 = ys[-1]
ax.plot([0, 1], [0, y1], "k--", lw=1.2, label="constant marginal value (linear ref)")
ax.fill_between(fx, ys, [y1 * f for f in fx], where=[a < y1 * f for a, f in zip(ys, fx)], color="tab:blue", alpha=0.12)
ax.axhspan(-FLOOR, FLOOR, color="gray", alpha=0.15, label="noise floor ±0.014")
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("GT fraction"); ax.set_ylabel("median paired Δacc vs frac0")
ax.annotate("≤0.10 within noise; value appears at 0.25,\nrobust at 0.50, largest at 1.0 → back-loaded, NOT a frugal knee",
            (0.12, y1 * 0.55), fontsize=8.5, color="tab:blue")
ax.set_title("Phase 1 — no-knee diagnostic (paired): a frugal knee bows ABOVE the line; ours sits on/below")
ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "phase1_knee_diagnostic.png", dpi=140); plt.close(fig)
print("wrote phase1_knee_diagnostic.png")

# ===== mixing − gt_only gap vs fraction =====
fig, ax = plt.subplots(figsize=(8, 5))
for loss, col in [("xent", "tab:blue"), ("logconf", "tab:red")]:
    ys = []
    for f in FRACS:
        diffs = []
        for sg, w in strict:
            for s in SEEDS:
                if exc(s, w) or exc(s, sg): continue
                if (loss, w, sg, f, s) in mixing and (loss, sg, f, s) in gtonly:
                    diffs.append(mixing[(loss, w, sg, f, s)] - gtonly[(loss, sg, f, s)])
        ys.append(med(diffs))
    ax.plot(FRACS, ys, "o-", color=col, label=loss)
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("GT fraction"); ax.set_ylabel("median (mixing − gt_only)")
ax.set_title("Phase 1 — mixing beats gt_only at every budget < 1.0\n(confounded by data quantity in Phase 1 → de-confounded in Phase 1b · A)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "phase1_mixing_vs_gtonly.png", dpi=140); plt.close(fig)
print("wrote phase1_mixing_vs_gtonly.png")
print("PHASE 1 EXTRA PLOTS DONE")
