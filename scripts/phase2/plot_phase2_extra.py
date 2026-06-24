#!/usr/bin/env python3
"""Phase 2 extra plots — Δ-vs-naive bars (the scorecard) + per-pair heatmap (where methods help/hurt).

Complements plot_phase2.py (the overlay). Loss-matched naive (gt_anchored→logconf, rest→xent).
  phase2_delta_bars.png       median Δ(method−naive) per method × fraction, vs floor band
  phase2_per_pair_heatmap.png method × pair Δ@0.50 heatmap (diverging)
"""
import csv, glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "results/plots"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]; RANK = {m: i for i, m in enumerate(ORDER)}
SHORT = {"gpt2": "g2", "gpt2-medium": "M", "gpt2-large": "L", "gpt2-xl": "XL"}
EXCLUDE = {(1, "gpt2-large")}; FLOOR = 0.014; FRACS = [0.1, 0.25, 0.5]
strict = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]
METHOD_LOSS = {"weighted": "xent", "soft_gt": "xent", "reliability": "xent", "gt_early_stop": "xent", "gt_anchored": "logconf"}
def exc(seed, m): return (seed, m) in EXCLUDE

naive = {}
for r in csv.DictReader((ROOT / "results/phase1/phase1_results.csv").open()):
    if r["condition"] == "mixing" and r["ds_name"] == "boolq":
        naive[(r["loss"], int(r["seed"]), r["weak_model"], r["strong_model"], float(r["gt_fraction_requested"]))] = float(r["accuracy"])
meth = {}
for cfgp in glob.glob(str(ROOT / "results/data/phase2_*/**/config.json"), recursive=True):
    if "phase2_sciq" in cfgp: continue
    c = json.load(open(cfgp)); cm = c.get("combination_method")
    if not cm or cm == "naive": continue
    try: a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
    except Exception: continue
    meth[(cm, c["seed"], c.get("weak_model_size"), c["model_size"], c["gt_fraction"])] = a
methods = ["gt_anchored", "soft_gt", "reliability", "weighted", "gt_early_stop"]
methods = [m for m in methods if any(k[0] == m for k in meth)]

def delta(m, f, pair=None):
    loss = METHOD_LOSS[m]; ds = []
    for sg, w in (strict if pair is None else [pair]):
        for s in [0, 1, 2]:
            if exc(s, w) or exc(s, sg): continue
            km, kn = (m, s, w, sg, f), (loss, s, w, sg, f)
            if km in meth and kn in naive: ds.append(meth[km] - naive[kn])
    return st.median(ds) if ds else None

# ===== delta bars =====
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(FRACS)); width = 0.16
colors = {"gt_anchored": "tab:blue", "soft_gt": "tab:green", "reliability": "tab:purple", "weighted": "tab:red", "gt_early_stop": "tab:orange"}
for i, m in enumerate(methods):
    vals = [delta(m, f) for f in FRACS]
    off = (i - (len(methods) - 1) / 2) * width
    ax.bar(x + off, [v or 0 for v in vals], width, label=f"{m} ({METHOD_LOSS[m]})", color=colors[m])
ax.axhspan(-FLOOR, FLOOR, color="gray", alpha=0.18, label="noise floor ±0.014")
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([f"{f:.2f}" for f in FRACS])
ax.set_xlabel("GT fraction"); ax.set_ylabel("median Δacc(method − naive)")
ax.set_title("Phase 2 — Δ vs naive (loss-matched): only gt_anchored clears the floor; none beat naive xent")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "phase2_delta_bars.png", dpi=140); plt.close(fig)
print("wrote phase2_delta_bars.png")

# ===== per-pair heatmap @0.5 =====
fig, ax = plt.subplots(figsize=(9, 4.6))
M = np.full((len(methods), len(strict)), np.nan)
for mi, m in enumerate(methods):
    for pi, pair in enumerate(strict):
        d = delta(m, 0.5, pair)
        if d is not None: M[mi, pi] = d
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(strict))); ax.set_xticklabels([f"{SHORT[w]}→{SHORT[s]}" for s, w in strict], rotation=45, ha="right")
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
for mi in range(len(methods)):
    for pi in range(len(strict)):
        if not np.isnan(M[mi, pi]): ax.text(pi, mi, f"{M[mi,pi]:+.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Phase 2 — Δacc(method − naive) per pair @ frac 0.50 (blue=helps, red=hurts)")
fig.colorbar(im, ax=ax, fraction=0.04, label="Δacc")
fig.tight_layout(); fig.savefig(OUT / "phase2_per_pair_heatmap.png", dpi=140); plt.close(fig)
print("wrote phase2_per_pair_heatmap.png")
print("PHASE 2 EXTRA PLOTS DONE")
