#!/usr/bin/env python3
"""Phase 2 overlay plot — each method's median-accuracy curve vs the naive xent/logconf curves.

Visualizes the left-shift / ceiling-raise readouts: x = GT fraction {0.10,0.25,0.50}, y = median
test accuracy over the 6 strict pairs × 3 seeds (EXCLUDE seed1 gpt2-large). Naive xent and naive
logconf mixing are drawn as black reference curves; gt_anchored is a logconf method (compare to the
logconf reference), the rest are xent. Saves results/plots/phase2_overlay.png.
"""
import csv, glob, json, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
EXCLUDE = {(1, "gpt2-large")}
FRACS = [0.1, 0.25, 0.5]
strict = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]
METHOD_LOSS = {"weighted": "xent", "soft_gt": "xent", "reliability": "xent",
               "gt_early_stop": "xent", "gt_anchored": "logconf"}

def exc(seed, m): return (seed, m) in EXCLUDE

# naive baseline from phase1
naive = {}
for r in csv.DictReader((ROOT / "results/phase1/phase1_results.csv").open()):
    if r["condition"] == "mixing" and r["ds_name"] == "boolq":
        naive[(r["loss"], int(r["seed"]), r["weak_model"], r["strong_model"],
               float(r["gt_fraction_requested"]))] = float(r["accuracy"])

# phase2 method runs
meth = {}
for cfgp in glob.glob(str(ROOT / "results/data/phase2_*/**/config.json"), recursive=True):
    if "phase2_sciq" in cfgp: continue
    c = json.load(open(cfgp)); cm = c.get("combination_method")
    if not cm or cm == "naive": continue
    try: a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
    except Exception: continue
    meth[(cm, c["seed"], c.get("weak_model_size"), c["model_size"], c["gt_fraction"])] = a

def med_method(m, f):
    v = [meth[(m, s, w, st_, f)] for st_, w in strict for s in [0,1,2]
         if (m, s, w, st_, f) in meth and not exc(s, w) and not exc(s, st_)]
    return st.median(v) if v else None

def med_naive(loss, f):
    v = [naive[(loss, s, w, st_, f)] for st_, w in strict for s in [0,1,2]
         if (loss, s, w, st_, f) in naive and not exc(s, w) and not exc(s, st_)]
    return st.median(v) if v else None

methods = sorted({k[0] for k in meth})
fig, ax = plt.subplots(figsize=(8, 5.5))
# naive references
for loss, style in [("xent", dict(color="black", lw=2.4, ls="-")),
                    ("logconf", dict(color="black", lw=2.0, ls=":"))]:
    ys = [med_naive(loss, f) for f in FRACS]
    if any(y is not None for y in ys):
        ax.plot(FRACS, ys, label=f"naive {loss}", marker="s", **style)
# methods
colors = {"weighted": "tab:red", "soft_gt": "tab:green", "reliability": "tab:purple",
          "gt_anchored": "tab:blue", "gt_early_stop": "tab:orange"}
for m in methods:
    ys = [med_method(m, f) for f in FRACS]
    n = sum(1 for k in meth if k[0] == m)
    ls = "--" if METHOD_LOSS.get(m) == "logconf" else "-"
    lbl = f"{m} ({METHOD_LOSS.get(m,'xent')}, n={n})"
    ax.plot(FRACS, ys, label=lbl, marker="o", color=colors.get(m), ls=ls, alpha=0.9)

ax.set_xlabel("GT fraction"); ax.set_ylabel("median test accuracy (6 strict pairs × 3 seeds)")
ax.set_title("Phase 2 — combination methods vs naive mixing (BoolQ)")
ax.set_xticks(FRACS); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
out = ROOT / "results/plots/phase2_overlay.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=140)
print(f"saved {out}  (methods: {methods})")
