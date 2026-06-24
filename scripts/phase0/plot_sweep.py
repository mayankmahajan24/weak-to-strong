#!/usr/bin/env python3
"""Canonical W2SG sweep plot (openai/weak-to-strong style) — ACCURACY and PGR on the y-axis.

Reproduces the paper's figure structure (x = strong model labelled with its GT accuracy; one
coloured line per weak teacher; xent solid/circle, logconf dashed/x; ground-truth line; seed bands;
PGR inset) for the GPT-2 family on BoolQ + SciQ, seeds 0,1,2. Produces both the accuracy-y and
PGR-y variants (the original paper reports both). Professionally styled via scripts/plot_style.py.

Outputs: results/plots/sweep_acc_<task>.png, sweep_pgr_<task>.png
"""
import glob, json, sys, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_style; plot_style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = ROOT / "results/plots"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
SEEDS = [0, 1, 2]
COL = plot_style.MODEL_COLORS

gt, tr = {}, {}    # (model,task,seed)->GT acc ; (weak,strong,task,loss,seed)->transfer acc
for s in SEEDS:
    for cfgp in glob.glob(str(ROOT / f"results/data/baseline/seed{s}/*/config.json")):
        c = json.load(open(cfgp)); ds = c.get("ds_name"); m = c.get("model_size"); loss = c.get("loss", "xent")
        if ds is None or m is None or c.get("gt_fraction", 0) not in (0, 0.0, None): continue
        try: a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
        except Exception: continue
        w = c.get("weak_model_size")
        if not w:
            if loss == "xent": gt[(m, ds, s)] = a
        else:
            tr[(w, m, ds, loss, s)] = a

def stats(vals):  # median, min, max
    return (st.median(vals), min(vals), max(vals)) if vals else (None, None, None)
def gt_stats(m, ds): return stats([gt[(m, ds, s)] for s in SEEDS if (m, ds, s) in gt])
def tr_stats(w, sg, ds, loss): return stats([tr[(w, sg, ds, loss, s)] for s in SEEDS if (w, sg, ds, loss, s) in tr])
def pgr_stats(w, sg, ds, loss):
    vals = []
    for s in SEEDS:
        if (w, ds, s) in gt and (sg, ds, s) in gt and (w, sg, ds, loss, s) in tr:
            wg, scg = gt[(w, ds, s)], gt[(sg, ds, s)]
            if scg - wg > 0: vals.append((tr[(w, sg, ds, loss, s)] - wg) / (scg - wg))
    return stats(vals)

def plot_sweep(ds, mode):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    xpos = {m: i for i, m in enumerate(ORDER)}
    gt_med = {m: gt_stats(m, ds)[0] for m in ORDER}
    # ground-truth line (accuracy mode only)
    if mode == "acc":
        gx = [xpos[m] for m in ORDER]; gm = [gt_stats(m, ds)[0] for m in ORDER]
        glo = [gt_stats(m, ds)[1] for m in ORDER]; ghi = [gt_stats(m, ds)[2] for m in ORDER]
        ax.fill_between(gx, glo, ghi, color=COL["ground truth"], alpha=0.10, zorder=1)
        ax.plot(gx, gm, "-o", color=COL["ground truth"], lw=2.4, label="ground truth", zorder=4)
    else:
        ax.axhline(0, color="0.4", ls=":", lw=1.2)   # imitation
        ax.axhline(1, color="0.4", ls="--", lw=1.0)  # full recovery
        ax.text(0.02, 1.01, "full recovery (PGR=1)", fontsize=8, color="0.4", transform=ax.get_yaxis_transform())
        ax.text(0.02, 0.0, "imitation (PGR=0)", fontsize=8, color="0.4", va="bottom", transform=ax.get_yaxis_transform())
    # weak-model lines
    statf = tr_stats if mode == "acc" else pgr_stats
    lo_rank = 0 if mode == "acc" else 1   # acc includes self (w→w); pgr needs strict
    for w in ORDER:
        for loss, dash, mk in [("xent", "-", "o"), ("logconf", "--", "x")]:
            strongs = [sg for sg in ORDER if RANK[sg] - RANK[w] >= lo_rank]
            pts = [(xpos[sg],) + statf(w, sg, ds, loss) for sg in strongs]
            pts = [(x, m, lo, hi) for (x, m, lo, hi) in pts if m is not None]
            if not pts: continue
            xs = [p[0] for p in pts]; ms = [p[1] for p in pts]
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=COL[w], alpha=0.13, zorder=1)
            ax.plot(xs, ms, ls=dash, marker=mk, color=COL[w], lw=1.8, alpha=0.95, zorder=3, markersize=6)
    # x ticks: model (gt_acc)
    ax.set_xticks([xpos[m] for m in ORDER])
    ax.set_xticklabels([f"{m}\n({gt_med[m]:.3f})" for m in ORDER])
    ax.set_xlabel("strong (student) model — labelled with its ground-truth accuracy")
    if mode == "acc":
        ax.set_ylabel("test accuracy")
        ax.set_title(f"W2SG sweep — {ds.upper()} (GPT-2 family, 3 seeds, band = seed range)")
    else:
        ax.set_ylabel("PGR  (performance-gap recovered)")
        ax.set_title(f"W2SG sweep — PGR — {ds.upper()} (GPT-2 family, 3 seeds)")
        ax.set_ylim(-2.0, 1.35)   # clip extreme outliers (small-denominator PGR instability)
        ax.text(0.985, 0.02, "y clipped to [−2, 1.35];\nextreme small-denominator\nPGR outliers off-scale",
                transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom", color="0.4")
    # legends: weak model colours + loss styles
    weak_handles = [Line2D([0], [0], color=COL[w], lw=2.2, label=w) for w in ORDER]
    if mode == "acc": weak_handles = [Line2D([0], [0], color=COL["ground truth"], lw=2.4, label="ground truth")] + weak_handles
    loss_handles = [Line2D([0], [0], color="0.35", ls="-", marker="o", label="xent"),
                    Line2D([0], [0], color="0.35", ls="--", marker="x", label="logconf")]
    leg1 = ax.legend(handles=weak_handles, title="weak_model_size", loc="upper left",
                     bbox_to_anchor=(0.0, 1.0), fontsize=8.5, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=loss_handles, title="loss", loc="upper left",
              bbox_to_anchor=(0.0, 1.0 - 0.046 * (len(weak_handles) + 1.6)), fontsize=8.5, title_fontsize=9)
    # PGR inset (accuracy mode)
    if mode == "acc":
        strict = [(sg, w) for sg in ORDER for w in ORDER if RANK[w] < RANK[sg]]
        rows = []
        for loss in ["xent", "logconf"]:
            v = [pgr_stats(w, sg, ds, loss)[0] for sg, w in strict if pgr_stats(w, sg, ds, loss)[0] is not None]
            rows.append((loss, st.median(v) if v else float("nan")))
        txt = "median PGR\n" + "\n".join(f"  {l:<8}{p:+.3f}" for l, p in rows)
        ax.text(0.985, 0.03, txt, transform=ax.transAxes, fontsize=8.5, family="monospace",
                ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))
    fig.tight_layout()
    out = OUT / f"sweep_{mode}_{ds}.png"; fig.savefig(out); plt.close(fig)
    print(f"wrote {out.name}")

for ds in ["boolq", "sciq"]:
    for mode in ["acc", "pgr"]:
        plot_sweep(ds, mode)
print("SWEEP PLOTS DONE")
