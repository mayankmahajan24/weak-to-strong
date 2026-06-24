#!/usr/bin/env python3
"""Required-deliverable figure: GPT-2 W2SG sweep WITH a GT budget mixed in (default 25%).

Same canonical paper format and professional style as scripts/phase0/plot_sweep.py (x = strong
model labelled with its GT accuracy; one line per weak teacher; xent solid/circle, logconf
dashed/x; ground-truth line; seed bands; median-PGR inset) — but the transfer lines come from the
naive_mixing runs at gt_fraction=FRAC instead of the pure-weak baseline. Uses the *canonical*
strict-pair PGR convention (weak<strong, positive denominator, EXCLUDE seed-1 gpt2-large), so the
reported median PGR matches RESULTS_phase1 and the fraction curve (xent ≈ +0.30 at 0.25).

Reads results/phase1/phase1_results.csv. Output: results/plots/sweep_acc_<ds>_gf<fdir>.png
Usage: python plot_sweep_gf.py [--frac=0.25] [--ds=boolq]
"""
import csv, sys, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_style; plot_style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

opts = {a[2:].split("=")[0]: a[2:].split("=")[1] for a in sys.argv[1:] if a.startswith("--")}
FRAC = float(opts.get("frac", 0.25)); DS = opts.get("ds", "boolq")
FDIR = f"{int(round(FRAC*100)):03d}"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]; RANK = {m: i for i, m in enumerate(ORDER)}
SEEDS = [0, 1, 2]; EXCLUDE = {(1, "gpt2-large")}; COL = plot_style.MODEL_COLORS
def exc(s, m): return (s, m) in EXCLUDE

gt, tr = {}, {}   # (model,seed)->GT acc(xent) ; (weak,strong,loss,seed)->mixing acc @FRAC
for r in csv.DictReader((ROOT / "results/phase1/phase1_results.csv").open()):
    if r["ds_name"] != DS:
        continue
    s = int(r["seed"]); a = float(r["accuracy"]); f = float(r["gt_fraction_requested"])
    if r["condition"] == "baseline" and r["weak_model"] == "" and r["loss"] == "xent":
        gt[(r["strong_model"], s)] = a
    elif r["condition"] == "mixing" and abs(f - FRAC) < 1e-9:
        tr[(r["weak_model"], r["strong_model"], r["loss"], s)] = a

def stats(v): return (st.median(v), min(v), max(v)) if v else (None, None, None)
def gt_stats(m): return stats([gt[(m, s)] for s in SEEDS if (m, s) in gt])
def tr_stats(w, sg, loss): return stats([tr[(w, sg, loss, s)] for s in SEEDS
                                         if (w, sg, loss, s) in tr and not exc(s, w) and not exc(s, sg)])
def pgr_med(loss):
    v = []
    for sg in ORDER:
        for w in ORDER:
            if RANK[w] >= RANK[sg]:
                continue
            for s in SEEDS:
                if exc(s, w) or exc(s, sg):
                    continue
                if (w, s) in gt and (sg, s) in gt and (w, sg, loss, s) in tr and gt[(sg, s)] - gt[(w, s)] > 0:
                    v.append((tr[(w, sg, loss, s)] - gt[(w, s)]) / (gt[(sg, s)] - gt[(w, s)]))
    return st.median(v) if v else float("nan")

xpos = {m: i for i, m in enumerate(ORDER)}
gt_med = {m: gt_stats(m)[0] for m in ORDER}
fig, ax = plt.subplots(figsize=(8.5, 6.2))
gx = [xpos[m] for m in ORDER]
ax.fill_between(gx, [gt_stats(m)[1] for m in ORDER], [gt_stats(m)[2] for m in ORDER],
                color=COL["ground truth"], alpha=0.10, zorder=1)
ax.plot(gx, [gt_med[m] for m in ORDER], "-o", color=COL["ground truth"], lw=2.4, label="ground truth", zorder=4)
for w in ORDER:
    for loss, dash, mk in [("xent", "-", "o"), ("logconf", "--", "x")]:
        pts = [(xpos[sg],) + tr_stats(w, sg, loss) for sg in ORDER if RANK[sg] >= RANK[w]]
        pts = [(x, m, lo, hi) for (x, m, lo, hi) in pts if m is not None]
        if not pts:
            continue
        ax.fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts], color=COL[w], alpha=0.13, zorder=1)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=dash, marker=mk, color=COL[w], lw=1.8, alpha=0.95, zorder=3, markersize=6)
ax.set_xticks([xpos[m] for m in ORDER]); ax.set_xticklabels([f"{m}\n({gt_med[m]:.3f})" for m in ORDER])
ax.set_xlabel("strong (student) model — labelled with its ground-truth accuracy")
ax.set_ylabel("test accuracy")
ax.set_title(f"W2SG sweep — {DS.upper()} + {int(round(FRAC*100))}% ground truth (GPT-2 family, 3 seeds, band = seed range)")
weak_handles = [Line2D([0], [0], color=COL["ground truth"], lw=2.4, label="ground truth")] + \
               [Line2D([0], [0], color=COL[w], lw=2.2, label=w) for w in ORDER]
loss_handles = [Line2D([0], [0], color="0.35", ls="-", marker="o", label="xent"),
                Line2D([0], [0], color="0.35", ls="--", marker="x", label="logconf")]
leg1 = ax.legend(handles=weak_handles, title="weak_model_size", loc="upper left", fontsize=8.5, title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=loss_handles, title="loss", loc="upper left", bbox_to_anchor=(0.0, 1.0 - 0.046 * 6.6), fontsize=8.5, title_fontsize=9)
txt = "median PGR\n" + "\n".join(f"  {l:<8}{pgr_med(l):+.3f}" for l in ["xent", "logconf"])
ax.text(0.985, 0.03, txt, transform=ax.transAxes, fontsize=8.5, family="monospace", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))
fig.tight_layout()
out = ROOT / f"results/plots/sweep_acc_{DS}_gf{FDIR}.png"; fig.savefig(out); plt.close(fig)
print(f"wrote {out.name}  (median PGR xent={pgr_med('xent'):+.3f}, logconf={pgr_med('logconf'):+.3f})")
