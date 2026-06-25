#!/usr/bin/env python3
"""Elicitation scaling figure: elicited accuracy vs strong-model size, BoolQ | SciQ.

Reads results/elicitation/runs/*.json. Lines: full-supervised linear probe (the ceiling on linear
elicitation), k=256 probe, CCS@32 (unsupervised); dashed chance reference. Shows the headline:
flat-at-chance on BoolQ, rises with model size on SciQ. -> docs/figs/elicitation_scaling.png
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_style  # noqa: E402
plot_style.setup()

ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
XLAB = ["gpt2\n124M", "medium\n355M", "large\n774M", "xl\n1.5B"]
# Anthropic accents
C_FULL, C_PROBE, C_CCS, C_CH = "#d97757", "#6a9bcc", "#788c5d", "#b0aea5"

runs = {}
for fn in glob.glob(str(ROOT / "results/elicitation/runs/*.json")):
    r = json.load(open(fn))
    runs[(r["ds"], r["model"], r["seed"])] = r
seeds = sorted({k[2] for k in runs})


def series(ds, getter):
    med, lo, hi = [], [], []
    for m in ORDER:
        vals = [getter(runs[(ds, m, s)]) for s in seeds if (ds, m, s) in runs]
        med.append(st.median(vals)); lo.append(min(vals)); hi.append(max(vals))
    return med, lo, hi


def plot_panel(ax, ds, title, show_legend):
    x = list(range(len(ORDER)))
    specs = [
        ("full-supervised probe", lambda r: r["m1_full_supervised"], C_FULL, "o", "-"),
        ("k=256 probe", lambda r: r["m1_probe"]["256"], C_PROBE, "s", "-"),
        ("CCS (unsupervised)", lambda r: r["m2_ccs"]["32"], C_CCS, "^", "--"),
    ]
    for label, get, color, mk, ls in specs:
        med, lo, hi = series(ds, get)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, med, color=color, marker=mk, linestyle=ls, label=label, zorder=3)
    ch = st.median([runs[(ds, m, s)]["chance"] for m in ORDER for s in seeds if (ds, m, s) in runs])
    ax.axhline(ch, color=C_CH, linestyle=":", linewidth=1.6, zorder=1)
    ax.text(0.02, ch, "chance", color="0.45", fontsize=8.5, va="bottom", ha="left",
            transform=ax.get_yaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(XLAB)
    ax.set_title(title)
    ax.set_xlabel("strong model")
    ax.set_ylim(0.44, 0.70)
    if show_legend:
        ax.legend(loc="upper left")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
plot_panel(a1, "boolq", "BoolQ — flat", True)
plot_panel(a2, "sciq", "SciQ — rises with model size", False)
a1.set_ylabel("elicited accuracy")
fig.tight_layout()
out = ROOT / "docs/figs/elicitation_scaling.png"
fig.savefig(out)
print("wrote", out)
