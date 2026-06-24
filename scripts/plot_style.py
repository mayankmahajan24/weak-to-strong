"""Shared publication-quality plot style (via the data-viz-plots skill conventions).

Import in any plot script:  import sys; sys.path.insert(0, <scripts/>); import plot_style; plot_style.setup()
Provides a consistent seaborn-whitegrid theme, despined axes, 300-dpi export, and a fixed
colorblind-friendly (Okabe-Ito) colour map for the GPT-2 family so colours match across figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def setup():
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.9,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


# Okabe-Ito colorblind-safe palette, fixed per model (consistent across all figures)
MODEL_COLORS = {
    "gpt2": "#E69F00",          # orange
    "gpt2-medium": "#56B4E9",   # sky blue
    "gpt2-large": "#009E73",    # green
    "gpt2-xl": "#D55E00",       # vermillion
    "ground truth": "#000000",
}
# Method palette (Phase 2), also Okabe-Ito-derived
METHOD_COLORS = {
    "gt_anchored": "#0072B2", "soft_gt": "#009E73", "reliability": "#CC79A7",
    "weighted": "#D55E00", "gt_early_stop": "#E69F00",
}
# Task palette
TASK_COLORS = {"boolq": "#0072B2", "sciq": "#009E73", "BoolQ": "#0072B2", "SciQ": "#009E73"}


def despine_offset(ax):
    sns.despine(ax=ax, offset=4, trim=False)
