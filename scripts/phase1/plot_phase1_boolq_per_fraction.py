"""
Phase 1 — per-fraction replica of the Phase 0 `boolq.png` plot.

Same canonical weak-to-strong figure (accuracy vs strong-model GT accuracy; hue=weak model,
xent solid / logconf dashed; min-max range shading across seeds; median-PGR inset table) but
the transfer lines come from the *mixing* runs at a given gt_fraction instead of the pure-weak
baseline. Produces one PNG per fraction plus a combined contact sheet.

The black "ground truth" line and the x-axis positions come from the GT baselines (shared
across fractions), so each panel is directly comparable to the original boolq.png (= frac 0).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "results", "data")
OUT = os.path.join(ROOT, "results", "plots")
os.makedirs(OUT, exist_ok=True)

SEEDS = [0, 1, 2]
MODELS_TO_PLOT = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

# Principled failed-ceiling filter (see plot_phase1.py): seed-1 gpt2-large GT = 0.662
# < gpt2-medium 0.697 — a degenerate "stronger" model; excluded as weak and strong.
EXCLUDE = {(1, "gpt2-large")}


def load_df():
    """GT baselines (weak empty) + all mixing runs, as a flat dataframe."""
    recs = []
    sources = [(os.path.join(DATA, "baseline", f"seed{s}"), "baseline") for s in SEEDS]
    sources += [(os.path.join(DATA, "naive_mixing"), "mixing")]
    for root, cond in sources:
        for rf in glob.glob(os.path.join(root, "**/results_summary.json"), recursive=True):
            cf = os.path.join(os.path.dirname(rf), "config.json")
            if not os.path.exists(cf):
                continue
            c = json.load(open(cf))
            if c.get("ds_name") != "boolq" or c.get("model_size") not in MODELS_TO_PLOT:
                continue
            seed = c.get("seed", 0)
            if seed not in SEEDS:
                continue
            if cond == "mixing" and c.get("gt_seed") != seed:
                continue  # drop stale gt_seed=42 artifacts
            if (seed, c["model_size"]) in EXCLUDE or (seed, c.get("weak_model_size")) in EXCLUDE:
                continue  # principled failed-ceiling filter
            rec = dict(model_size=c["model_size"], loss=c["loss"], seed=seed,
                       weak_model_size=c.get("weak_model_size"),
                       gt_fraction=float(c.get("gt_fraction") or 0.0),
                       cond=cond, accuracy=json.load(open(rf))["accuracy"],
                       _d=os.path.dirname(rf))
            recs.append(rec)
    # uniqueness guard — no stale/residual duplicates may leak into the plots
    seen = {}
    for r in recs:
        k = (r["cond"], r["seed"], r["loss"], r["model_size"], r["weak_model_size"], r["gt_fraction"])
        if k in seen:
            raise SystemExit(f"DUPLICATE logical run {k}:\n  {seen[k]}\n  {r['_d']}")
        seen[k] = r["_d"]
    return pd.DataFrame.from_records(recs).drop(columns=["_d"])


def render(df_frac, frac, ax=None, base_lookup=None, base_order=None):
    """Render the canonical boolq plot for one fraction onto ax (mixing + GT baseline)."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
    cur = df_frac.copy()
    # GT baseline must be xent (loss doesn't apply to GT supervision)
    cur = cur[~((cur["weak_model_size"].isna()) & (cur["loss"] == "logconf"))]

    cur["strong_model_accuracy"] = cur["model_size"].map(base_lookup)
    mask_w = ~cur["weak_model_size"].isna()
    cur.loc[mask_w, "weak_model_accuracy"] = cur.loc[mask_w, "weak_model_size"].map(base_lookup)

    valid = (mask_w & (cur["weak_model_size"] != cur["model_size"])
             & (cur["strong_model_accuracy"] > cur["weak_model_accuracy"]))
    cur.loc[valid, "pgr"] = ((cur.loc[valid, "accuracy"] - cur.loc[valid, "weak_model_accuracy"])
                             / (cur.loc[valid, "strong_model_accuracy"]
                                - cur.loc[valid, "weak_model_accuracy"]))
    cur.loc[cur["weak_model_size"].isna(), "weak_model_size"] = "ground truth"
    plot_df = cur.sort_values("strong_model_accuracy").sort_values("loss", ascending=False)

    # stable colors across fractions: fixed weak-model -> color map
    order = ["ground truth"] + MODELS_TO_PLOT
    pal = sns.color_palette("colorblind", n_colors=len(MODELS_TO_PLOT))
    cmap = {"ground truth": "black", **{m: pal[i] for i, m in enumerate(MODELS_TO_PLOT)}}

    sns.lineplot(data=plot_df, x="strong_model_accuracy", y="accuracy",
                 hue="weak_model_size", style="loss", markers=True,
                 palette={k: cmap[k] for k in plot_df["weak_model_size"].unique()},
                 errorbar=("pi", 100), ax=ax, legend=(standalone))

    pgr_tbl = plot_df[~plot_df["pgr"].isna()].groupby("loss").aggregate({"pgr": "median"})
    if not pgr_tbl.empty:
        pd.plotting.table(ax, pgr_tbl.round(4), loc="lower right",
                          colWidths=[0.12, 0.12], cellLoc="center", rowLoc="center")
    ax.set_xticks([base_lookup[m] for m in base_order])
    ax.set_xticklabels([f"{m} ({base_lookup[m]:.3f})" for m in base_order], rotation=90, fontsize=8)
    ax.set_xlabel("Strong model accuracy (ground truth)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"boolq — gt_fraction={frac} (mean of {plot_df['seed'].nunique()} seeds, shaded=range)")
    if standalone:
        ax.legend(loc="upper left", fontsize=8)
        p = os.path.join(OUT, f"boolq_gf={frac}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
        print("saved", p)


def main():
    df = load_df()
    # x positions / ceiling: mean GT accuracy per model across seeds (weak empty, xent)
    gt = df[(df["cond"] == "baseline") & (df["weak_model_size"].isna()) & (df["loss"] == "xent")]
    base = gt.groupby("model_size")["accuracy"].mean()
    base_lookup = base.to_dict()
    base_order = list(base.sort_values().index)
    print(f"loaded {len(df)} records; GT x-anchors: "
          + ", ".join(f"{m}={base_lookup[m]:.3f}" for m in base_order))

    gt_rows = df[df["weak_model_size"].isna()]  # GT baseline (ground-truth line), shared
    for frac in FRACTIONS:
        mix = df[(df["cond"] == "mixing") & (df["gt_fraction"] == frac)]
        render(pd.concat([gt_rows, mix]), frac, base_lookup=base_lookup, base_order=base_order)

    # combined contact sheet (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for ax, frac in zip(axes.ravel(), FRACTIONS):
        mix = df[(df["cond"] == "mixing") & (df["gt_fraction"] == frac)]
        render(pd.concat([gt_rows, mix]), frac, ax=ax, base_lookup=base_lookup, base_order=base_order)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("Phase 1 — boolq accuracy by weak→strong transfer, per gt_fraction "
                 "(mixing; frac 0 = original boolq.png)", y=1.01, fontsize=15)
    p = os.path.join(OUT, "boolq_by_fraction_grid.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
