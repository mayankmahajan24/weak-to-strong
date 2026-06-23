"""
Phase 1 plots — supervision-scaling fraction curve + scale interaction + accuracy grid.

Produces ENSEMBLE figures (3-seed aggregate, range-shaded) and BY-SEED tiled figures.

PGR convention (standard weak-to-strong):
    floor(seed, m)   = accuracy of m trained on ground truth (xent), that seed
    ceiling(seed, m) = floor for the strong model
    PGR = (transfer_acc - floor(weak)) / (ceiling(strong) - floor(weak)),  valid if denom > 0
- mixing  point (seed, weak, strong, frac): transfer_acc = the naive_mixing run
- gt_only point (seed, weak, strong, frac): transfer_acc = the gt_only run for `strong` at
  `frac` (weak labels discarded, so identical across weak source) — placed on the SAME
  (weak,strong) pairs as mixing so the two curves share support and are directly comparable.
frac=0 anchor for mixing = the baseline pure-weak transfer (gt_fraction=0).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
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
MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
MODEL_IDX = {m: i for i, m in enumerate(MODELS)}
FRACTIONS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
LOSSES = ["xent", "logconf"]

LOSS_COLOR = {"xent": "#1f77b4", "logconf": "#d62728"}
COND_STYLE = {"mixing": "-", "gt_only": ":"}

# Principled failed-ceiling filter (pre-stated, outcome-independent): drop any GT ceiling
# that fails to exceed the next-smaller model's GT accuracy — a model that scores below a
# smaller one has not trained as a "stronger" model, and a PGR that divides by a degenerate
# ceiling is meaningless. seed-1 gpt2-large GT = 0.662 < gpt2-medium 0.697 fails this test
# (reproduced deterministically by the S5 regen, so it is a real bad-optimization outcome,
# not corruption). Excluded as weak AND strong for seed 1.
EXCLUDE = {(1, "gpt2-large")}
def is_excl(seed, model):
    return (seed, model) in EXCLUDE


# ---------------------------------------------------------------- load
def load_records():
    recs = []
    roots = (
        [(os.path.join(DATA, "baseline", f"seed{s}"), "baseline") for s in SEEDS]
        + [(os.path.join(DATA, "naive_mixing"), "mixing"),
           (os.path.join(DATA, "gt_only"), "gt_only")]
    )
    for root, cond in roots:
        for sp in glob.glob(os.path.join(root, "**/results_summary.json"), recursive=True):
            d = os.path.dirname(sp)
            cp = os.path.join(d, "config.json")
            if not os.path.exists(cp):
                continue
            c = json.load(open(cp))
            s = json.load(open(sp))
            if c.get("ds_name") != "boolq":
                continue
            seed = c.get("seed")
            if seed not in SEEDS:
                continue
            if cond != "baseline" and c.get("gt_seed") != seed:
                continue  # drop stale gt_seed=42 artifacts
            recs.append(dict(
                cond=cond, seed=seed, loss=c.get("loss"),
                strong=c.get("model_size"), weak=c.get("weak_model_size"),
                frac=float(c.get("gt_fraction") or 0.0),
                acc=float(s.get("accuracy")),
                _d=d,
            ))
    # uniqueness guard — no stale/residual duplicates may leak into the plots
    seen = {}
    for r in recs:
        k = (r["cond"], r["seed"], r["loss"], r["strong"], r["weak"], r["frac"])
        if k in seen:
            raise SystemExit(f"DUPLICATE logical run {k}:\n  {seen[k]}\n  {r['_d']}")
        seen[k] = r["_d"]
    return recs


def build_floor(recs):
    """floor[(seed, model)] = GT(xent) accuracy."""
    floor = {}
    for r in recs:
        if r["cond"] == "baseline" and r["weak"] is None and r["loss"] == "xent":
            if is_excl(r["seed"], r["strong"]):
                continue  # degenerate ceiling — do not anchor PGR on it
            floor[(r["seed"], r["strong"])] = r["acc"]
    missing = [(s, m) for s in SEEDS for m in MODELS
               if (s, m) not in floor and not is_excl(s, m)]
    if missing:
        print("WARNING missing GT floor for:", missing)
    return floor


# ---------------------------------------------------------------- PGR points
def pgr_points(recs, floor):
    """Return list of dicts: seed, loss, cond, weak, strong, frac, pgr, acc — over valid pairs."""
    # index accuracies
    mix = {}   # (seed,loss,weak,strong,frac) -> acc
    gto = {}   # (seed,loss,strong,frac) -> acc (weak source discarded)
    for r in recs:
        if r["cond"] == "mixing":
            mix[(r["seed"], r["loss"], r["weak"], r["strong"], r["frac"])] = r["acc"]
        elif r["cond"] == "baseline" and r["weak"] is not None:
            mix[(r["seed"], r["loss"], r["weak"], r["strong"], 0.0)] = r["acc"]  # frac0 anchor
        elif r["cond"] == "gt_only":
            gto[(r["seed"], r["loss"], r["strong"], r["frac"])] = r["acc"]

    pts = []
    pairs = [(w, s) for w in MODELS for s in MODELS if MODEL_IDX[w] < MODEL_IDX[s]]
    for seed in SEEDS:
        for loss in LOSSES:
            for w, s in pairs:
                fl, ce = floor.get((seed, w)), floor.get((seed, s))
                if fl is None or ce is None or ce - fl <= 0:
                    continue
                denom = ce - fl
                for f in FRACTIONS:
                    a = mix.get((seed, loss, w, s, f))
                    if a is not None:
                        pts.append(dict(seed=seed, loss=loss, cond="mixing",
                                        weak=w, strong=s, frac=f,
                                        pgr=(a - fl) / denom, acc=a))
                    if f > 0:
                        ag = gto.get((seed, loss, s, f))
                        if ag is not None:
                            pts.append(dict(seed=seed, loss=loss, cond="gt_only",
                                            weak=w, strong=s, frac=f,
                                            pgr=(ag - fl) / denom, acc=ag))
    return pts


def agg(pts, keys, val="pgr"):
    """median + min/max of `val` grouped by tuple(keys)."""
    g = defaultdict(list)
    for p in pts:
        g[tuple(p[k] for k in keys)].append(p[val])
    return {k: (np.median(v), np.min(v), np.max(v), len(v)) for k, v in g.items()}


# ---------------------------------------------------------------- noise floor
def noise_floor(recs):
    """Per-pair 3-seed accuracy range for xent baseline transfers (weak!=strong)."""
    byp = defaultdict(dict)
    for r in recs:
        if r["cond"] == "baseline" and r["loss"] == "xent" and r["weak"] is not None \
                and r["weak"] != r["strong"]:
            if is_excl(r["seed"], r["strong"]) or is_excl(r["seed"], r["weak"]):
                continue
            byp[(r["weak"], r["strong"])][r["seed"]] = r["acc"]
    ranges = {p: (max(d.values()) - min(d.values())) for p, d in byp.items() if len(d) >= 2}
    arr = np.array(list(ranges.values())) if ranges else np.array([0.0])
    return dict(mean=arr.mean(), median=np.median(arr), max=arr.max(),
                argmax=max(ranges, key=ranges.get) if ranges else None, per_pair=ranges)


# ---------------------------------------------------------------- plots
def curve_stats(pts, loss, cond, seed=None):
    """Two-stage aggregate: per-seed median over pairs, then summarise across seeds.
    Returns frac -> (point_estimate, lo, hi). Avoids per-pair PGR-denominator blowups."""
    byf = {}
    for f in FRACTIONS:
        sub = [p["pgr"] for p in pts if p["loss"] == loss and p["cond"] == cond
               and p["frac"] == f and (seed is None or p["seed"] == seed)]
        if not sub:
            continue
        if seed is not None:
            byf[f] = (np.median(sub), None, None)
        else:
            per_seed = [np.median([p["pgr"] for p in pts if p["loss"] == loss
                                   and p["cond"] == cond and p["frac"] == f and p["seed"] == sd])
                        for sd in SEEDS
                        if any(p["loss"] == loss and p["cond"] == cond and p["frac"] == f
                               and p["seed"] == sd for p in pts)]
            byf[f] = (float(np.median(per_seed)), float(np.min(per_seed)), float(np.max(per_seed)))
    return byf


def draw_curve(ax, pts, seed=None, noise_pgr=None, title=""):
    for loss in LOSSES:
        for cond in ["mixing", "gt_only"]:
            st = curve_stats(pts, loss, cond, seed=seed)
            if not st:
                continue
            xs = sorted(st)
            md = [st[f][0] for f in xs]
            ax.plot(xs, md, COND_STYLE[cond], color=LOSS_COLOR[loss], marker="o", ms=4,
                    lw=2 if cond == "mixing" else 1.6,
                    label=f"{loss} {cond}")
            if seed is None:  # seed-range band (range of per-seed medians) on ensemble only
                lo = [st[f][1] for f in xs]; hi = [st[f][2] for f in xs]
                ax.fill_between(xs, lo, hi, color=LOSS_COLOR[loss], alpha=0.12)
    if noise_pgr:
        ax.axhspan(-noise_pgr, noise_pgr, color="gray", alpha=0.15, zorder=0)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels([str(f) for f in FRACTIONS], fontsize=8)
    ax.set_xlabel("gt_fraction")
    ax.set_ylabel("PGR (median over pairs)")
    ax.set_title(title)


def plot_fraction_curve(pts, noise_pgr):
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_curve(ax, pts, seed=None, noise_pgr=noise_pgr,
               title="Phase 1 — Supervision scaling: PGR vs gt_fraction (3-seed, BoolQ)\n"
                     "solid=mixing, dotted=GT-only, blue=xent, red=logconf; shaded=seed range, "
                     "gray=noise floor")
    ax.legend(loc="upper left", fontsize=9)
    p = os.path.join(OUT, "phase1_fraction_curve.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", p)


def plot_fraction_curve_by_seed(pts, noise_pgr):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        draw_curve(ax, pts, seed=seed, noise_pgr=noise_pgr, title=f"seed {seed}")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Phase 1 — Fraction curve, tiled by seed (BoolQ)", y=1.02, fontsize=13)
    p = os.path.join(OUT, "phase1_fraction_curve_by_seed.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", p)


def slopes(pts, seed, loss, cond="mixing"):
    """For each strong model, slope of PGR vs frac (median over weak partners per frac)."""
    out = {}
    for s in MODELS:
        sub = [p for p in pts if p["strong"] == s and p["loss"] == loss
               and p["cond"] == cond and (seed is None or p["seed"] == seed)]
        if not sub:
            continue
        byf = defaultdict(list)
        for p in sub:
            byf[p["frac"]].append(p["pgr"])
        xs = sorted(byf)
        if len(xs) < 2:
            continue
        ys = [np.median(byf[f]) for f in xs]
        out[s] = np.polyfit(xs, ys, 1)[0]
    return out


def draw_scale(ax, pts, seed=None, title=""):
    for loss in LOSSES:
        if seed is None:  # ensemble: mean ± range across seeds
            per_seed = {sd: slopes(pts, sd, loss) for sd in SEEDS}
            xs, mean, lo, hi = [], [], [], []
            for s in MODELS:
                vals = [per_seed[sd][s] for sd in SEEDS if s in per_seed[sd]]
                if not vals:
                    continue
                xs.append(MODEL_IDX[s]); mean.append(np.mean(vals))
                lo.append(np.min(vals)); hi.append(np.max(vals))
            if xs:
                ax.plot(xs, mean, "-o", color=LOSS_COLOR[loss], label=loss, lw=2)
                ax.fill_between(xs, lo, hi, color=LOSS_COLOR[loss], alpha=0.15)
        else:
            sl = slopes(pts, seed, loss)
            xs = [MODEL_IDX[s] for s in MODELS if s in sl]
            ys = [sl[s] for s in MODELS if s in sl]
            if xs:
                ax.plot(xs, ys, "-o", color=LOSS_COLOR[loss], label=loss, lw=2)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=30, fontsize=8)
    ax.set_xlabel("strong model (size →)")
    ax.set_ylabel("ΔPGR / Δgt_fraction (slope)")
    ax.set_title(title)


def plot_scale(pts):
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_scale(ax, pts, seed=None,
               title="Phase 1 — Scale interaction: does GT's marginal value grow with model size?\n"
                     "(mixing PGR slope vs fraction; 3-seed mean, shaded=range)")
    ax.legend(fontsize=10)
    p = os.path.join(OUT, "phase1_scale_interaction.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", p)


def plot_scale_by_seed(pts):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        draw_scale(ax, pts, seed=seed, title=f"seed {seed}")
    axes[0].legend(fontsize=9)
    fig.suptitle("Phase 1 — Scale interaction, tiled by seed (BoolQ)", y=1.02, fontsize=13)
    p = os.path.join(OUT, "phase1_scale_interaction_by_seed.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", p)


def draw_acc_panel(ax, recs, floor, seed, frac, loss="xent"):
    """Accuracy vs strong-model-GT-acc, hue=weak model, for one fraction. Ensemble if seed=None."""
    pair_acc = defaultdict(list)  # (weak,strong) -> [acc per seed]
    seeds = SEEDS if seed is None else [seed]
    # x position = mean GT acc of strong model (excluding degenerate ceilings)
    xof = {m: np.mean([floor[(sd, m)] for sd in seeds if (sd, m) in floor])
           for m in MODELS if any((sd, m) in floor for sd in seeds)}
    for r in recs:
        if r["loss"] != loss or r["strong"] is None:
            continue
        if r["seed"] not in seeds:
            continue
        if is_excl(r["seed"], r["strong"]) or is_excl(r["seed"], r["weak"]):
            continue  # principled failed-ceiling filter
        is_mix = (r["cond"] == "mixing" and r["frac"] == frac) or \
                 (r["cond"] == "baseline" and r["weak"] is not None and frac == 0.0)
        if not is_mix:
            continue
        pair_acc[(r["weak"], r["strong"])].append(r["acc"])
    weaks = sorted({w for (w, s) in pair_acc}, key=lambda m: MODEL_IDX.get(m, 9))
    pal = sns.color_palette("colorblind", n_colors=max(len(weaks), 1))
    cmap = {w: pal[i] for i, w in enumerate(weaks)}
    for w in weaks:
        pts_ = []
        for s in MODELS:
            if (w, s) in pair_acc and s in xof:
                v = pair_acc[(w, s)]
                pts_.append((xof[s], np.mean(v), np.min(v), np.max(v)))
        pts_.sort()  # connect markers left-to-right in x (GT acc), not model-size order
        if pts_:
            xs, ys, lo, hi = zip(*pts_)
            ax.plot(xs, ys, "-o", color=cmap[w], ms=4, label=f"weak={w}")
            if seed is None and len(seeds) > 1:
                ax.fill_between(xs, lo, hi, color=cmap[w], alpha=0.15)
    # GT ceiling reference (diagonal)
    gx = sorted(xof[m] for m in MODELS if m in xof)
    ax.plot(gx, gx, "k--", lw=0.8, alpha=0.5, label="GT ceiling")
    ax.set_title(f"gt_fraction={frac}", fontsize=10)
    ax.set_xlabel("strong GT acc", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_accuracy_grid(recs, floor, loss="xent"):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    for i, frac in enumerate(FRACTIONS):
        draw_acc_panel(axes[i], recs, floor, seed=None, frac=frac, loss=loss)
    axes[0].legend(fontsize=7, loc="upper left")
    for j in range(len(FRACTIONS), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Phase 1 — Accuracy grid by fraction ({loss}, 3-seed mean, shaded=range, BoolQ)",
                 y=1.0, fontsize=14)
    p = os.path.join(OUT, "phase1_accuracy_grid.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig); print("saved", p)


def plot_accuracy_grid_by_seed(recs, floor, loss="xent"):
    fig, axes = plt.subplots(len(SEEDS), len(FRACTIONS),
                             figsize=(4 * len(FRACTIONS), 3.4 * len(SEEDS)))
    for i, seed in enumerate(SEEDS):
        for j, frac in enumerate(FRACTIONS):
            draw_acc_panel(axes[i][j], recs, floor, seed=seed, frac=frac, loss=loss)
            if j == 0:
                axes[i][j].set_ylabel(f"seed {seed}\naccuracy", fontsize=9)
    axes[0][0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Phase 1 — Accuracy grid, seed (rows) × fraction (cols) [{loss}, BoolQ]",
                 y=1.0, fontsize=14)
    p = os.path.join(OUT, "phase1_accuracy_grid_by_seed.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)


# ---------------------------------------------------------------- main
def main():
    recs = load_records()
    floor = build_floor(recs)
    pts = pgr_points(recs, floor)
    print(f"loaded {len(recs)} records; {len(pts)} PGR points")

    nf = noise_floor(recs)
    # express noise floor as approx PGR band: median per-pair acc range / median denom
    denoms = [floor[(s, ss)] - floor[(s, w)]
              for s in SEEDS for w in MODELS for ss in MODELS
              if MODEL_IDX.get(w, 9) < MODEL_IDX.get(ss, 9)
              and (s, w) in floor and (s, ss) in floor and floor[(s, ss)] > floor[(s, w)]]
    noise_pgr = nf["median"] / np.median(denoms) if denoms else 0.05
    print(f"noise floor (acc): mean={nf['mean']:.4f} median={nf['median']:.4f} "
          f"max={nf['max']:.4f} ({nf['argmax']}) -> ~PGR band ±{noise_pgr:.3f}")

    plot_fraction_curve(pts, noise_pgr)
    plot_fraction_curve_by_seed(pts, noise_pgr)
    plot_scale(pts)
    plot_scale_by_seed(pts)
    plot_accuracy_grid(recs, floor, loss="xent")
    plot_accuracy_grid_by_seed(recs, floor, loss="xent")


if __name__ == "__main__":
    main()
