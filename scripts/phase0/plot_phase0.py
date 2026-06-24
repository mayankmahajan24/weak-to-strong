#!/usr/bin/env python3
"""Phase 0 plots — baseline reproduction & the core W2SG phenomenon (seeds 0,1,2 ONLY).

Seeds 3,4 are a later cross-check reserve and are deliberately excluded here. Reads
results/data/baseline/seed{0,1,2}/ (GT ceilings + frac=0 transfers, boolq+sciq, xent+logconf).
Generates:
  phase0_w2sg_<task>.png       canonical W2SG: student vs teacher acc, per pair, w/ imitation
                               diagonal + strong-ceiling band (xent + logconf panels)
  phase0_ceilings.png          GT ceiling acc vs model size, both tasks, seed spread
  phase0_pgr_by_pair_<task>.png  PGR per strict pair (xent vs logconf)
  phase0_transfer_heatmap.png  weak×strong transfer-acc heatmap (xent), both tasks
  phase0_xent_vs_logconf.png   median PGR xent vs logconf, both tasks
"""
import glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/plots"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
SHORT = {"gpt2": "g2", "gpt2-medium": "M", "gpt2-large": "L", "gpt2-xl": "XL"}
PARAMS = {"gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1558}  # M params
RANK = {m: i for i, m in enumerate(ORDER)}
SEEDS = [0, 1, 2]
TASKS = ["boolq", "sciq"]
EXCLUDE = {(1, "gpt2-large")}
strict = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]

# ---- load baseline ----
gt = {}     # (model, task, seed) -> GT acc (xent)
tr = {}     # (weak, strong, task, loss, seed) -> transfer acc
for s in SEEDS:
    for cfgp in glob.glob(str(ROOT / f"results/data/baseline/seed{s}/*/config.json")):
        c = json.load(open(cfgp)); ds = c.get("ds_name"); m = c.get("model_size"); loss = c.get("loss", "xent")
        if ds is None or m is None or c.get("gt_fraction", 0) not in (0, 0.0, None):
            continue
        try: a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
        except Exception: continue
        w = c.get("weak_model_size")
        if not w:
            if loss == "xent": gt[(m, ds, s)] = a
        else:
            tr[(w, m, ds, loss, s)] = a

def med_gt(m, ds):
    v = [gt[(m, ds, s)] for s in SEEDS if (m, ds, s) in gt]; return st.median(v) if v else None
def med_tr(w, sg, ds, loss):
    v = [tr[(w, sg, ds, loss, s)] for s in SEEDS if (w, sg, ds, loss, s) in tr]; return st.median(v) if v else None

# ===== 1. Canonical W2SG plot (student vs teacher), per task, xent+logconf panels =====
for ds in TASKS:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    for ax, loss in zip(axes, ["xent", "logconf"]):
        # reference: imitation diagonal (y=x) and per-strong ceilings
        lo, hi = 0.45, 0.80
        ax.plot([lo, hi], [lo, hi], "k:", lw=1, label="imitation (student=teacher)")
        cmap = {"gpt2-medium": "tab:blue", "gpt2-large": "tab:orange", "gpt2-xl": "tab:red"}
        for strong in ORDER[1:]:
            sc = med_gt(strong, ds)
            if sc: ax.axhline(sc, color=cmap[strong], ls="--", lw=0.8, alpha=0.5)
            for weak in [w for w in ORDER if RANK[w] < RANK[strong]]:
                tw, t = med_gt(weak, ds), med_tr(weak, strong, ds, loss)
                if tw and t:
                    ax.scatter(tw, t, color=cmap[strong], s=70, zorder=3,
                               edgecolor="k", lw=0.5)
                    ax.annotate(f"{SHORT[weak]}→{SHORT[strong]}", (tw, t), fontsize=6.5,
                                xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("weak teacher GT accuracy"); ax.set_title(f"{loss}")
        ax.grid(alpha=0.3); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    axes[0].set_ylabel("strong student (transfer) accuracy")
    # legend proxies
    from matplotlib.lines import Line2D
    proxies = [Line2D([0],[0],marker="o",color="w",markerfacecolor=c,markeredgecolor="k",label=f"→{SHORT[m]}")
               for m,c in [("gpt2-medium","tab:blue"),("gpt2-large","tab:orange"),("gpt2-xl","tab:red")]]
    proxies += [Line2D([0],[0],ls=":",color="k",label="imitation y=x"),
                Line2D([0],[0],ls="--",color="gray",label="strong GT ceiling")]
    axes[1].legend(handles=proxies, fontsize=7, loc="lower right")
    fig.suptitle(f"Phase 0 — W2SG on {ds.upper()} (3 seeds): points above the diagonal generalize; the ceiling is the target")
    fig.tight_layout(); fig.savefig(OUT / f"phase0_w2sg_{ds}.png", dpi=140); plt.close(fig)
    print(f"wrote phase0_w2sg_{ds}.png")

# ===== 2. GT ceilings vs model size =====
fig, ax = plt.subplots(figsize=(7.5, 5))
for ds, col in zip(TASKS, ["tab:blue", "tab:green"]):
    xs = [PARAMS[m] for m in ORDER]
    meds = [med_gt(m, ds) for m in ORDER]
    # seed spread as error bars (min/max)
    errs = []
    for m in ORDER:
        v = [gt[(m, ds, s)] for s in SEEDS if (m, ds, s) in gt]
        errs.append((med_gt(m, ds) - min(v), max(v) - med_gt(m, ds)) if v else (0, 0))
    yerr = np.array(errs).T
    ax.errorbar(xs, meds, yerr=yerr, marker="o", capsize=4, color=col, label=ds, lw=1.8)
    for m in ORDER:
        ax.annotate(SHORT[m], (PARAMS[m], med_gt(m, ds)), fontsize=7, xytext=(4, -8), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("model size (M params, log)"); ax.set_ylabel("GT ceiling accuracy")
ax.set_title("Phase 0 — GT ceilings vs scale (3 seeds; bars = seed min/max)\nnote mid-family instability (large/medium spread)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "phase0_ceilings.png", dpi=140); plt.close(fig)
print("wrote phase0_ceilings.png")

# ===== 3. PGR by pair, per task (xent vs logconf) =====
def pgr(weak, strong, ds, loss):
    vals = []
    for s in SEEDS:
        if (weak, ds, s) in gt and (strong, ds, s) in gt and (weak, strong, ds, loss, s) in tr:
            wg, sg, t = gt[(weak, ds, s)], gt[(strong, ds, s)], tr[(weak, strong, ds, loss, s)]
            if sg - wg > 0: vals.append((t - wg) / (sg - wg))
    return st.median(vals) if vals else None
for ds in TASKS:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{SHORT[w]}→{SHORT[s]}" for s, w in strict]
    x = np.arange(len(strict)); width = 0.38
    for i, (loss, col) in enumerate([("xent", "tab:blue"), ("logconf", "tab:red")]):
        vals = [pgr(w, s, ds, loss) for s, w in strict]
        ax.bar(x + (i - 0.5) * width, [v if v is not None else 0 for v in vals], width, label=loss, color=col)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("PGR (median over seeds)"); ax.set_title(f"Phase 0 — PGR by pair, {ds.upper()} (PGR=1 full recovery, 0 imitation)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f"phase0_pgr_by_pair_{ds}.png", dpi=140); plt.close(fig)
    print(f"wrote phase0_pgr_by_pair_{ds}.png")

# ===== 4. Transfer-acc heatmap (xent), both tasks =====
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, ds in zip(axes, TASKS):
    M = np.full((len(ORDER), len(ORDER)), np.nan)
    for si, strong in enumerate(ORDER):
        for wi, weak in enumerate(ORDER):
            if RANK[weak] < RANK[strong]:
                t = med_tr(weak, strong, ds, "xent")
                if t: M[wi, si] = t
            elif weak == strong:
                g = med_gt(strong, ds)
                if g: M[wi, si] = g
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.5, vmax=0.78)
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[m] for m in ORDER]); ax.set_yticks(range(4))
    ax.set_yticklabels([SHORT[m] for m in ORDER]); ax.set_xlabel("strong (student)"); ax.set_ylabel("weak (teacher)")
    ax.set_title(f"{ds.upper()} transfer acc (diag=GT ceiling)")
    for wi in range(4):
        for si in range(4):
            if not np.isnan(M[wi, si]): ax.text(si, wi, f"{M[wi,si]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Phase 0 — transfer accuracy matrix, xent (3 seeds)")
fig.tight_layout(); fig.savefig(OUT / "phase0_transfer_heatmap.png", dpi=140); plt.close(fig)
print("wrote phase0_transfer_heatmap.png")

# ===== 5. xent vs logconf median PGR, both tasks =====
fig, ax = plt.subplots(figsize=(6.5, 5))
x = np.arange(len(TASKS)); width = 0.35
for i, (loss, col) in enumerate([("xent", "tab:blue"), ("logconf", "tab:red")]):
    vals = [st.median([pgr(w, s, ds, loss) for s, w in strict if pgr(w, s, ds, loss) is not None]) for ds in TASKS]
    ax.bar(x + (i - 0.5) * width, vals, width, label=loss, color=col)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([t.upper() for t in TASKS])
ax.set_ylabel("median PGR over strict pairs"); ax.set_title("Phase 0 — xent vs logconf (logconf is inert/harmful)")
ax.grid(alpha=0.3, axis="y"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "phase0_xent_vs_logconf.png", dpi=140); plt.close(fig)
print("wrote phase0_xent_vs_logconf.png")
print("PHASE 0 PLOTS DONE")
