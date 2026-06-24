#!/usr/bin/env python3
"""Phase 1b plots — the premise gate: de-confound (A), allocation-null (B), SciQ replication (C).

Reads phase1 naive/gt_only (phase1_results.csv) + phase1b_random/phase1b_oracle/sciq_mixing/
sciq_random (run dirs). xent, EXCLUDE={(1,'gpt2-large')}, floor=0.014, MDE=0.0071. Generates:
  phase1b_A_deconfound.png   random_labels < gt_only < naive mixing (BoolQ + SciQ)
  phase1b_B_allocation.png   oracle − naive Δ per fraction + per-pair points vs floor/MDE (null)
  phase1b_C_sciq_vs_boolq.png  PGR & Δacc vs fraction, SciQ vs BoolQ (replication, cleaner PGR)
"""
import csv, glob, json, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/plots"; OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]; RANK = {m: i for i, m in enumerate(ORDER)}
EXCLUDE = {(1, "gpt2-large")}; FLOOR = 0.014; MDE = 0.0071; SEEDS = [0, 1, 2]
strict = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]
def exc(seed, m): return (seed, m) in EXCLUDE

# ---- phase1 csv: naive mixing + gt_only + frac0 transfers + GT ceilings (boolq xent) ----
naive_b, gtonly_b = {}, {}     # (w,s,frac,seed) / (s,frac,seed) -> acc
base_b, gt_b = {}, {}          # frac0 transfer (w,s,seed) ; GT ceiling (model,seed)
for r in csv.DictReader((ROOT / "results/phase1/phase1_results.csv").open()):
    if r["ds_name"] != "boolq" or r["loss"] != "xent": continue
    f, s = float(r["gt_fraction_requested"]), int(r["seed"]); acc = float(r["accuracy"])
    w, sg = r["weak_model"], r["strong_model"]
    if r["condition"] == "mixing":
        naive_b[(w, sg, f, s)] = acc
    elif r["condition"] == "gt_only":
        gtonly_b[(sg, f, s)] = acc
    elif r["condition"] == "baseline":
        if w == "": gt_b[(sg, s)] = acc                 # GT ceiling
        elif w != sg: base_b[(w, sg, s)] = acc          # frac0 transfer

def load_dir(sub):
    d = {}
    for cfgp in glob.glob(str(ROOT / f"results/data/{sub}/**/config.json"), recursive=True):
        c = json.load(open(cfgp))
        try: a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
        except Exception: continue
        d[(c.get("weak_model_size"), c["model_size"], c["gt_fraction"], c["seed"])] = a
    return d
rand_b = load_dir("phase1b_random"); oracle_b = load_dir("phase1b_oracle")
mix_s = load_dir("sciq_mixing"); rand_s = load_dir("sciq_random")

def med(vals): return st.median(vals) if vals else None
def cells(frac, src):  # (w,s,seed) keys present in src at this frac, exclusion applied
    return [(w, sg, seed) for (w, sg, f, seed) in src if f == frac and not exc(seed, w) and not exc(seed, sg)]

# ===== A — de-confound: random < gt_only < mixing =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (task, mixsrc, randsrc, gtsrc) in zip(
        axes, [("BoolQ", naive_b, rand_b, gtonly_b), ("SciQ", mix_s, rand_s, None)]):
    fracs = sorted({f for (_, _, f, _) in randsrc})
    x = np.arange(len(fracs)); width = 0.27
    bars = [("random_labels", randsrc, "tab:gray"), ("naive mixing", mixsrc, "tab:green")]
    if gtsrc is not None: bars.insert(1, ("gt_only", gtsrc, "tab:orange"))
    for i, (name, src, col) in enumerate(bars):
        vals = []
        for f in fracs:
            cc = cells(f, randsrc)  # restrict to the random cells for a matched comparison
            if name == "gt_only":
                vals.append(med([src[(sg, f, seed)] for (w, sg, seed) in cc if (sg, f, seed) in src]))
            else:
                vals.append(med([src[(w, sg, f, seed)] for (w, sg, seed) in cc if (w, sg, f, seed) in src]))
        off = (i - (len(bars) - 1) / 2) * width
        ax.bar(x + off, [v or 0 for v in vals], width, label=name, color=col)
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([f"{f:.2f}" for f in fracs])
    ax.set_xlabel("GT fraction"); ax.set_title(f"{task}"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
axes[0].set_ylabel("median test accuracy")
fig.suptitle("Phase 1b · A — weak labels are INFORMATIVE: random_labels < gt_only < naive mixing\n(noise replacing weak labels HURTS → mixing's win is real information, not data quantity)")
fig.tight_layout(); fig.savefig(OUT / "phase1b_A_deconfound.png", dpi=140); plt.close(fig)
print("wrote phase1b_A_deconfound.png")

# ===== B — allocation null: oracle − naive =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fracs = sorted({f for (_, _, f, _) in oracle_b})
# left: per-pair Δ points + median, with floor/MDE bands
axL, axR = axes
for xi, f in enumerate(fracs):
    diffs = []
    for (w, sg, seed) in cells(f, oracle_b):
        if (w, sg, f, seed) in oracle_b and (w, sg, f, seed) in naive_b:
            diffs.append(oracle_b[(w, sg, f, seed)] - naive_b[(w, sg, f, seed)])
    jit = np.random.default_rng(xi).normal(0, 0.04, len(diffs))
    axL.scatter(np.full(len(diffs), xi) + jit, diffs, color="tab:purple", alpha=0.6, s=28)
    axL.scatter([xi], [med(diffs)], color="k", marker="D", s=70, zorder=5, label="median" if xi == 0 else None)
axL.axhspan(-FLOOR, FLOOR, color="gray", alpha=0.15, label="noise floor ±0.014")
axL.axhline(MDE, color="red", ls="--", lw=0.8, label="MDE 0.0071"); axL.axhline(-MDE, color="red", ls="--", lw=0.8)
axL.axhline(0, color="k", lw=0.8); axL.set_xticks(range(len(fracs))); axL.set_xticklabels([f"{f:.2f}" for f in fracs])
axL.set_xlabel("GT fraction"); axL.set_ylabel("Δacc(oracle − naive) per (pair,seed)")
axL.set_title("Oracle allocation buys nothing\n(coin-flip sign, within noise)"); axL.legend(fontsize=7); axL.grid(alpha=0.3)
# right: oracle vs naive median acc bars
x = np.arange(len(fracs)); width = 0.38
for i, (name, src, col) in enumerate([("naive (random placement)", naive_b, "tab:green"), ("oracle (perfect targeting)", oracle_b, "tab:purple")]):
    vals = []
    for f in fracs:
        cc = cells(f, oracle_b)
        vals.append(med([src[(w, sg, f, seed)] for (w, sg, seed) in cc if (w, sg, f, seed) in src]))
    axR.bar(x + (i - 0.5) * width, [v or 0 for v in vals], width, label=name, color=col)
axR.set_xticks(x); axR.set_xticklabels([f"{f:.2f}" for f in fracs]); axR.set_ylim(0.6, 0.72)
axR.set_xlabel("GT fraction"); axR.set_ylabel("median acc"); axR.set_title("Oracle ≈ naive at every budget"); axR.legend(fontsize=8); axR.grid(alpha=0.3, axis="y")
fig.suptitle("Phase 1b · B — WHERE you spend GT does not matter (allocation NULL): a perfect error-targeting oracle ties random")
fig.tight_layout(); fig.savefig(OUT / "phase1b_B_allocation.png", dpi=140); plt.close(fig)
print("wrote phase1b_B_allocation.png")

# ===== C — SciQ vs BoolQ replication (PGR + Δacc vs fraction) =====
# Source-consistent + paired over STRICT pairs: BoolQ from the canonical csv (frac0=base_b,
# ceilings=gt_b), SciQ from the dirs (frac0 + ceilings from baseline-dir). This makes the BoolQ
# Δacc curve match the Phase-1 knee diagnostic (+0.003…+0.075) rather than the pooled-bias version.
def sciq_ref():
    gt, b0 = {}, {}
    for s in SEEDS:
        for cfgp in glob.glob(str(ROOT / f"results/data/baseline/seed{s}/*/config.json")):
            c = json.load(open(cfgp))
            if c.get("ds_name") != "sciq" or c.get("gt_fraction", 0) not in (0, 0.0, None): continue
            try: a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
            except Exception: continue
            if not c.get("weak_model_size") and c.get("loss", "xent") == "xent": gt[(c["model_size"], s)] = a
            elif c.get("weak_model_size"): b0[(c["weak_model_size"], c["model_size"], s)] = a
    return gt, b0
gt_s, base_s = sciq_ref()
def pgr_curve(task, mixsrc):
    gt, b0 = (gt_b, base_b) if task == "BoolQ" else (gt_s, base_s)
    fracs = sorted({f for (_, _, f, _) in mixsrc if f > 0})
    pgr_by_f, dacc_by_f = {}, {}
    for f in fracs:
        pgrs, daccs = [], []
        for (sg, w) in strict:            # STRICT pairs only, both metrics
            for seed in SEEDS:
                if exc(seed, w) or exc(seed, sg) or (w, sg, f, seed) not in mixsrc: continue
                m = mixsrc[(w, sg, f, seed)]
                if (w, sg, seed) in b0: daccs.append(m - b0[(w, sg, seed)])      # paired Δacc
                if (w, seed) in gt and (sg, seed) in gt and gt[(sg, seed)] - gt[(w, seed)] > 0:
                    pgrs.append((m - gt[(w, seed)]) / (gt[(sg, seed)] - gt[(w, seed)]))
        pgr_by_f[f] = med(pgrs); dacc_by_f[f] = med(daccs)
    return fracs, pgr_by_f, dacc_by_f

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for task, mixsrc, col in [("BoolQ", naive_b, "tab:blue"), ("SciQ", mix_s, "tab:green")]:
    fr, pg, da = pgr_curve(task, mixsrc)
    axes[0].plot(fr, [pg[f] for f in fr], "o-", color=col, label=task)
    axes[1].plot(fr, [da[f] for f in fr], "o-", color=col, label=task)
axes[0].axhline(0, color="k", lw=0.8); axes[0].set_xlabel("GT fraction"); axes[0].set_ylabel("median PGR")
axes[0].set_title("PGR vs fraction — SciQ positive from 0.10, BoolQ starts negative"); axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xlabel("GT fraction"); axes[1].set_ylabel("median Δacc vs frac0")
axes[1].set_title("Δacc vs fraction — both gradual/no-knee; SciQ smaller raw"); axes[1].grid(alpha=0.3); axes[1].legend()
fig.suptitle("Phase 1b · C — findings REPLICATE on SciQ (cleaner in PGR, smaller in raw acc; no-knee shape holds both tasks)")
fig.tight_layout(); fig.savefig(OUT / "phase1b_C_sciq_vs_boolq.png", dpi=140); plt.close(fig)
print("wrote phase1b_C_sciq_vs_boolq.png")
print("PHASE 1b PLOTS DONE")
