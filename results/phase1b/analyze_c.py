#!/usr/bin/env python3
"""Phase 1b Component C — SciQ vs BoolQ (testbed validity) + SciQ de-confound.

Q: is GT-mixing larger / cleaner where baseline W2SG signal is positive (SciQ, PGR≈+0.17)
than where it's ~zero (BoolQ)? And does the weak-label-informativeness of Component A
replicate on SciQ?

Reads SciQ baseline (ceilings + frac0 transfer anchors) from results/data/baseline/, the
sciq_mixing / sciq_random run dirs, and BoolQ naive mixing from phase1_results.csv.
Per-seed validity rule (drop weak_GT>strong_GT) applied for PGR; raw-acc uses all pairs.
"""
import csv, glob, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
FRACS = [0.1, 0.25, 0.5, 1.0]

# ---------- SciQ baseline: GT ceilings + frac0 transfer anchors ----------
sgt, sbase = {}, {}   # (seed,model)->ceiling ; (seed,weak,strong)->frac0 transfer acc
for cfgp in glob.glob(str(ROOT / "results/data/baseline/seed*/*/config.json")):
    c = json.load(open(cfgp))
    if c.get("ds_name") != "sciq" or c.get("loss") != "xent":
        continue
    sp = cfgp.replace("config.json", "results_summary.json")
    try:
        a = json.load(open(sp))["accuracy"]
    except Exception:
        continue
    s, strong, weak = c["seed"], c["model_size"], c.get("weak_model_size")
    if not weak:
        sgt[(s, strong)] = a
    else:
        sbase[(s, weak, strong)] = a

# ---------- SciQ mixing + random ----------
smix, srand = {}, {}
for cfgp in glob.glob(str(ROOT / "results/data/sciq_*/**/config.json"), recursive=True):
    c = json.load(open(cfgp))
    a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
    s, f, strat = c["seed"], c["gt_fraction"], c.get("mixing_strategy")
    strong, weak = c["model_size"], c.get("weak_model_size")
    if strat == "random_labels":
        srand[(s, strong, f)] = a
    else:
        smix[(s, weak, strong, f)] = a

# ---------- BoolQ naive mixing + frac0 from phase1_results.csv ----------
bmix, bbase, bgt = {}, {}, {}
for r in csv.DictReader((ROOT / "results/phase1/phase1_results.csv").open()):
    if r["loss"] != "xent" or r["ds_name"] != "boolq":
        continue
    s = int(r["seed"]); a = float(r["accuracy"]); f = float(r["gt_fraction_requested"])
    if r["condition"] == "mixing":
        bmix[(s, r["weak_model"], r["strong_model"], f)] = a
    elif r["condition"] == "baseline" and r["weak_model"] not in ("", None):
        bbase[(s, r["weak_model"], r["strong_model"])] = a
    elif r["condition"] == "baseline":
        bgt[(s, r["strong_model"])] = a

BEXCL = {(1, "gpt2-large")}
def valid(gt, s, w, strong):
    if RANK[w] >= RANK[strong]:
        return False
    if (s, w) not in gt or (s, strong) not in gt:
        return False
    return gt[(s, w)] > gt[(s, strong)] is False and gt[(s, strong)] > gt[(s, w)]

pairs_all = [(w, s) for s in ORDER for w in ORDER if RANK[w] <= RANK[s]]   # 10 (incl self)
strict = [(w, s) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]        # 6

def curve(mix, base, gt, exclude=set(), dataset=""):
    """median Δacc(frac-0) over pairs×seeds (raw, all pairs) + median PGR (valid strict)."""
    out = {}
    for f in FRACS:
        dacc, pgrs = [], []
        for seed in [0, 1, 2]:
            for w, strong in pairs_all:
                if (seed, strong) in exclude or (seed, w) in exclude:
                    continue
                if (seed, w, strong, f) in mix and (seed, w, strong) in base:
                    dacc.append(mix[(seed, w, strong, f)] - base[(seed, w, strong)])
            for w, strong in strict:
                if (seed, strong) in exclude or (seed, w) in exclude:
                    continue
                if (seed, w) in gt and (seed, strong) in gt and gt[(seed, strong)] > gt[(seed, w)] \
                   and (seed, w, strong, f) in mix:
                    denom = gt[(seed, strong)] - gt[(seed, w)]
                    pgrs.append((mix[(seed, w, strong, f)] - gt[(seed, w)]) / denom)
        out[f] = (st.median(dacc) if dacc else None, len(dacc),
                  st.median(pgrs) if pgrs else None, len(pgrs))
    return out

sc = curve(smix, sbase, sgt, exclude=set(), dataset="sciq")
bc = curve(bmix, bbase, bgt, exclude=BEXCL, dataset="boolq")

print("=" * 78)
print("COMPONENT C — SciQ vs BoolQ naive-mixing fraction curve (xent)")
print("=" * 78)
print(f"  {'frac':>5} | {'SciQ Δacc':>10} {'SciQ PGR':>9} | {'BoolQ Δacc':>11} {'BoolQ PGR':>10}")
for f in FRACS:
    sd, sn, sp, spn = sc[f]; bd, bn, bp, bpn = bc[f]
    print(f"  {f:>5} | {sd:>+10.4f} {sp:>+9.3f} | {bd:>+11.4f} {bp:>+10.3f}")
print("  (Δacc = median over all pairs×seeds vs frac0; PGR = median over valid strict pairs)")

print("\n" + "=" * 78)
print("COMPONENT A REPLICATION ON SciQ — naive mixing vs random_labels (same rows/steps)")
print("=" * 78)
print(f"  {'frac':>5} {'mix−rand (median)':>18} {'#>0 / n':>9}   random_labels median acc")
for f in [0.1, 0.25]:
    diffs, rvals = [], []
    for seed in [0, 1, 2]:
        for w, strong in strict:
            if (seed, w, strong, f) in smix and (seed, strong, f) in srand:
                diffs.append(smix[(seed, w, strong, f)] - srand[(seed, strong, f)])
        for strong in ORDER:
            if (seed, strong, f) in srand:
                rvals.append(srand[(seed, strong, f)])
    print(f"  {f:>5} {st.median(diffs):>+18.4f} {sum(d>0 for d in diffs):>5}/{len(diffs)}"
          f"        {st.median(rvals):.4f}")

print("\n" + "=" * 78)
print("READ")
print("=" * 78)
print(f"  SciQ Δacc @0.25 = {sc[0.25][0]:+.4f} vs BoolQ {bc[0.25][0]:+.4f}")
print(f"  SciQ Δacc @0.50 = {sc[0.5][0]:+.4f} vs BoolQ {bc[0.5][0]:+.4f}")
print(f"  SciQ Δacc @1.00 = {sc[1.0][0]:+.4f} vs BoolQ {bc[1.0][0]:+.4f}")
