#!/usr/bin/env python3
"""Phase 1b Components A & B analysis (local).

A — De-confound: does weak-label INFORMATION beat row-count? Compare naive mixing vs
    random_labels (same rows/steps, non-GT labels randomized) at the same strong model.
B — Oracle ceiling: does targeting GT at the weak teacher's errors beat naive (random
    allocation) at the same pair? Upper bound on allocation value.

Reads phase1b_{oracle,random} run dirs + phase1_results.csv (naive mixing, gt_only, GT
ceilings). xent, BoolQ. EXCLUDE seed-1 gpt2-large. Read against noise floor 0.014 and the
Component-0 aggregate MDE 0.0071.
"""
import csv, glob, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results/phase1/phase1_results.csv"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
EXCLUDE = {(1, "gpt2-large")}
FLOOR, MDE = 0.014, 0.0071
FRACS = [0.1, 0.25]

def exc(seed, m): return (seed, m) in EXCLUDE

# --- phase 1 naive mixing + gt_only + GT ceilings (xent) ---
naive, gtonly, gt = {}, {}, {}
for r in csv.DictReader(CSV.open()):
    if r["loss"] != "xent" or r["ds_name"] != "boolq":
        continue
    s = int(r["seed"]); a = float(r["accuracy"]); f = float(r["gt_fraction_requested"])
    if r["condition"] == "mixing":
        naive[(s, r["weak_model"], r["strong_model"], f)] = a
    elif r["condition"] == "gt_only":
        gtonly[(s, r["strong_model"], f)] = a
    elif r["condition"] == "baseline" and r["weak_model"] in ("", None):
        gt[(s, r["strong_model"])] = a

# --- phase 1b runs ---
oracle, randl = {}, {}
for cfgp in glob.glob(str(ROOT / "results/data/phase1b_*/**/config.json"), recursive=True):
    c = json.load(open(cfgp))
    summ = json.load(open(cfgp.replace("config.json", "results_summary.json")))
    s = c["seed"]; f = c["gt_fraction"]; a = summ["accuracy"]; strat = c.get("mixing_strategy")
    strong = c["model_size"]; weak = c.get("weak_model_size")
    if strat == "oracle":
        oracle[(s, weak, strong, f)] = a
    elif strat == "random_labels":
        randl[(s, strong, f)] = a

print("=" * 74)
print("COMPONENT B — ORACLE vs NAIVE (xent, BoolQ; oracle−naive at same pair,frac,seed)")
print("=" * 74)
strict = [(w, s) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]
print(f"  {'frac':>5} {'median Δ':>10} {'mean Δ':>9} {'#>0 / n':>9} {'vs floor/MDE':>16}")
B = {}
for f in FRACS:
    diffs = []
    for w, strong in strict:
        for seed in [0, 1, 2]:
            if exc(seed, w) or exc(seed, strong):
                continue
            k_o = (seed, w, strong, f); k_n = (seed, w, strong, f)
            if k_o in oracle and k_n in naive:
                diffs.append(oracle[k_o] - naive[k_n])
    B[f] = diffs
    med = st.median(diffs); mean = st.mean(diffs); npos = sum(d > 0 for d in diffs)
    verdict = "ABOVE" if med > FLOOR else ("> MDE" if med > MDE else "within noise")
    print(f"  {f:>5} {med:>+10.4f} {mean:>+9.4f} {npos:>5}/{len(diffs)} {verdict:>16}")

print("\n" + "=" * 74)
print("COMPONENT A — WEAK-LABEL INFO: naive mixing vs random_labels (same rows/steps)")
print("=" * 74)
print("  per strong model, median over (weak-partner, seed) of [naive_mixing − random_labels]")
print(f"  {'frac':>5} {'strong':>12} {'mix−rand':>10} {'#>0 / n':>9}")
A = {}
for f in FRACS:
    alld = []
    for strong in ["gpt2-medium", "gpt2-large", "gpt2-xl"]:  # have weak partners
        d = []
        for w in ORDER:
            if RANK[w] >= RANK[strong]:
                continue
            for seed in [0, 1, 2]:
                if exc(seed, w) or exc(seed, strong):
                    continue
                if (seed, w, strong, f) in naive and (seed, strong, f) in randl:
                    d.append(naive[(seed, w, strong, f)] - randl[(seed, strong, f)])
        alld += d
        if d:
            print(f"  {f:>5} {strong:>12} {st.median(d):>+10.4f} {sum(x>0 for x in d):>5}/{len(d)}")
    A[f] = alld
    print(f"  {f:>5} {'ALL':>12} {st.median(alld):>+10.4f} {sum(x>0 for x in alld):>5}/{len(alld)}  "
          f"<- {'ABOVE floor' if st.median(alld)>FLOOR else ('> MDE' if st.median(alld)>MDE else 'within noise')}")

print("\n" + "=" * 74)
print("ORDERING CONTEXT — median acc by condition (xent, valid strict pairs, excl applied)")
print("=" * 74)
print(f"  {'frac':>5} {'gt_only':>9} {'random_lbl':>11} {'naive_mix':>10} {'oracle':>9}")
for f in FRACS:
    def med_over(d, key):
        v = [d[k] for k in key if k in d]
        return st.median(v) if v else float("nan")
    # gt_only & random by strong model; naive & oracle by pair
    go = [gtonly[(s, strong, f)] for s in [0,1,2] for strong in ["gpt2-medium","gpt2-large","gpt2-xl"]
          if (s, strong, f) in gtonly and not exc(s, strong)]
    rl = [randl[(s, strong, f)] for s in [0,1,2] for strong in ["gpt2-medium","gpt2-large","gpt2-xl"]
          if (s, strong, f) in randl and not exc(s, strong)]
    nm = [naive[(s,w,strong,f)] for s in [0,1,2] for w,strong in strict
          if (s,w,strong,f) in naive and not exc(s,w) and not exc(s,strong)]
    orc = [oracle[(s,w,strong,f)] for s in [0,1,2] for w,strong in strict
           if (s,w,strong,f) in oracle and not exc(s,w) and not exc(s,strong)]
    print(f"  {f:>5} {st.median(go):>9.4f} {st.median(rl):>11.4f} {st.median(nm):>10.4f} {st.median(orc):>9.4f}")

print("\n" + "=" * 74)
print("VERDICTS (vs pre-registered kill criteria)")
print("=" * 74)
for f in FRACS:
    bmed = st.median(B[f]); amed = st.median(A[f])
    print(f"  frac {f}: B oracle−naive = {bmed:+.4f} ({'HEADROOM' if bmed>FLOOR else 'null' } vs floor {FLOOR}); "
          f"A mix−random = {amed:+.4f} ({'INFORMATIVE' if amed>FLOOR else 'artifact/within-noise'})")
