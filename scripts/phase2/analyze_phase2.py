#!/usr/bin/env python3
"""Phase 2 portfolio analysis — score each method vs naive, against NOTES_phase2 predictions.

For each combination method, at each fraction, computes Δacc(method − naive) over the valid
(pair, seed) cells and reports the three pre-registered readouts:
  - beat-naive   : median Δ > noise floor (0.014)
  - left-shift   : method@0.10 median acc ≥ naive@0.25 median acc (sample efficiency)
  - ceiling-raise: Δ > floor at 0.50 (beats naive where naive works)

Loss-matched baseline: gt_anchored is a logconf method → compared to naive *logconf* mixing;
all other methods are xent → compared to naive *xent* mixing. gt_early_stop spends its budget on
validation, so it's compared to naive xent mixing at the same fraction (does selecting-with-GT
beat training-with-GT). EXCLUDE = {(1,"gpt2-large")}. Reads phase2_<method> run dirs + the
Phase-1 naive baseline from phase1_results.csv.
"""
import csv, glob, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "phase1" / "phase1_results.csv"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
EXCLUDE = {(1, "gpt2-large")}
FLOOR = 0.014
FRACS = [0.1, 0.25, 0.5]
# method -> the naive loss it should be compared against
METHOD_LOSS = {"weighted": "xent", "soft_gt": "xent", "reliability": "xent",
               "gt_early_stop": "xent", "gt_anchored": "logconf"}

def exc(seed, m):
    return (seed, m) in EXCLUDE

# --- naive baseline (Phase-1 mixing), keyed by (loss, seed, weak, strong, frac) ---
naive = {}
for r in csv.DictReader(CSV.open()):
    if r["condition"] == "mixing" and r["ds_name"] == "boolq":
        naive[(r["loss"], int(r["seed"]), r["weak_model"], r["strong_model"],
               float(r["gt_fraction_requested"]))] = float(r["accuracy"])

# --- phase 2 method runs ---
meth = {}  # (method, seed, weak, strong, frac) -> acc
for cfgp in glob.glob(str(ROOT / "results/data/phase2_*/**/config.json"), recursive=True):
    if "phase2_sciq" in cfgp:
        continue  # SciQ confirm runs handled separately
    c = json.load(open(cfgp))
    cm = c.get("combination_method")
    if not cm or cm == "naive":
        continue
    try:
        a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
    except Exception:
        continue
    meth[(cm, c["seed"], c.get("weak_model_size"), c["model_size"], c["gt_fraction"])] = a

methods = sorted({k[0] for k in meth})
strict = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]
print(f"phase2 methods found: {methods}  ({len(meth)} runs)\n")

PRED = {  # NOTES_phase2 directional pre-registrations (for scoring)
    "weighted": "small + at 0.10, fading by 0.50",
    "soft_gt": "≈ neutral / within noise",
    "gt_anchored": "STRONG: rescue logconf toward xent-like",
    "reliability": "+ iff weak errors feature-predictable (uncertain)",
    "gt_early_stop": "small + (sample-efficiency)",
}

print("=" * 80)
print(f"METHOD vs NAIVE — median Δacc over valid (pair,seed), floor={FLOOR}")
print("=" * 80)
summary = {}
for m in methods:
    loss = METHOD_LOSS.get(m, "xent")
    print(f"\n[{m}]  (vs naive {loss})  prediction: {PRED.get(m,'?')}")
    print(f"  {'frac':>5} {'medΔ':>9} {'meanΔ':>9} {'#>0/n':>9} {'method med-acc':>14} {'naive med-acc':>13}")
    by_frac = {}
    for f in FRACS:
        diffs, m_accs, n_accs = [], [], []
        for strong, weak in strict:
            for seed in [0, 1, 2]:
                if exc(seed, weak) or exc(seed, strong):
                    continue
                km = (m, seed, weak, strong, f)
                kn = (loss, seed, weak, strong, f)
                if km in meth and kn in naive:
                    diffs.append(meth[km] - naive[kn])
                    m_accs.append(meth[km]); n_accs.append(naive[kn])
        if diffs:
            med = st.median(diffs)
            by_frac[f] = (med, st.median(m_accs), st.median(n_accs), sum(d > 0 for d in diffs), len(diffs))
            print(f"  {f:>5} {med:>+9.4f} {st.mean(diffs):>+9.4f} {sum(d>0 for d in diffs):>4}/{len(diffs)}"
                  f"  {st.median(m_accs):>14.4f} {st.median(n_accs):>13.4f}")
        else:
            print(f"  {f:>5} {'(no cells yet)':>30}")
    summary[m] = by_frac

print("\n" + "=" * 80)
print("THREE READOUTS + VERDICT")
print("=" * 80)
# naive xent median acc by fraction (for left-shift reference)
def naive_med(loss, f):
    v = [naive[(loss, s, w, st_, f)] for s in [0,1,2] for st_, w in strict
         if (loss, s, w, st_, f) in naive and not exc(s, w) and not exc(s, st_)]
    return st.median(v) if v else None

for m in methods:
    loss = METHOD_LOSS.get(m, "xent")
    bf = summary[m]
    beat = [f for f in FRACS if f in bf and bf[f][0] > FLOOR]
    # left-shift: method@0.10 acc >= naive@0.25 acc (same loss)
    ls = (0.1 in bf and naive_med(loss, 0.25) is not None and bf[0.1][1] >= naive_med(loss, 0.25))
    ceil = (0.5 in bf and bf[0.5][0] > FLOOR)
    verdict = "CLEARS FLOOR @ " + ",".join(str(f) for f in beat) if beat else "within noise (null)"
    print(f"  {m:14s}: beat-naive={'Y@'+str(beat) if beat else 'n'}  "
          f"left-shift={'Y' if ls else 'n'}  ceiling-raise={'Y' if ceil else 'n'}  -> {verdict}")

print("\nAny method with beat-naive=Y is a candidate to PROMOTE to 5 seeds (NOTES_phase2 doctrine)")
print("and to REPLICATE on SciQ (run_portfolio_driver.py --ds=sciq --only=<method>).")
