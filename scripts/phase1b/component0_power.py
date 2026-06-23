#!/usr/bin/env python3
"""Phase 1b — Component 0: power / minimum-detectable-effect (MDE).

Question: can this testbed resolve a *strategy-vs-naive* difference at all, before we
spend compute on Phase 2 strategies? We estimate the noise of the paired per-(pair,seed)
contrast that Components A/B will use (cond − naive at the same pair, seed, fraction),
then report the MDE at 80% power.

Noise is estimated empirically, two independent ways:
  (1) sigma_d_direct: SD over (pair,seed) of Δacc = acc(mixing@0.01) − acc(baseline).
      At 0.01 the true effect is ~nil (Phase 1), so this Δ is ≈ pure paired-difference noise.
  (2) sigma_d_fromrun: sqrt(2) · sigma_run, where sigma_run = pooled across-seed SD of the
      baseline (frac=0) per-pair accuracy. Conservative (treats two runs as independent;
      a shared seed/init would make the real paired noise smaller).

MDE (two-sided, alpha=0.05, power=0.80) = 2.80 · SE ;  SE = sigma_d / sqrt(N).
One-sided (we have directional predictions) = 2.49 · SE.
"""
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "results" / "phase1" / "phase1_results.csv"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
EXCLUDE = {(1, "gpt2-large")}          # pre-stated failed-ceiling filter (carried from Phase 1)
NOISE_FLOOR = 0.014                    # Phase 1 M5 (median per-pair 3-seed range)

def excluded(seed, model):
    return (seed, model) in EXCLUDE

rows = [r for r in csv.DictReader(CSV.open()) if r["loss"] == "xent" and r["ds_name"] == "boolq"]
for r in rows:
    r["seed"] = int(r["seed"]); r["acc"] = float(r["accuracy"]); r["frac"] = float(r["gt_fraction_requested"])

# per-(seed,weak,strong) accuracy: baseline (frac0 anchor) + mixing fractions
acc = {}
for r in rows:
    s, w, strong = r["seed"], r["weak_model"], r["strong_model"]
    if excluded(s, w) or excluded(s, strong):
        continue
    if r["condition"] == "mixing":
        acc[(s, w, strong, r["frac"])] = r["acc"]
    elif r["condition"] == "baseline" and w not in ("", None):
        acc[(s, w, strong, 0.0)] = r["acc"]

# strict transfer pairs (weak < strong) — the meaningful contrast units
def strict(w, s): return w not in ("", None) and RANK.get(w, 99) < RANK.get(s, -1)

# (1) sigma_run from baseline across-seed SD, pooled over strict pairs
base_by_pair = defaultdict(dict)
for (s, w, strong, f), a in acc.items():
    if f == 0.0 and strict(w, strong):
        base_by_pair[(w, strong)][s] = a
run_sds = [st.pstdev(list(d.values())) for d in base_by_pair.values() if len(d) >= 2]
sigma_run = math.sqrt(sum(x * x for x in run_sds) / len(run_sds))  # pooled (RMS)
sigma_d_fromrun = math.sqrt(2) * sigma_run

# (2) sigma_d_direct from Δ(0.01 − 0) paired differences across (pair,seed)
def delta_cells(frac):
    out = []
    for (s, w, strong, f), a in acc.items():
        if f == frac and strict(w, strong) and (s, w, strong, 0.0) in acc:
            out.append(a - acc[(s, w, strong, 0.0)])
    return out

d001 = delta_cells(0.01)
d005 = delta_cells(0.05)
sigma_d_direct = st.pstdev(d001)
N = len(d001)  # number of valid (pair,seed) contrast cells

def mde(sigma_d, n, two_sided=True):
    se = sigma_d / math.sqrt(n)
    z = (1.96 if two_sided else 1.645) + 0.84   # z_{1-a} + z_{power=.80}
    return z * se, se

print("=" * 70)
print("PHASE 1b — COMPONENT 0: POWER / MDE  (xent, BoolQ, seed-1 gpt2-large excluded)")
print("=" * 70)
print(f"\nValid strict-pair contrast cells N = {N}  (pairs × seeds)")
print(f"Noise floor (Phase 1 M5): {NOISE_FLOOR:.4f}")
print(f"\nNoise estimates for the paired (cond − naive) per-cell difference:")
print(f"  sigma_run (baseline across-seed, pooled)      = {sigma_run:.4f}")
print(f"  sigma_d_direct  = SD of Δacc(0.01−0) cells    = {sigma_d_direct:.4f}   [primary]")
print(f"  sigma_d_fromrun = sqrt(2)·sigma_run           = {sigma_d_fromrun:.4f}   [conservative]")
print(f"  (cross-check: Δacc(0.05−0) mean={st.mean(d005):+.4f}, SD={st.pstdev(d005):.4f})")

print(f"\n{'estimator':<34}{'SE':>9}{'MDE 2-sided':>14}{'MDE 1-sided':>14}")
for label, sd in [("primary (sigma_d_direct)", sigma_d_direct),
                  ("conservative (sqrt2·run)", sigma_d_fromrun)]:
    m2, se = mde(sd, N, True); m1, _ = mde(sd, N, False)
    print(f"  {label:<32}{se:>9.4f}{m2:>14.4f}{m1:>14.4f}")

# realized effect sizes for context
real = {}
for f in [0.10, 0.25, 0.50, 1.0]:
    dl = delta_cells(f)
    if dl: real[f] = st.median(dl)
print("\nRealized naive effect sizes (median Δacc vs baseline, for scale):")
for f, v in real.items():
    print(f"  gt_fraction={f}: {v:+.4f}")

mde2 = mde(sigma_d_direct, N, True)[0]
mde1 = mde(sigma_d_direct, N, False)[0]
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"  Primary MDE (1-sided, 80% power): {mde1:.4f} acc")
print(f"  A Phase-2 strategy must beat naive by > ~{mde1:.3f} to be detectable here.")
print(f"  For reference: naive itself moves +{real.get(0.25,float('nan')):.3f} at 0.25, "
      f"+{real.get(0.50,float('nan')):.3f} at 0.50.")
adequate = mde1 <= 0.02
print(f"  Gate (MDE_1sided <= 0.02): {'PASS — testbed can resolve plausible strategy gains' if adequate else 'FAIL — underpowered; add seeds before strategy claims'}")
print(f"  Note: oracle−naive (Component B) is paired at same (pair,seed) → true noise likely")
print(f"        BELOW sigma_d_direct (shared split/init), so this MDE is an upper bound.")
