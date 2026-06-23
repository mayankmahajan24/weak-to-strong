#!/usr/bin/env python3
"""Seed-independent robustness / self-skeptic checks on the seed-1 Phase 1 data.

Run BEFORE seeds 0/2 land, so the pipeline + reading are frozen. Answers:
  1. Validation: gt_fraction_actual vs requested, degenerate/NaN runs, 1.0-vs-ceiling.
  2. Is the xent knee a real aggregate effect or driven by a single pair? (per-pair curve)
  3. Does the knee survive metric choice? (median-PGR vs mean-PGR vs raw-acc-delta)
  4. Effect size vs known noise floor (Phase-0 identity check FP nondeterminism ~0.0018).
"""
import csv
import statistics as st
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "results" / "phase1" / "phase1_seed1_results.csv"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
IDENTITY_NOISE = 0.0018  # Phase 0 M3: gt_fraction=0.0 vs baseline, same hardware

rows = list(csv.DictReader(CSV.open()))
for r in rows:
    r["accuracy"] = float(r["accuracy"])
    r["frac"] = float(r["gt_fraction_requested"])
    r["frac_actual"] = float(r["gt_fraction_actual"])

gt = {(r["loss"], r["strong_model"]): r["accuracy"]
      for r in rows if r["condition"] == "baseline" and r["weak_model"] in ("", None)}

def valid(loss, s, w):
    return (w not in ("", None) and s != w and RANK.get(s, -1) > RANK.get(w, 99)
            and gt[(loss, s)] > gt[(loss, w)])

def pgr(loss, s, w, acc):
    return (acc - gt[(loss, w)]) / (gt[(loss, s)] - gt[(loss, w)])

mix = {}  # (loss,frac,s,w) -> acc
for r in rows:
    if r["condition"] == "mixing":
        mix[(r["loss"], r["frac"], r["strong_model"], r["weak_model"])] = r["accuracy"]
    if r["condition"] == "baseline" and r["weak_model"] not in ("", None):
        mix[(r["loss"], 0.0, r["strong_model"], r["weak_model"])] = r["accuracy"]
gtonly = {(r["loss"], r["frac"], r["strong_model"]): r["accuracy"]
          for r in rows if r["condition"] == "gt_only"}

FRACS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
vpairs = [(s, w) for s in ORDER for w in ORDER if valid("xent", s, w)]

print("=" * 70)
print("CHECK 1 — VALIDATION")
print("=" * 70)
# 1a gt_fraction_actual vs requested.
#   MIXING: actual must track requested (the realized GT share of the trainset).
#   GT_ONLY: actual==1.0 by construction (weak labels discarded -> 100% of used
#            labels are GT); `requested` there denotes the sampled subset size.
mix_worst = max(abs(r["frac"] - r["frac_actual"])
                for r in rows if r["condition"] == "mixing")
go_ok = all(abs(r["frac_actual"] - 1.0) < 1e-6
            for r in rows if r["condition"] == "gt_only")
print(f"  mixing  gt_fraction_actual vs requested: max abs err = {mix_worst:.4f}"
      f"  -> {'OK' if mix_worst < 0.01 else 'FLAG'}")
print(f"  gt_only gt_fraction_actual == 1.0 (expected, labels all-GT): "
      f"{'OK' if go_ok else 'FLAG'}")
# 1b degenerate runs
degen = [r for r in rows if r["accuracy"] != r["accuracy"] or abs(r["accuracy"] - 0.5) < 0.01]
print(f"  degenerate runs (NaN or acc~0.5): {len(degen)}  -> {'OK' if not degen else 'FLAG'}")
# 1c gt_fraction=1.0 vs GT ceiling (should be close but not identical; different split)
print("  gt_fraction=1.0 mixing acc vs GT ceiling (xent, self-pairs excluded):")
for s in ORDER:
    vals = [mix[k] for k in mix if k[0] == "xent" and k[1] == 1.0 and k[2] == s]
    if vals:
        m = st.median(vals)
        print(f"    {s:12s}: mix@1.0={m:.3f}  ceiling={gt[('xent',s)]:.3f}  diff={m-gt[('xent',s)]:+.3f}")

print("\n" + "=" * 70)
print("CHECK 2 — IS THE xent KNEE REAL OR ONE-PAIR-DRIVEN? (per-pair PGR)")
print("=" * 70)
hdr = "  pair".ljust(22) + "".join(f"{f:>8}" for f in FRACS)
print(hdr)
signs = {f: 0 for f in FRACS}
for s, w in vpairs:
    cells = []
    for f in FRACS:
        a = mix.get(("xent", f, s, w))
        if a is None:
            cells.append("   --")
        else:
            p = pgr("xent", s, w, a)
            cells.append(f"{p:+.2f}")
            if f >= 0.25 and p > 0:
                signs[f] += 1
    label = f"{s.replace('gpt2','g')}<-{w.replace('gpt2','g')}"
    print(f"  {label:20s}" + "".join(f"{c:>8}" for c in cells))
print(f"  # pairs with PGR>0 at frac>=0.25:  " +
      "  ".join(f"{f}:{signs[f]}/{len(vpairs)}" for f in FRACS if f >= 0.25))

print("\n" + "=" * 70)
print("CHECK 3 — DOES THE KNEE SURVIVE METRIC CHOICE? (xent)")
print("=" * 70)
print("  frac  median-PGR   mean-PGR   raw-acc-delta-vs-0.0 (median over valid pairs)")
base = {(s, w): mix[("xent", 0.0, s, w)] for s, w in vpairs}
for f in FRACS:
    pgrs, deltas = [], []
    for s, w in vpairs:
        a = mix.get(("xent", f, s, w))
        if a is not None:
            pgrs.append(pgr("xent", s, w, a))
            deltas.append(a - base[(s, w)])
    if pgrs:
        print(f"  {f:<5} {st.median(pgrs):+8.3f}  {st.mean(pgrs):+9.3f}   {st.median(deltas):+.4f}")

print("\n" + "=" * 70)
print("CHECK 4 — EFFECT SIZE vs NOISE FLOOR")
print("=" * 70)
d10 = st.median([mix[("xent", 0.10, s, w)] - base[(s, w)] for s, w in vpairs])
d25 = st.median([mix[("xent", 0.25, s, w)] - base[(s, w)] for s, w in vpairs])
print(f"  Phase-0 identity FP noise floor (within-seed):  {IDENTITY_NOISE:.4f} acc")
print(f"  raw-acc delta at 0.10 vs 0.0:  {d10:+.4f}  ({abs(d10)/IDENTITY_NOISE:.1f}x noise floor)")
print(f"  raw-acc delta at 0.25 vs 0.0:  {d25:+.4f}  ({abs(d25)/IDENTITY_NOISE:.1f}x noise floor)")
print("  (caveat: identity noise floor is a within-seed FP-only lower bound; the true")
print("   cross-seed noise floor needs seeds 0/2 and will be larger.)")
