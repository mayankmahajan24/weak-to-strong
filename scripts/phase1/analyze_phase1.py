#!/usr/bin/env python3
"""Phase 1 seed-1 analysis: fraction curve, mixing vs GT-only, scale interaction.

PGR = (transfer_acc - weak_GT_acc) / (strong_GT_acc - weak_GT_acc)
  - GT acc = blank-weak baseline run (model fine-tuned on true labels).
  - Valid pairs: strict (strong arch > weak arch) AND strong_GT > weak_GT.
  - Report median PGR across valid pairs (per paper convention).
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "results" / "phase1" / "phase1_seed1_results.csv"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
rows = list(csv.DictReader(CSV.open()))
for r in rows:
    r["accuracy"] = float(r["accuracy"])
    r["frac"] = float(r["gt_fraction_requested"])

# --- GT ceilings: blank-weak baseline runs, per (loss, model) ---
gt = {}  # (loss, model) -> acc
for r in rows:
    if r["condition"] == "baseline" and r["weak_model"] in ("", None):
        gt[(r["loss"], r["strong_model"])] = r["accuracy"]

print("GT ceilings (blank-weak baseline):")
for loss in ("xent", "logconf"):
    print(f"  {loss}: " + "  ".join(f"{m.replace('gpt2','g')}={gt[(loss,m)]:.3f}" for m in ORDER))

def valid_pair(loss, strong, weak):
    if weak in ("", None) or strong == weak:
        return False
    if RANK.get(strong, -1) <= RANK.get(weak, 99):
        return False
    return gt[(loss, strong)] > gt[(loss, weak)]

def pgr(loss, strong, weak, acc):
    denom = gt[(loss, strong)] - gt[(loss, weak)]
    return (acc - gt[(loss, weak)]) / denom

# enumerate valid pairs per loss
valid = {loss: [(s, w) for s in ORDER for w in ORDER if valid_pair(loss, s, w)]
         for loss in ("xent", "logconf")}
print("\nValid PGR pairs:")
for loss in ("xent", "logconf"):
    print(f"  {loss} ({len(valid[loss])}): " +
          ", ".join(f"{s.replace('gpt2','g')}<-{w.replace('gpt2','g')}" for s, w in valid[loss]))

# --- collect transfer accuracies for mixing & gt_only by (condition,loss,frac,strong,weak) ---
acc = {}
for r in rows:
    if r["condition"] in ("mixing",):
        acc[("mixing", r["loss"], r["frac"], r["strong_model"], r["weak_model"])] = r["accuracy"]
# gt_only runs are stored once per (model,loss) since weak labels are discarded; key by strong model
gtonly = {}  # (loss,frac,strong) -> acc
for r in rows:
    if r["condition"] == "gt_only":
        gtonly[(r["loss"], r["frac"], r["strong_model"])] = r["accuracy"]

# baseline transfer (frac=0) accuracies, for the 0.0 anchor of the mixing curve
for r in rows:
    if r["condition"] == "baseline" and r["weak_model"] not in ("", None):
        acc[("mixing", r["loss"], 0.0, r["strong_model"], r["weak_model"])] = r["accuracy"]

FRACS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]  # seed-1 recovery csv (no 0.75); 3-seed 0.75 lives in phase1_results.csv

def median_pgr_mixing(loss, frac):
    vals = []
    for s, w in valid[loss]:
        a = acc.get(("mixing", loss, frac, s, w))
        if a is not None:
            vals.append(pgr(loss, s, w, a))
    return (st.median(vals), len(vals)) if vals else (None, 0)

def median_pgr_gtonly(loss, frac):
    # GT-only: strong model trained on `frac` GT only; pair it against each valid weak floor
    vals = []
    for s, w in valid[loss]:
        a = gtonly.get((loss, frac, s))
        if a is not None:
            vals.append(pgr(loss, s, w, a))
    return (st.median(vals), len(vals)) if vals else (None, 0)

print("\n" + "=" * 72)
print("FRACTION CURVE — median PGR over valid pairs")
print("=" * 72)
for loss in ("xent", "logconf"):
    print(f"\n[{loss}]  frac :   mixing PGR   |  gt_only PGR")
    for f in FRACS:
        mp, mn = median_pgr_mixing(loss, f)
        gp, gn = median_pgr_gtonly(loss, f) if f > 0 else (None, 0)
        ms = f"{mp:+.3f} (n={mn})" if mp is not None else "   --      "
        gs = f"{gp:+.3f} (n={gn})" if gp is not None else "   --"
        print(f"        {f:<5}:  {ms}  |  {gs}")

# --- raw median accuracy curve (all 10 pairs, simpler signal) ---
print("\n" + "=" * 72)
print("RAW median accuracy over all 10 mixing pairs (incl self/inverted)")
print("=" * 72)
for loss in ("xent", "logconf"):
    print(f"\n[{loss}] frac : median_acc  (min..max)")
    for f in FRACS:
        vals = [acc[k] for k in acc if k[0] == "mixing" and k[1] == loss and k[2] == f]
        if vals:
            print(f"       {f:<5}: {st.median(vals):.4f}  ({min(vals):.3f}..{max(vals):.3f})  n={len(vals)}")

# --- mixing vs gt_only head-to-head (same strong model, raw acc) ---
print("\n" + "=" * 72)
print("MIXING vs GT-ONLY (raw acc, xent) — does mixing beat data-starved GT-only?")
print("=" * 72)
print("  Per fraction: median over the 4 strong models of [best mixing acc] vs [gt_only acc]")
for f in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
    mix_by_strong, go_by_strong = [], []
    for s in ORDER:
        mvals = [acc[k] for k in acc if k[0] == "mixing" and k[1] == "xent" and k[2] == f and k[3] == s]
        g = gtonly.get(("xent", f, s))
        if mvals and g is not None:
            mix_by_strong.append(st.median(mvals))
            go_by_strong.append(g)
    if mix_by_strong:
        dm, dg = st.median(mix_by_strong), st.median(go_by_strong)
        print(f"  frac={f:<5}: mixing={dm:.4f}  gt_only={dg:.4f}  delta={dm-dg:+.4f}")

# --- scale interaction: PGR slope vs gt_fraction, per strong model (xent) ---
print("\n" + "=" * 72)
print("SCALE INTERACTION (xent) — per-strong-model PGR across fraction")
print("=" * 72)
print("  Strong model PGR averaged over its valid weak partners, by fraction")
for s in ORDER:
    partners = [w for (ss, w) in valid["xent"] if ss == s]
    if not partners:
        print(f"  {s:12s}: no valid pairs (GT anomaly or smallest model)")
        continue
    line = []
    for f in FRACS:
        vals = []
        for w in partners:
            a = acc.get(("mixing", "xent", f, s, w))
            if a is not None:
                vals.append(pgr("xent", s, w, a))
        line.append(f"{f}={st.median(vals):+.2f}" if vals else f"{f}=--")
    print(f"  {s:12s} (weak={[p.replace('gpt2','g') for p in partners]}): " + "  ".join(line))
