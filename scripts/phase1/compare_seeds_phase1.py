#!/usr/bin/env python3
"""Per-seed comparison of the Phase 1 fraction sweep, scored against the
pre-registered predictions in NOTES_phase1.md (P1-P6).

Reads phase1_results.csv (multi-seed). Computes, per seed: GT ceilings, valid
pairs, the xent/logconf fraction curves (raw acc + median PGR), mixing-vs-GT-only,
and the scale-interaction zero-crossing. Then reports seed0-vs-seed1 agreement.
All analysis decisions follow NOTES_phase1.md and are NOT changed here.
"""
import csv
import statistics as st
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "results" / "phase1" / "phase1_results.csv"
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
FRACS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
SEEDS = [0, 1]

rows = list(csv.DictReader(CSV.open()))
for r in rows:
    r["accuracy"] = float(r["accuracy"])
    r["frac"] = float(r["gt_fraction_requested"])
    r["seed"] = int(r["seed"])


def gt_ceilings(seed):
    return {(r["loss"], r["strong_model"]): r["accuracy"] for r in rows
            if r["seed"] == seed and r["condition"] == "baseline" and r["weak_model"] in ("", None)}


def valid_pairs(gt, loss):
    return [(s, w) for s in ORDER for w in ORDER
            if w not in ("", None) and s != w and RANK[s] > RANK[w]
            and (loss, s) in gt and (loss, w) in gt and gt[(loss, s)] > gt[(loss, w)]]


def mixing_acc(seed):
    d = {}
    for r in rows:
        if r["seed"] != seed:
            continue
        if r["condition"] == "mixing":
            d[(r["loss"], r["frac"], r["strong_model"], r["weak_model"])] = r["accuracy"]
        if r["condition"] == "baseline" and r["weak_model"] not in ("", None):
            d[(r["loss"], 0.0, r["strong_model"], r["weak_model"])] = r["accuracy"]
    return d


def gtonly_acc(seed):
    return {(r["loss"], r["frac"], r["strong_model"]): r["accuracy"] for r in rows
            if r["seed"] == seed and r["condition"] == "gt_only"}


def pgr(gt, loss, s, w, acc):
    return (acc - gt[(loss, w)]) / (gt[(loss, s)] - gt[(loss, w)])


PER = {}
for seed in SEEDS:
    gt = gt_ceilings(seed)
    PER[seed] = dict(gt=gt, vp={L: valid_pairs(gt, L) for L in ("xent", "logconf")},
                     mix=mixing_acc(seed), go=gtonly_acc(seed))

# ---------- GT ceilings ----------
print("=" * 74)
print("GT CEILINGS (xent) — note seed-1 gpt2-large anomaly")
print("=" * 74)
print(f"  {'seed':5s} " + "".join(f"{m.replace('gpt2','g'):>10}" for m in ORDER))
for seed in SEEDS:
    g = PER[seed]["gt"]
    print(f"  {seed:<5} " + "".join(f"{g[('xent', m)]:>10.3f}" for m in ORDER))
print("  valid xent pairs/seed: " +
      "  ".join(f"seed{s}={len(PER[s]['vp']['xent'])}" for s in SEEDS))
for s in SEEDS:
    print(f"    seed{s}: " + ", ".join(f"{a.replace('gpt2','g')}<-{b.replace('gpt2','g')}"
                                       for a, b in PER[s]["vp"]["xent"]))

# ---------- xent fraction curve, both seeds ----------
def curve_rawacc(seed, loss):
    out = {}
    mix = PER[seed]["mix"]
    for f in FRACS:
        vals = [v for k, v in mix.items() if k[0] == loss and k[1] == f]
        out[f] = st.median(vals) if vals else None
    return out

def curve_pgr(seed, loss):
    out = {}
    gt, mix, vp = PER[seed]["gt"], PER[seed]["mix"], PER[seed]["vp"][loss]
    for f in FRACS:
        vals = [pgr(gt, loss, s, w, mix[(loss, f, s, w)]) for s, w in vp if (loss, f, s, w) in mix]
        out[f] = st.median(vals) if vals else None
    return out

print("\n" + "=" * 74)
print("P1/P2/P6 — xent FRACTION CURVE per seed")
print("=" * 74)
for metric, fn in [("raw median acc (all 10 pairs)", curve_rawacc), ("median PGR (valid pairs)", curve_pgr)]:
    print(f"\n  [{metric}]")
    print("    seed " + "".join(f"{f:>8}" for f in FRACS))
    for seed in SEEDS:
        c = fn(seed, "xent")
        print(f"    {seed:<5}" + "".join((f"{c[f]:>8.3f}" if c[f] is not None else f"{'--':>8}") for f in FRACS))
# knee deltas
print("\n  knee check (raw acc, median over 10 pairs):")
for seed in SEEDS:
    c = curve_rawacc(seed, "xent")
    d10 = c[0.10] - c[0.0]; d25 = c[0.25] - c[0.0]; d2510 = c[0.25] - c[0.10]; d5025 = c[0.50] - c[0.25]
    print(f"    seed{seed}: Δ(.10-.0)={d10:+.4f}  Δ(.25-.0)={d25:+.4f}  "
          f"Δ(.25-.10)={d2510:+.4f}  Δ(.50-.25)={d5025:+.4f}")

# ---------- mixing vs gt_only (P3) ----------
print("\n" + "=" * 74)
print("P3 — MIXING vs GT-ONLY (xent, raw acc; median over 4 strong models)")
print("=" * 74)
print("    seed " + "".join(f"{f:>9}" for f in FRACS[1:]))
for seed in SEEDS:
    mix, go = PER[seed]["mix"], PER[seed]["go"]
    cells = []
    for f in FRACS[1:]:
        diffs = []
        for s in ORDER:
            mvals = [v for k, v in mix.items() if k[0] == "xent" and k[1] == f and k[2] == s]
            g = go.get(("xent", f, s))
            if mvals and g is not None:
                diffs.append(st.median(mvals) - g)
        cells.append(st.median(diffs) if diffs else None)
    print(f"    {seed:<5}" + "".join((f"{c:>+9.4f}" if c is not None else f"{'--':>9}") for c in cells))

# ---------- logconf null (P4) ----------
print("\n" + "=" * 74)
print("P4 — LOGCONF raw median acc per seed (expect flat)")
print("=" * 74)
print("    seed " + "".join(f"{f:>8}" for f in FRACS))
for seed in SEEDS:
    c = curve_rawacc(seed, "logconf")
    print(f"    {seed:<5}" + "".join((f"{c[f]:>8.3f}" if c[f] is not None else f"{'--':>8}") for f in FRACS))
for seed in SEEDS:
    c = curve_rawacc(seed, "logconf")
    span = max(v for v in c.values() if v is not None) - min(v for v in c.values() if v is not None)
    print(f"    seed{seed}: full-range span across fractions = {span:.4f}")

# ---------- scale interaction zero-crossing (P5) ----------
print("\n" + "=" * 74)
print("P5 — SCALE INTERACTION: PGR zero-crossing fraction per valid pair (xent)")
print("=" * 74)
print("  (fraction at which per-pair PGR first turns positive; later = needs more GT)")
for seed in SEEDS:
    gt, mix, vp = PER[seed]["gt"], PER[seed]["mix"], PER[seed]["vp"]["xent"]
    print(f"  seed{seed}:")
    for s, w in sorted(vp, key=lambda p: (RANK[p[0]], RANK[p[1]])):
        gap = gt[("xent", s)] - gt[("xent", w)]
        cross = None
        for f in FRACS:
            if (("xent", f, s, w) in mix) and pgr(gt, "xent", s, w, mix[("xent", f, s, w)]) > 0:
                cross = f
                break
        cs = f"{cross}" if cross is not None else ">1.0"
        print(f"    {s.replace('gpt2','g'):8s}<-{w.replace('gpt2','g'):10s} GTgap={gap:+.3f}  crosses@ {cs}")
