#!/usr/bin/env python3
"""Mechanism-lite: characterize the weak teachers from the preserved weak_labels (no GPU).

Reads each weak teacher's BoolQ-xent weak_labels (gt_label, hard_label, soft_label) and reports,
per (seed, teacher):
  - error rate (hard != gt) and accuracy;
  - confidence calibration: accuracy within bins of weak confidence max(soft_label) — is a
    confident weak label actually more likely correct? (underpins M4 reliability weighting);
  - error overlap across teachers on the SAME seed (same transfer split): do bigger teachers fix
    the smaller teacher's errors, or repeat them? (context for the oracle-null and W2SG itself).

Needs pyarrow. Run with the scratch venv:  <venv>/bin/python scripts/phase2/weak_label_analysis.py
"""
import glob
import statistics as st
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]


def load_arrow(path):
    try:
        tbl = pa.ipc.open_stream(pa.memory_map(path, "r")).read_all()
    except Exception:
        tbl = pa.ipc.open_file(pa.memory_map(path, "r")).read_all()
    return tbl.column("gt_label").to_pylist(), tbl.column("hard_label").to_pylist(), \
        tbl.column("soft_label").to_pylist()


def find_wl(seed, model):
    pat = str(ROOT / f"results/data/baseline/seed{seed}/*ms={model}-nd*/weak_labels/data-00000-of-00001.arrow")
    for p in glob.glob(pat):
        if "dn=boolq" in p and "l=xent" in p and "wms=" not in p:
            return p
    return None


teachers = {}  # (seed, model) -> (gt, hard, conf)
for seed in [0, 1, 2]:
    for m in ORDER:
        p = find_wl(seed, m)
        if p:
            gt, hard, soft = load_arrow(p)
            conf = [max(s) for s in soft]
            teachers[(seed, m)] = (gt, hard, conf)

print("=" * 74)
print("WEAK-TEACHER ERROR RATES (BoolQ, per seed)")
print("=" * 74)
print(f"  {'model':>12} " + "  ".join(f"seed{s}" for s in [0, 1, 2]) + "   mean")
for m in ORDER:
    rates = []
    for s in [0, 1, 2]:
        if (s, m) in teachers:
            gt, hard, _ = teachers[(s, m)]
            rates.append(sum(g != h for g, h in zip(gt, hard)) / len(gt))
    if rates:
        print(f"  {m:>12} " + "  ".join(f"{r:.3f}" for r in rates) + f"   {st.mean(rates):.3f}")

print("\n" + "=" * 74)
print("CONFIDENCE CALIBRATION — is a confident weak label more accurate? (seed 0)")
print("=" * 74)
print("  Accuracy within weak-confidence bins (underpins M4 reliability weighting)")
bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
for m in ORDER:
    if (0, m) not in teachers:
        continue
    gt, hard, conf = teachers[(0, m)]
    row = []
    for lo, hi in bins:
        idx = [i for i in range(len(gt)) if lo <= conf[i] < hi]
        if idx:
            acc = sum(gt[i] == hard[i] for i in idx) / len(idx)
            row.append(f"{acc:.2f}(n{len(idx)})")
        else:
            row.append("  -   ")
    print(f"  {m:>12}  " + " ".join(f"{c:>10}" for c in row))
print("  bins:        " + " ".join(f"{f'[{lo:.1f},{hi:.1f})':>10}" for lo, hi in bins))

print("\n" + "=" * 74)
print("ERROR OVERLAP across teachers (seed 0, same transfer split)")
print("=" * 74)
print("  P(bigger wrong | smaller wrong): do larger teachers REPEAT the smaller's errors?")
for i, ms in enumerate(ORDER):
    for mb in ORDER[i + 1:]:
        if (0, ms) in teachers and (0, mb) in teachers:
            gs, hs, _ = teachers[(0, ms)]
            gb, hb, _ = teachers[(0, mb)]
            small_wrong = [k for k in range(len(gs)) if gs[k] != hs[k]]
            if small_wrong:
                both = sum(gb[k] != hb[k] for k in small_wrong) / len(small_wrong)
                print(f"  {ms:>12} wrong -> {mb:>12} also wrong: {both:.2f}  "
                      f"(smaller err-rate {len(small_wrong)/len(gs):.2f})")
