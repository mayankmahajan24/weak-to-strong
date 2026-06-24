#!/usr/bin/env python3
"""5-seed phase-0 baseline analysis — GT ceilings, W2SG transfer, PGR across seeds 0-4.

Reads results/data/baseline/seed{0..4}/ (GT ceilings + frac=0 transfers, boolq+sciq, xent+logconf).
GT ceiling = xent GT (logconf GT is meaningless). Reports per (dataset, loss): median PGR over the
6 strict weak<strong pairs, per seed, then mean±std across the 5 seeds. Raw-accuracy first.
"""
import glob, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
SEEDS = [0, 1, 2, 3, 4]

gt = {}   # (model, ds, seed) -> acc  (xent GT only)
tr = {}   # (weak, strong, ds, loss, seed) -> acc
for s in SEEDS:
    for cfgp in glob.glob(str(ROOT / f"results/data/baseline/seed{s}/*/config.json")):
        c = json.load(open(cfgp))
        ds = c.get("ds_name"); model = c.get("model_size"); loss = c.get("loss", "xent")
        if ds is None or model is None:
            continue
        if c.get("gt_fraction", 0) not in (0, 0.0, None):
            continue
        try:
            acc = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
        except Exception:
            continue
        weak = c.get("weak_model_size")
        if not weak:
            if loss == "xent":
                gt[(model, ds, s)] = acc
        else:
            tr[(weak, model, ds, loss, s)] = acc

strict = [(s_, w) for s_ in ORDER for w in ORDER if RANK[w] < RANK[s_]]  # 6 weak<strong pairs

print("=" * 78)
print("GT CEILINGS (xent) — mean ± std across 5 seeds")
print("=" * 78)
print(f"  {'model':>12}  {'boolq':>22}  {'sciq':>22}")
for m in ORDER:
    cells = []
    for ds in ["boolq", "sciq"]:
        vals = [gt[(m, ds, s)] for s in SEEDS if (m, ds, s) in gt]
        cells.append(f"{st.mean(vals):.3f}±{st.pstdev(vals):.3f} (n{len(vals)})" if vals else "—")
    print(f"  {m:>12}  {cells[0]:>22}  {cells[1]:>22}")
# per-seed gpt2-large to re-examine the instability fresh
print("\n  gpt2-large boolq GT by seed: " +
      "  ".join(f"s{s}={gt[('gpt2-large','boolq',s)]:.3f}" for s in SEEDS if ('gpt2-large','boolq',s) in gt))

def pgr(weak, strong, ds, loss, s):
    if (weak, ds, s) not in gt or (strong, ds, s) not in gt:
        return None
    wg, sg = gt[(weak, ds, s)], gt[(strong, ds, s)]
    t = tr.get((weak, strong, ds, loss, s))
    if t is None or sg - wg <= 0:   # need a real capability gap
        return None
    return (t - wg) / (sg - wg), t, wg, sg

for excl in [False, True]:
    tag = "EXCLUDING (seed1, gpt2-large)" if excl else "ALL 5 seeds, ALL pairs"
    print("\n" + "=" * 78)
    print(f"W2SG CORE — median PGR over 6 strict pairs, per seed  [{tag}]")
    print("=" * 78)
    for ds in ["boolq", "sciq"]:
        for loss in ["xent", "logconf"]:
            per_seed_med = []
            raw_beats = []  # transfer - weak_gt, to check raw improvement over teacher
            for s in SEEDS:
                pgrs = []
                for strong, weak in strict:
                    if excl and s == 1 and (strong == "gpt2-large" or weak == "gpt2-large"):
                        continue
                    r = pgr(weak, strong, ds, loss, s)
                    if r:
                        pgrs.append(r[0]); raw_beats.append(r[1] - r[2])
                if pgrs:
                    per_seed_med.append(st.median(pgrs))
            if per_seed_med:
                m, sd = st.mean(per_seed_med), st.pstdev(per_seed_med)
                rb = st.mean(raw_beats)
                print(f"  {ds:6s} {loss:8s}: median PGR = {m:+.3f} ± {sd:.3f}  "
                      f"(per-seed medians: {[f'{x:+.2f}' for x in per_seed_med]});  "
                      f"mean raw transfer−weak_gt = {rb:+.3f}")
