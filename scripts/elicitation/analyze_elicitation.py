#!/usr/bin/env python3
"""Aggregate elicitation runs -> EL1 scaling + PGR/sample-efficiency vs the noise floor (CPU).

Reads results/elicitation/runs/*.json (from run_elicitation.py) and the Phase-0 GT ceilings
(results/data/baseline/seed*/), then reports, per task:
  EL1  — elicited accuracy vs strong-model size (probe@k, full-supervised, CCS, random control).
  EL2/3 — elicitation PGR per strict (weak<strong) pair vs the weak baseline, and the k (label
          budget) where elicitation first clears the 0.014 floor — the sample-efficiency readout.
PGR convention matches the rest of the study: strict pairs, EXCLUDE {(1,gpt2-large)}, median over
(pair,seed). Run after the box produces the per-config jsons.
"""
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
EXCLUDE = {(1, "gpt2-large")}
FLOOR = 0.014
strict = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]


def load_ceilings():
    """(model, ds, seed) -> GT-ceiling accuracy (xent baseline, weak==strong)."""
    gt = {}
    for cfgp in glob.glob(str(ROOT / "results/data/baseline/seed*/*/config.json")):
        c = json.load(open(cfgp))
        if c.get("gt_fraction", 0) not in (0, 0.0, None) or c.get("loss", "xent") != "xent":
            continue
        if c.get("weak_model_size"):
            continue  # transfer run, not a ceiling
        try:
            a = json.load(open(cfgp.replace("config.json", "results_summary.json")))["accuracy"]
        except Exception:
            continue
        gt[(c["model_size"], c.get("ds_name"), c["seed"])] = a
    return gt


def load_runs():
    runs = {}
    for fn in glob.glob(str(ROOT / "results/elicitation/runs/*.json")):
        r = json.load(open(fn))
        runs[(r["ds"], r["model"], r["seed"])] = r
    return runs


def elicit_acc(runs, ds, model, seed, method, k=None):
    r = runs.get((ds, model, seed))
    if r is None:
        return None
    if method == "full":
        return r.get("m1_full_supervised")
    if method == "random":
        return r.get("control_random_direction")
    table = {"probe": "m1_probe", "ccs": "m2_ccs"}[method]
    d = r.get(table)
    return None if d is None else d.get(str(k), d.get(k))


def exc(seed, m):
    return (seed, m) in EXCLUDE


def main():
    gt = load_ceilings()
    runs = load_runs()
    if not runs:
        print("no elicitation runs found under results/elicitation/runs/ — run the box step first")
        return
    tasks = sorted({k[0] for k in runs})
    seeds = sorted({k[2] for k in runs})
    ks = sorted({int(kk) for r in runs.values() for kk in (r.get("m1_probe") or {})})

    for ds in tasks:
        print(f"\n===== {ds.upper()} =====")
        # EL1 — elicited accuracy vs model size
        print("EL1  elicited accuracy by strong model (median over seeds)")
        hdr = f"  {'model':<12}" + "".join(f"p@{k:<5}" for k in ks) + f"{'full':>7}{'ccs@32':>8}{'rand':>7}{'chance':>8}"
        print(hdr)
        for m in ORDER:
            row = f"  {m:<12}"
            for k in ks:
                v = [elicit_acc(runs, ds, m, s, "probe", k) for s in seeds]
                v = [x for x in v if x is not None]
                row += f"{(st.median(v) if v else float('nan')):<7.3f}"[:7]
            for label, meth, kk in [("full", "full", None), ("ccs", "ccs", 32), ("rand", "random", None)]:
                v = [elicit_acc(runs, ds, m, s, meth, kk) for s in seeds]
                v = [x for x in v if x is not None]
                row += f"{(st.median(v) if v else float('nan')):>8.3f}"
            ch = [runs[(ds, m, s)]["chance"] for s in seeds if (ds, m, s) in runs]
            row += f"{(st.median(ch) if ch else float('nan')):>8.3f}"
            print(row)

        # EL2/EL3 — elicitation PGR per strict pair (median), probe@k, vs the floor
        print("\nEL2/3  median elicitation PGR over strict pairs (probe@k); first k clearing the floor")
        for k in ks:
            pgrs, deltas = [], []
            for (sg, w) in strict:
                for s in seeds:
                    if exc(s, w) or exc(s, sg):
                        continue
                    acc = elicit_acc(runs, ds, sg, s, "probe", k)
                    wg, scg = gt.get((w, ds, s)), gt.get((sg, ds, s))
                    if acc is None or wg is None or scg is None or scg - wg <= 0:
                        continue
                    pgrs.append((acc - wg) / (scg - wg))
                    deltas.append(acc - wg)             # vs the weak teacher's own ceiling
            mp = f"{st.median(pgrs):+.3f}" if pgrs else "  n/a"
            md = f"{st.median(deltas):+.4f}" if deltas else "  n/a"
            clears = "yes" if deltas and st.median(deltas) > FLOOR else "no"
            print(f"  k={k:<4} medPGR={mp:>7}  medΔ(acc−weakGT)={md:>8}  >floor({FLOOR})={clears}  n={len(pgrs)}")


if __name__ == "__main__":
    main()
