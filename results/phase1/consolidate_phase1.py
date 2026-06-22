#!/usr/bin/env python3
"""Consolidate Phase 1 seed-1 run results into a single CSV.

Walks the per-run directories (each holding config.json + results_summary.json),
extracts the key numbers needed for plotting / PGR / further phases, and writes a
flat table. The heavy results.pkl files are intentionally ignored.

Conditions captured:
  - baseline  : gt_fraction=0 (ms==wms -> GT ceiling; ms!=wms -> pure-weak transfer)
  - mixing    : naive_mixing/* (weak labels + gt_fraction GT, gt_seed=1)
  - gt_only   : gt_only/*       (gt_fraction GT only, weak labels discarded)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA = ROOT / "results" / "data"
OUT = Path(__file__).resolve().parent / "phase1_seed1_results.csv"

# (glob root, condition label, is_gt_only)
SOURCES = [
    (DATA / "naive_mixing", "mixing", False),
    (DATA / "gt_only", "gt_only", True),
    (DATA / "baseline" / "seed1", "baseline", False),
]

FIELDS = [
    "condition", "ds_name", "loss", "strong_model", "weak_model",
    "gt_fraction_requested", "gt_fraction_actual", "gt_seed", "seed",
    "lr", "accuracy", "run_dir",
]


def load_run(run_dir: Path):
    cfg_p = run_dir / "config.json"
    sum_p = run_dir / "results_summary.json"
    if not cfg_p.exists() or not sum_p.exists():
        return None
    cfg = json.loads(cfg_p.read_text())
    summ = json.loads(sum_p.read_text())
    return cfg, summ


rows = []
for src, condition, _ in SOURCES:
    if not src.exists():
        continue
    for sum_p in src.rglob("results_summary.json"):
        run_dir = sum_p.parent
        loaded = load_run(run_dir)
        if loaded is None:
            continue
        cfg, summ = loaded
        if cfg.get("ds_name") != "boolq":
            continue  # Phase 1 is BoolQ only
        if cfg.get("seed") != 1:
            continue  # this consolidation is seed 1
        # Canonical Phase 1 selection: gt_seed must equal seed (=1). Baseline runs
        # have no gt_seed (None) and are kept. This drops old Phase 0 gt_seed=42 artifacts.
        if condition != "baseline" and cfg.get("gt_seed") != 1:
            continue
        rows.append({
            "condition": condition,
            "ds_name": cfg.get("ds_name"),
            "loss": cfg.get("loss"),
            "strong_model": cfg.get("model_size"),
            "weak_model": cfg.get("weak_model_size"),
            "gt_fraction_requested": summ.get("gt_fraction_requested", cfg.get("gt_fraction") or 0.0),
            "gt_fraction_actual": summ.get("gt_fraction_actual", cfg.get("gt_fraction") or 0.0),
            "gt_seed": cfg.get("gt_seed"),
            "seed": cfg.get("seed"),
            "lr": cfg.get("lr"),
            "accuracy": summ.get("accuracy"),
            "run_dir": str(run_dir.relative_to(ROOT)),
        })

rows.sort(key=lambda r: (r["condition"], str(r["gt_fraction_requested"]), r["loss"],
                         str(r["strong_model"]), str(r["weak_model"])))

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)

# console summary
by_cond = {}
for r in rows:
    key = (r["condition"], r["gt_fraction_requested"])
    by_cond[key] = by_cond.get(key, 0) + 1
print(f"Wrote {len(rows)} rows -> {OUT.relative_to(ROOT)}")
for (cond, frac), n in sorted(by_cond.items(), key=lambda x: (x[0][0], str(x[0][1]))):
    print(f"  {cond:10s} frac={frac}: {n} runs")
