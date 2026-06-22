# Phase 1 — Seed 1 results capture

Captured 2026-06-22 from Vast.ai instance `42131402` (8× H200, ssh5.vast.ai:11402).
Only the lightweight per-run artifacts were pulled (`config.json`, `log.jsonl`,
`results_summary.json`); the 8 MB-per-run `results.pkl` prediction dumps were **not**
transferred (not needed for numbers/plotting/further phases).

## What was pulled from the instance (seed 1, BoolQ, gt_seed=1)

| Condition | Fractions | Runs each | Source |
|-----------|-----------|-----------|--------|
| `naive_mixing` | 0.05, 0.10, 0.25, 0.50, 1.0 | 20 (10 pairs × 2 losses) | instance pulled this run: 050, 100 new; 025 overlap |
| `gt_only` | 0.01, 0.05, 0.10, 0.25, 0.50, 1.0 | 8 (4 models × 2 losses) | instance (all new) |

`naive_mixing/{001,005,010}` and `025` were already local from the prior Phase 0/early
Phase 1 session (those dirs retain their local `results.pkl`).

## Consolidated table

`phase1_seed1_results.csv` — one row per run, 196 rows total:
- `baseline` frac=0.0: 28 runs (GT ceilings where strong==weak, plus pure-weak transfer)
- `mixing`   frac∈{0.01,0.05,0.10,0.25,0.50,1.0}: 20 each
- `gt_only`  frac∈{0.01,0.05,0.10,0.25,0.50,1.0}: 8 each

Columns: condition, ds_name, loss, strong_model, weak_model, gt_fraction_requested,
gt_fraction_actual, gt_seed, seed, lr, accuracy, run_dir.

Regenerate with: `python3 results/phase1/consolidate_phase1.py`
(filters to BoolQ, seed=1, gt_seed=1 — drops old Phase 0 gt_seed=42 artifacts).

## Quick read (xent mixing, median over 10 pairs)

| frac | median acc | min | max |
|------|-----------|-----|-----|
| 0.01 | 0.657 | 0.632 | 0.743 |
| 0.05 | 0.662 | 0.626 | 0.737 |
| 0.10 | 0.664 | 0.623 | 0.734 |
| 0.25 | 0.693 | 0.628 | 0.749 |
| 0.50 | 0.683 | 0.627 | 0.735 |
| 1.00 | 0.743 | 0.675 | 0.767 |

Monotonic-increasing with diminishing returns through 0.25, consistent with the
pre-registered concave-curve prediction (the 0.50 dip is within the per-pair spread).

## Status vs plan milestones

- M1 (GT weak_labels regen for seeds 0/2): not done — still seed-1 only.
- M2 (seed-1 remaining fractions): **complete** — all mixing fractions present.
- M4 (GT-only controls): **complete for seed 1** — all 6 fractions present.
- M3 (seeds 0 & 2 sweep), M5/M6 (noise floor, plots): pending (need seeds 0/2 + analysis).

## Instance note

Instance `42131402` was still **running** (no active jobs) at capture time. It holds the
`results.pkl` files if ever needed. Destroy it to stop billing once you're confident the
capture is complete.
