# Phase 1b Results — Testbed & Premise Validation

Gate before Phase 2. Narrative spine: [`../RESEARCH_PATH.md`](../RESEARCH_PATH.md).
Plan: `plans/phase1b.md`. xent-only, BoolQ (+ SciQ for Component C),
3 seeds, `EXCLUDE = {(1, "gpt2-large")}`, noise floor 0.014.

Scoring table (filled as components complete):

| Test | Prediction | Result | Verdict |
|---|---|---|---|
| 0 Power (MDE) | 0.01–0.02 | **MDE₁ = 0.0071** | **PASS — adequate** |
| A mixing − random_labels | > floor | _pending_ | _pending_ |
| B oracle − naive @0.10 | > floor | _pending_ | _pending_ |
| C SciQ vs BoolQ | SciQ larger | _pending_ | _pending_ |

---

## Component 0 — Power / minimum-detectable-effect (local, no GPU) ✅ PASS

`results/phase1b/component0_power.py`. Estimates the noise of the paired per-(pair,seed)
contrast (`cond − naive` at the same pair/seed/fraction) that Components A/B will use, then
the MDE at 80% power.

| Quantity | Value |
|---|---|
| Valid strict-pair contrast cells (N = pairs × seeds, exclusion applied) | 14 |
| σ_run (baseline across-seed, pooled) | 0.0089 |
| σ_d direct (SD of Δacc@0.01 cells; true effect ≈ nil there) — **primary** | 0.0106 |
| σ_d conservative (√2·σ_run, treats runs as independent) | 0.0125 |
| SE of aggregate (σ_d/√N) | 0.0028 |
| **MDE, 1-sided, 80% power** | **0.0071 acc** |
| MDE, 2-sided | 0.0080 acc |

Realized naive effects for scale: Δacc = +0.003 (0.10), +0.024 (0.25), +0.030 (0.50),
+0.075 (1.0).

**Verdict: PASS.** The aggregate contrast resolves effects ≥ ~0.007 acc — below the 0.02
gate and below the naive effect itself. A real aggregate effect of ≳0.01 in Components A/B
will be detectable; a null there would be a genuine null, not an artifact of low power.

**Caveats (pre-stated).** (1) This MDE is for the *aggregate* over 14 cells; a single-pair
effect needs ~3× more (~0.025) to be detectable — we claim aggregate, not per-pair, effects.
(2) It is an *upper bound*: oracle/naive share split+init, so their true paired noise is
likely below σ_d. (3) Assumes a roughly uniform shift; a strategy that helps only a few pairs
could wash out in the aggregate — we will also report the per-cell sign rate, not just the median.

**→ Informed:** the gate clears, so a null in A/B is meaningful. Proceed to implement + run A/B.

## Harness readiness for Components A & B ✅ (code, local, no GPU)

New allocation strategies added to `weak_to_strong/label_mixing.py` and wired through
`train_simple.py` via `--mixing_strategy` (folder/summary tagged only when ≠ `naive`, so
Phase-1 `naive` runs stay byte-identical):
- `oracle` — GT on the weak-teacher's wrong rows first, then fill (Component B).
- `random_labels` — GT on the same rows as naive, non-GT rows get Bernoulli(0.5) labels
  (Component A; row/step count held equal to mixing).

Unit-tested in `results/phase1b/test_label_mixing.py` — **17/17 pass** (synthetic edge cases
+ a real preserved weak_labels arrow). Pre-run fact from the real data (gpt2→ BoolQ, seed 0):
weak error rate **0.346**; **oracle@10% achieves 100% error-coverage vs naive's ~33%** — the
allocation lever is large; Component B will test whether it converts to accuracy.

**Runs still pending** (need a GPU): A (24 runs), B (60 runs) on BoolQ; C (168) on SciQ.
