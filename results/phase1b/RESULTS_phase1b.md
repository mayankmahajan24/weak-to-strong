# Phase 1b Results — Testbed & Premise Validation

Gate before Phase 2. Plan: `plans/phase1b.md`. xent-only, BoolQ (+ SciQ for Component C),
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
