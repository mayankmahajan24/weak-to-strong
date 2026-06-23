# Phase 1b Results — Testbed & Premise Validation

Gate before Phase 2. Narrative spine: [`../RESEARCH_PATH.md`](../RESEARCH_PATH.md).
Plan: `plans/phase1b.md`. xent-only, BoolQ (+ SciQ for Component C),
3 seeds, `EXCLUDE = {(1, "gpt2-large")}`, noise floor 0.014.

Scoring table (filled as components complete):

| Test | Prediction | Result | Verdict |
|---|---|---|---|
| 0 Power (MDE) | 0.01–0.02 | **MDE₁ = 0.0071** | **PASS — adequate** |
| A mixing − random_labels | > floor | **+0.079 / +0.081, 15/15 pairs** | ✅ **weak labels INFORMATIVE** (confound resolved) |
| B oracle − naive @{0.10,0.25} | > floor | **+0.0006 / −0.0049, ~coin-flip** | ❌ **allocation NULL** (no headroom) |
| C SciQ vs BoolQ | SciQ larger | _pending_ | _pending_ |

**Phase 1b read (A+B, BoolQ):** *Which* labels matters; *where* you place the GT does not.
Weak labels carry real information (de-confounds the Phase-1 mixing>GT-only headline), but even
a perfect error-targeting oracle gives no aggregate gain over random allocation at 10–25% GT —
so Phase-2 **Axis A (allocation: uncertainty/disagreement/diversity) is killed** at this scale.
Phase 2, if pursued, should pivot to combination/loss/reliability axes (B/C/D), and Component C
(SciQ) still gates whether BoolQ was the right task.

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

Runs executed on 8×H200 (instance 42235643, destroyed after pull): **60/60 ok, 0 failed,
0 NaN/degenerate**. Analysis: `results/phase1b/analyze_ab.py`.

## Component A — Weak-label informativeness (de-confound) ✅ INFORMATIVE

`naive mixing − random_labels` at the same strong model, same row + step count (random_labels
replaces the non-GT weak labels with Bernoulli(0.5) noise), xent, exclusion applied:

| frac | gpt2-medium | gpt2-large | gpt2-xl | ALL (median) | #pairs>0 |
|---|---|---|---|---|---|
| 0.10 | +0.116 | +0.082 | +0.072 | **+0.079** | 15/15 |
| 0.25 | +0.087 | +0.058 | +0.085 | **+0.081** | 15/15 |

~8 accuracy points, **unanimous across all 15 (pair,seed) cells, both fractions** — ~11× the
noise floor. Ordering (median acc): `random_labels 0.61 < gt_only 0.62 < naive_mix 0.67–0.69`.
The noise labels actually *hurt* relative to data-starved GT-only, so mixing's win over GT-only
is **real weak-label information, not a training-set-size artifact.** This **resolves the
Phase-1 Result-2 confound in favor of "weak labels are informative,"** and upholds P3 cleanly.

## Component B — Oracle-allocation ceiling ❌ NULL (no headroom)

`oracle − naive` at the same (pair, frac, seed) — oracle spends the GT budget on the weak
teacher's wrong rows (verified 100% error-coverage); naive places it at random:

| frac | median Δ | mean Δ | #pairs>0 / n | vs floor (0.014) / MDE (0.007) |
|---|---|---|---|---|
| 0.10 | +0.0006 | +0.0009 | 8/15 | within noise |
| 0.25 | −0.0049 | −0.0047 | 5/15 | within noise |

Even the **upper bound** on error-targeting allocation gives no aggregate gain over random
placement, at either budget, with a coin-flip sign. Component 0 established we can resolve
≥0.0071, so this is a **confident null, not underpowered.**

**Implication:** *where* the GT budget is spent does not matter at GPT-2/BoolQ scale — so
Phase-2 **Axis A (allocation heuristics: uncertainty / disagreement / diversity) is killed**,
since they all approximate the error-targeting the oracle already bounds at zero. (Caveat:
oracle bounds *error-targeting* allocation specifically; a coverage/diversity axis orthogonal
to weak-error location is not directly bounded, but it is the weaker motivation.)

## What Phase 1b establishes (A+B)

1. **Which labels matters** — weak supervision is genuinely informative (A), and the
   mixing>GT-only headline is now de-confounded.
2. **Where you place GT does not** — allocation has no ceiling to chase here (B).
3. Phase 2 should therefore drop allocation (Axis A) and, if pursued, test **combination /
   loss-dynamics / reliability** axes (B/C/D) — pending Component C (SciQ) to confirm the
   testbed. The "how much" (Phase 1 fraction curve) + "which labels" (A) results are the
   defensible core; "where" is a clean negative result.

**Component C — SciQ validity: still pending** (the testbed-generality check).
