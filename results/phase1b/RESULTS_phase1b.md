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
| C SciQ vs BoolQ | SciQ larger/earlier | **cleaner in PGR, smaller in raw acc; A replicates 18/18** | ◑ **testbed validated; findings replicate; no bigger effects** |

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

## Component C — SciQ validity ◑ (testbed validated; findings replicate; no larger effects)

144 SciQ xent runs on 8×H200 (instance 42244966, destroyed after pull): mixing curve
(10 pairs × {0.10,0.25,0.50,1.0} × 3 seeds = 120) + random_labels control (24). 144/144 ok.
SciQ GT ceilings are healthy (gpt2-large is *not* anomalous on SciQ; one minor seed-1 xl←large
inversion handled by the generic validity rule). Analysis: `results/phase1b/analyze_c.py`.

**C.1 — Fraction curve, SciQ vs BoolQ (xent):**

| frac | SciQ Δacc | SciQ PGR | BoolQ Δacc | BoolQ PGR |
|---|---|---|---|---|
| 0.10 | +0.003 | **+0.16** | +0.003 | −0.18 |
| 0.25 | +0.007 | +0.44 | +0.018 | +0.30 |
| 0.50 | +0.023 | +0.72 | +0.026 | +0.28 |
| 1.00 | +0.033 | +0.94 | +0.055 | +1.04 |

*(Δacc = median over all pairs×seeds vs frac0; PGR = median over valid strict pairs.)*
SciQ is **cleaner** — PGR is **positive and monotonic from 0.10** (BoolQ starts negative and
only crosses ~0.25) — confirming the phenomenon is real where baseline W2SG signal exists. But
in **raw accuracy SciQ moves *less*** (smaller headroom; its weak transfer already sits closer
to ceiling). So the plan's prediction ("larger") holds as *earlier/cleaner in PGR*, and is
*reversed* in raw acc. **No-knee, gradual, back-loaded shape replicates on both tasks.**

**C.2 — Component A replicates on SciQ (weak labels informative):**

| frac | naive mixing − random_labels | #pairs>0 | random_labels median acc |
|---|---|---|---|
| 0.10 | **+0.157** | 18/18 | 0.527 (≈chance) |
| 0.25 | **+0.109** | 18/18 | 0.569 |

Even stronger than BoolQ (+0.08): random labels collapse SciQ to ~chance, so real weak labels
add **11–16 points** at equal rows. The 3 near-0.5 runs flagged in QC were exactly these
random_labels@0.10 cases — the control behaving as designed, not failures.

**What Component C establishes:** the testbed is valid (SciQ has real positive-PGR W2SG signal),
and the two load-bearing Phase-1/1b findings — **no knee / gradual curve** and **weak labels are
informative** — **replicate across both tasks.** But SciQ offers *smaller* absolute effects, so
it is not a higher-signal venue for a Phase-2 strategy bake-off. Cross-task robustness is the
gain, not bigger effects.

---

## Phase 1b overall verdict (0 + A + B + C)

| Question | Finding | Strength |
|---|---|---|
| Can the testbed resolve strategy effects? (0) | yes, MDE 0.007 | PASS |
| Does *which* label matter? (A) | **yes — weak labels informative** | robust, **both tasks** (15/15, 18/18) |
| Does *where* you place GT matter? (B) | **no — oracle ≈ naive** | confident null (BoolQ) |
| Is BoolQ the right task? (C) | SciQ cleaner-PGR but smaller raw; findings replicate | validated; no bigger effects |

**Decision for Phase 2.** Allocation (Axis A) is dead, so Phase 2 is **focused, not a broad
bake-off**: a small, pre-registered **combination-method portfolio** (B/C/D — weighted loss,
soft-GT, GT-anchored logconf, teacher-reliability weighting, GT-as-early-stopping) at
{0.10, 0.25, 0.50}, framed as "interesting even if it fails," with most methods expected null.
This tests the one untested lever (*how* to combine), and the rubric explicitly rewards breadth
of plausible attempts with rigor. The defensible contribution regardless of outcome is the
**cross-task-replicated characterization** (*how much* / *which* / *where*) + the gpt2-large
instability + the logconf null. The **larger model gap** is the scientifically correct next
lever but is **out of scope** (the brief fixes the universe to GPT-2) → future work in the talk.
Plan: `../../plans/phase2.md`; execution spec: `../../plans/PHASE2_PROMPT.md`.
