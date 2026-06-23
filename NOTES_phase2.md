# Phase 2 — Pre-Registration (combination-method portfolio)

**Registered:** 2026-06-23 ~20:51 UTC, **before any Phase-2 GPU run.**
**Anchor commit:** `f3acd25` (harness M1–M5 implemented + unit-tested; driver written; **no
Phase-2 results exist yet**).
**Purpose:** lock the methods, success criteria, and per-method predictions before the sweep, so
Phase 2 is a real test rather than post-hoc storytelling. Predictions below are frozen; outcomes
get scored in `results/phase2/RESULTS_phase2.md` and any pipeline change is logged in "Deviations"
at the bottom. Companion to `plans/phase2.md` (design) and `plans/PHASE2_PROMPT.md` (exec spec).

## Scope (decided by Phase 1b)
Allocation (Axis A) is **dead** — the Phase-1b oracle (perfect error-targeting) tied random
placement (`oracle − naive` within noise at 0.10 & 0.25; MDE 0.0071). So we do **not** test
allocation heuristics. Phase 2 tests the one untested axis: **how to combine weak + GT per row.**
Scale-up is out of scope (brief: GPT-2 only). xent except where a method is intrinsically logconf.

## Frozen analysis decisions (carry from Phase 1 / 1b)
- BoolQ, GPT-2 family, **3 seeds**, `gt_seed = seed`; `EXCLUDE = {(1, "gpt2-large")}` (failed-ceiling rule).
- **Noise floor = 0.014.** An effect counts only if the 3-seed median over pairs exceeds it.
- **Raw accuracy is the primary readout**; PGR secondary/caveated.
- **Baseline = Phase-1 naive mixing** at the same (pair, fraction, seed) — reused, not re-run.
- Budgets **{0.10, 0.25, 0.50}**; **6 strict weak<strong pairs** (~18 (pair,seed) cells/fraction).
- Three success readouts per method: **beat-naive** (median Δacc(method−naive) > 0.014 at a
  fraction), **left-shift** (method@0.10 reaches naive@0.25, or @0.25 reaches naive@0.50 —
  sample efficiency), **ceiling-raise** (beats naive at 0.50 where naive already works).

## Methods & predictions (frozen, directional)

| # | Method | Mechanism | Prediction |
|---|---|---|---|
| M1 | weighted (λ=4) | upweight GT rows in the loss | **Small +** at 0.10 (amplifies scarce clean signal), **fading by 0.50**; risk of overfitting the GT subset at high λ (watch train-vs-test gap). |
| M2 | soft_gt (ε=0.1) | label-smooth GT-row targets | **≈ neutral / within noise** — mostly regularization; little new signal. |
| M3 | gt_anchored | logconf, GT rows exempt from the confidence blend | **Strongest bet (directional):** moves logconf from deeply negative **toward xent-like** — i.e., the Phase-1 logconf null is *fixable* by protecting clean labels. Predict gt_anchored ≫ naive-logconf, approaching naive-xent. |
| M4 | reliability | weight weak rows by P(teacher correct\|conf) fit on GT subset | **+ iff** weak errors are feature-predictable from a few hundred GT rows; **genuinely uncertain** — "interesting even if it fails." |
| M5 | gt_early_stop | train on weak only; GT subset = checkpoint-selection val | **Small +** at low cost (clean model selection without training on the budget); a sample-efficiency angle. |

**Honest overall prior:** given the near-zero baseline and the oracle null, **we expect most
methods to land within the noise floor; ≤1–2 to clear it. M3 is the most likely positive** (it has
a concrete mechanism tied to a known failure). **A largely-null portfolio is a valid, reportable
result** — the rubric rewards plausible attempts honestly scored, and it would sharpen the
conclusion that at GPT-2 scale, beyond label quantity and weak-label informativeness, neither
*where* nor *how* you combine moves the needle (→ scale is the binding constraint).

## Pre-sweep gates (must pass on the box before the full run)
1. **Naive reproduction** — `gpt2-medium←gpt2 @0.25, seed 1, xent ≈ 0.673` bit-for-bit (the
   plumbing must not have perturbed the naive path).
2. **M5 smoke** — one `gt_early_stop` run: 0 GT rows in training, a checkpoint is selected, sane acc.

## Decision gate → presentation
Carry any method(s) that clear the floor into the talk as positive results; present the rest as
honest negatives. Either way, Phase 2 **completes the three-question story** (how much / where /
how). If the whole portfolio is null, that is the clean, defensible conclusion + the scale-up
proposal as future work.

## Deviations (append-only)

_(none yet)_
