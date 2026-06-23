# Phase 1b — Testbed & Premise Validation (gate before Phase 2)

**Status:** pre-registered plan. Written 2026-06-23, after the complete 3-seed Phase 1
(`results/phase1/RESULTS_phase1.md`). **No Phase 2 strategy work begins until Phase 1b
clears its gates.**

## Why this phase exists

Phase 1 (BoolQ, GPT-2, 3 seeds) produced **modest, confounded, or null** results on a
testbed with **near-zero baseline W2S signal**:
- Naive GT-mixing helps xent but is back-loaded (nothing ≤10%); **no knee** (P1 refuted).
- Mixing ≫ GT-only is the strongest effect **but confounded by training-set size** — not
  yet evidence that weak labels are *informative*.
- logconf inert (clean null); scale interaction underpowered (no claim).

Before building elaborate Phase 2 allocation/combination strategies, a rigorous researcher
must establish that **the instrument can detect the effects we'd be tuning** and that the
**premises hold**. Phase 1b answers three gating questions, each with a pre-registered kill
criterion, plus a power check. The intended output may legitimately be *"Phase 2 is not
testable at this scale; here is what would be required."*

## Fixed universe

- Models: GPT-2 family (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).
- Loss: **xent only** (logconf dropped per Phase-1 decision gate — inert at every budget).
- Seeds: 0, 1, 2; `--gt_seed = --seed`.
- Analysis exclusion (pre-stated, carried from Phase 1): `EXCLUDE = {(1, "gpt2-large")}`
  (failed-ceiling filter; reproduced by S5 regen → real bad-optimization outcome, not corruption).
- Noise floor (from Phase 1 M5, BoolQ xent, exclusion applied): **0.014** (median per-pair
  3-seed accuracy range). Every effect below is read against this.
- Reuse preserved baseline `weak_labels/` as `--weak_labels_path` (Phase-1 convention). Save
  slim results only (no `results.pkl`/`*.bin`). Clean sharded `pytorch_model-*.bin` between runs.

---

## Code changes required (small, localized)

All in the existing mixing seam. **Do not touch losses, model code, or training dynamics.**

1. **`weak_to_strong/label_mixing.py`** — add two strategies alongside `"naive"`:
   - `"oracle"`: select the GT rows as those the weak teacher got **wrong**
     (`gt_label != hard_label`). If #wrong > k, take a `gt_seed`-seeded sample of k wrong rows;
     if #wrong < k, take all wrong + fill remainder with a seeded random sample of the rest.
   - `"random_labels"`: select the k GT rows exactly as `"naive"` (same `gt_seed` sample), but
     for the **non-GT** rows replace `soft_label`/`hard_label` with a `gt_seed`-seeded random
     Bernoulli(0.5) label (so the non-GT rows carry the **same row/step count as mixing but
     zero information**). GT rows are real GT, identical to naive.
   - (optional) `"anti_oracle"`: GT on weak-**correct** rows (lower-bound sanity).
2. **`train_simple.py`** — add `mixing_strategy: str = "naive"` to `main()`, pass it into
   `apply_label_mixing(...)`, and when `strategy != "naive"` add `"mixing_strategy"` to the
   `config` dict (so `get_config_foldername` emits a distinct token and runs don't collide).
   Suggested token: `mstr={strategy}`.
3. Verify the new token appears in folder names; confirm `gt_fraction=0.0` path is untouched.

Validation of the code itself (cheap, before the sweep):
- `oracle` at gt_fraction = (#wrong / n) should select exactly the wrong set.
- `random_labels` with the SAME `gt_seed` as a naive run must place GT on the **identical**
  rows (diff only in the non-GT labels).
- A `naive` run re-run through the new code must reproduce the existing Phase-1 number
  bit-for-bit (no behavioral change to the default path).

---

## Component 0 — Power / minimum-detectable-effect (LOCAL, no GPU). Do first.

Using `results/phase1/phase1_results.csv` (xent, exclusion applied): bootstrap the
per-`(pair,seed)` accuracy distribution at `gt_fraction=0` to get the sampling distribution
of the Phase-1 estimator (median Δacc over pairs, across 3 seeds). Report the **MDE** at ~80%
power for a *strategy-vs-naive* contrast.

- **Pre-reg prediction:** MDE ≈ 0.01–0.02 acc (given floor 0.014, ~10 pairs, 3 seeds).
- **Gate:** if MDE ≳ the naive effect sizes themselves (i.e., we cannot resolve a difference
  smaller than the effect we already measured), Phase 2 strategy comparisons are **underpowered
  by construction** → flag loudly; more seeds are required before any strategy claim.

---

## Component A — De-confound Result 2 (is mixing > GT-only real?)

**Question:** is the mixing-over-GT-only advantage due to weak-label *information*, or just
*more training rows*?

**Runs (xent, BoolQ):** `random_labels` control at `gt_fraction ∈ {0.10, 0.25}`, for the 4
strong models × 3 seeds = **24 runs**. (Random-label accuracy is independent of weak source,
like gt_only, so vary only the strong model.) Mixing and GT-only already exist from Phase 1.

**Comparison (per strong model, per fraction, median over valid (pair,seed)):**
`acc(mixing) − acc(random_labels)`.

- **Pre-reg prediction:** mixing > random_labels by > noise floor at both fractions (weak
  labels carry real information beyond filling rows).
- **Verdicts:**
  - `mixing − random_labels > 0.014` → **Result 2 survives**: weak labels are informative. ✅
  - `mixing ≈ random_labels` (within noise) → **Result 2 was a data-quantity artifact**: weak
    labels add nothing beyond row count. Major reframe (see decision tree). ❌

---

## Component B — Oracle-allocation ceiling (does targeting GT have headroom?)

**Question:** at a small budget where naive does nothing, can the *best possible* allocation
(GT on exactly the examples the weak teacher gets wrong) beat naive?

**Runs (xent, BoolQ):** `oracle` at `gt_fraction ∈ {0.10, 0.25}`, 10 transfer pairs × 3 seeds
= **60 runs**. (Oracle depends on the weak teacher's errors → per-pair.) Naive already exists.
Optional `anti_oracle` (+60) as a lower bound.

**Comparison (per pair, per fraction, median over valid (pair,seed)):**
`acc(oracle) − acc(naive)`.

- **Pre-reg prediction:** oracle > naive at 0.10 (allocation matters most where budget is
  scarce); the gap shrinks by 0.25.
- **Verdicts:**
  - `oracle − naive > 0.014` → **allocation has headroom** → Phase 2 allocation strategies are
    justified (they approximate the oracle). ✅
  - `oracle ≈ naive` → **even perfect allocation does nothing** at this budget/scale → Phase 2
    allocation is a null here; do not build allocation strategies. ❌

---

## Component C — Testbed validity: SciQ naive curve (are we on the right task?)

**Question:** BoolQ has near-zero baseline W2S PGR; SciQ baseline xent PGR is **+0.17** (Phase
0). Is GT-mixing a larger/cleaner effect where the underlying phenomenon actually exists?

**Runs (xent, SciQ):** naive mixing sweep at `gt_fraction ∈ {0.10, 0.25, 0.50, 1.0}`, 10 pairs
× 3 seeds = **120 runs**; plus gt_only at the same fractions (4 strong × 3 seeds × 4 = 48) for
the mixing-vs-GT-only contrast on SciQ. SciQ baseline + weak_labels already exist from Phase 0.
(Full 6-fraction curve optional; the 4 key fractions are sufficient to compare shape/magnitude.)

- **Pre-reg prediction:** SciQ shows a larger and earlier-rising mixing effect than BoolQ
  (more baseline signal to recover).
- **Verdicts:**
  - SciQ effect clearly larger/cleaner → **BoolQ was a low-signal testbed**; make SciQ primary
    for Phase 2.
  - SciQ ≈ BoolQ (modest/null) → the small-scale null is **task-general** → the limiting factor
    is model scale, not task; escalate scale or report the bounded null.

---

## Run summary

| Component | Runs (xent) | Compute (8× H100/H200) | Gates |
|---|---|---|---|
| 0 Power | 0 (local) | — | MDE vs effect size |
| A De-confound | 24 | ~15 min | mixing vs random_labels |
| B Oracle ceiling | 60 (+60 opt) | ~30–60 min | oracle vs naive |
| C SciQ validity | 168 | ~2–2.5 h | SciQ vs BoolQ |
| **Total** | **~252** | **~3–3.5 h, ~$70–110** | |

Suggested order: **0 → (A + B in parallel) → C**. A+B are the premise gates and are cheap;
C is the larger generality check. (C can run concurrently if budget allows, since it answers
an orthogonal question.)

## Verification gates (per Phase-1 conventions)
- `mixing_strategy` token present in all new folder names; no collisions with Phase-1 runs.
- `naive` reproduction check passes (default path unchanged).
- 0 NaN / 0 degenerate (acc≈0.5) runs; `gt_fraction_actual ≈ requested` for mixing/random_labels
  (oracle/gt_only are exempt — actual reflects the selected subset).
- Consolidate into a Phase-1b CSV; re-use `EXCLUDE` and the 0.014 noise floor in all analysis.

## Pre-registered scoring table (fill after runs)

| Test | Prediction | Result | Verdict |
|---|---|---|---|
| 0 Power (MDE) | 0.01–0.02 | | adequate / underpowered |
| A mixing − random_labels | > floor | | informative / artifact |
| B oracle − naive @0.10 | > floor | | headroom / null |
| C SciQ vs BoolQ | SciQ larger | | switch task / task-general null |
