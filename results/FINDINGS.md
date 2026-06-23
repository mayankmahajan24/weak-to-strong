# Ground-Truth Mixing in Weak-to-Strong Generalization — Findings

*Synthesis writeup. Detailed per-phase numbers + figures: `phase0/RESULTS_phase0.md`,
`phase1/RESULTS_phase1.md`, `phase1b/RESULTS_phase1b.md`; decision log:
`RESEARCH_PATH.md`; methods/pre-registration: `../NOTES_phase0.md`, `../NOTES_phase1.md`.*

---

## Thesis & headline

Treat strong/ground-truth (GT) supervision as a **scarce budget** mixed into a weak teacher's
labels, and ask three questions: **how much** is needed, **where** to spend it, **how** to
combine it. On the GPT-2 family (BoolQ primary, SciQ replication), 3 seeds, raw-accuracy-first
with a measured noise floor and pre-registered predictions:

> **The value of scarce supervision at this scale comes from *quantity* and from the
> *informativeness of the weak labels themselves* — not from *where* you place the GT.**
> GT mixing helps cross-entropy transfer, but the gain is **gradual and back-loaded (no knee)**;
> **weak labels are genuinely informative** (a clean control rules out a mere data-quantity
> effect); and a **perfect error-targeting oracle does no better than random allocation**.
> Effects are modest and consistent across two tasks; the confidence-loss variant is inert.

This is a deliberately honest result set: of five pre-registered Phase-1 predictions, **one was
refuted, one unsupported, three held**; the single most exciting hypothesis (a "knee" at small
GT budgets, and supervision value scaling with model size) **did not survive**.

---

## Setup

- **Models:** GPT-2 family (124M → 1.5B; `gpt2`, `-medium`, `-large`, `-xl`). Fork of
  `openai/weak-to-strong`. Transfer = strong student trained on a weak teacher's labels over the
  *transfer split* (data the weak model never trained on; asserted in code).
- **Tasks:** BoolQ (primary), SciQ (replication). Both binary; chance = 0.5.
- **Conditions:** `gt_fraction ∈ {0.01…1.0}` of weak labels replaced by GT; mixing vs GT-only;
  xent vs logconf; 3 seeds (`gt_seed = seed`, reseeded per the seed-environment table).
- **Metrics:** raw test accuracy (primary, stable); PGR `(transfer − weak)/(strong_GT − weak)`
  (secondary, caveated — small adjacent-pair denominators inflate it). Aggregated median over
  pairs then seeds; every effect read against a **noise floor = 0.014** (3-seed per-pair range).
- **Baseline W2SG signal is weak at this scale:** BoolQ xent median PGR ≈ **−0.31**, SciQ ≈
  **+0.17** (Phase 0). This near-zero floor is the backdrop for every intervention below.

---

## Q1 — How much supervision? (the fraction curve)

**Finding: monotonic but back-loaded — there is no knee. A small GT budget does essentially
nothing; you need ≥25–50% before a reliable gain.**

BoolQ, xent, median Δaccuracy vs the pure-weak baseline (seed-1 `gpt2-large` excluded, see
Data-integrity):

| gt_fraction | 0.01 | 0.05 | 0.10 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|
| median Δacc | +0.001 | +0.004 | +0.003 | **+0.024** | +0.030 | +0.075 |
| vs noise floor | nil | within | within | ~1.7× | ~2×, unanimous | large |

- ≤0.10 is within the noise floor; the effect becomes detectable at 0.25, robust at 0.50, and
  is **largest at 1.0**. Marginal return per unit GT is *not* front-loaded.
- **Pre-registered prediction P1 (concave / diminishing-returns knee at 0.10–0.25) → REFUTED.**
  The curve is convex/back-loaded; the "knee at 0.25" an earlier single-seed draft claimed was a
  PGR zero-crossing artifact, erased once a second seed + a per-pair re-analysis were applied.
- **Replicates on SciQ** (Component C): same gradual no-knee shape; cleaner in PGR there
  (positive & monotonic from 0.10: +0.16 → +0.94) but *smaller in raw accuracy* (less headroom).

*Figure: `plots/phase1_fraction_curve.png` (+ `_by_seed`).*

---

## Q2 — Where to allocate it? (the allocation question)

**Finding: it doesn't matter. A perfect oracle that spends the GT budget exactly on the weak
teacher's errors does no better than random placement.**

BoolQ, xent, `oracle − naive` at the same (pair, fraction, seed):

| gt_fraction | median Δ | sign rate | verdict |
|---|---|---|---|
| 0.10 | +0.0006 | 8/15 | within noise |
| 0.25 | −0.0049 | 5/15 | within noise |

- The oracle had **100% error-coverage** (verified) vs naive's ~33%, yet produced no aggregate
  accuracy gain. A power analysis established we could resolve an aggregate effect ≥ **0.0071**
  (80% power), so this is a **confident null, not underpowered**.
- **Consequence:** allocation heuristics (uncertainty / disagreement / diversity — the planned
  "Axis A") all approximate the error-targeting the oracle already bounds at zero, so **they are
  not worth running at this scale.** This is the cleanest negative result in the study.

---

## Q3 — How to combine weak + GT? (informativeness & the loss)

**Finding: weak labels are genuinely informative — the mixing advantage is real, not a
data-quantity artifact — and the confidence loss (logconf) cannot use GT at all.**

- **Mixing > GT-only at every budget < 1.0** (BoolQ: +0.05–0.07). In Phase 1 this was *confounded*
  by training-set size (GT-only is data-starved at low fractions). **Phase 1b Component A
  de-confounds it** with a `random_labels` control (same rows + steps, non-GT labels randomized):

  | task | mixing − random_labels @0.10 | @0.25 | pairs positive |
  |---|---|---|---|
  | BoolQ | +0.079 | +0.081 | 15/15 |
  | SciQ | +0.157 | +0.109 | 18/18 |

  Ordering: `random_labels < gt_only < naive_mixing` — noise *hurts* relative to data-starved
  GT-only, so mixing's win is **real weak-label information.** Cross-task, unanimous, ~6–11× the
  noise floor. (P3 upheld and now clean.)
- **logconf is inert at every budget, including 100% GT** (Δacc ≈ 0 across fractions; clean
  null). Mechanism (`weak_to_strong/loss.py`): the confidence term blends the target with the
  model's own hardened predictions (`target = labels·(1−c) + strong_preds·c`, `c→0.5`), so even
  all-GT targets are ~50% self-prediction — the GT signal is structurally diluted. (P2 held;
  logconf dropped from later phases.)

---

## Secondary findings

- **Scale interaction: inconclusive (no claim).** ΔPGR/Δfraction is positive at all sizes but
  non-monotonic and underpowered; `gpt2-large` is 2-seed after the exclusion below.
  **Pre-registered P4 (value of GT grows with student size) → not supported.** Likely structural:
  the GPT-2 family spans ~12×, vs the ~115× where the original paper saw scale effects.
- **`gpt2-large` GT is optimization-unstable.** An 8-seed study (S7) found **27% of seeds** land
  in a low mode (GT acc < gpt2-medium's); seed-1's 0.662 is a representative draw, not an outlier
  (reproduced bit-for-bit). It is excluded *only* where it would divide by a degenerate PGR
  ceiling, by a pre-stated outcome-independent rule — kept everywhere else.

---

## Pre-registered scorecard (honesty trail)

Predictions committed to git **before** the confirming seeds/runs existed:

| # | Prediction | Outcome |
|---|---|---|
| P1 | xent curve concave, knee at 0.10–0.25 (small budget captures most benefit) | **REFUTED** — back-loaded, no knee |
| P2 | logconf flat | **Held** (flat; no recovery even at 100% GT) |
| P3 | Mixing > GT-only (weak labels add value) | **Held + de-confounded** (Component A) |
| P4 | Larger students extract more value | **Not supported** (underpowered, non-monotonic) |
| P5 | logconf shows no scale trend | **Held** (trivially — inert) |
| 1b-0 | Testbed can resolve a strategy contrast | **PASS** (MDE 0.007) |
| 1b-A | Mixing beats random-labels (info > quantity) | **Confirmed**, both tasks |
| 1b-B | Oracle beats naive (allocation has headroom) | **Refuted** — allocation is null |
| 1b-C | SciQ shows larger/earlier effect | **Partial** — cleaner PGR, *smaller* raw acc; shape replicates |

---

## Threats to validity

- **Model scale.** GPT-2 only (~12× span); near-zero baseline W2SG PGR. Effects may not transfer
  to frontier scale — the standard W2SG caveat, and the most likely reason scale-interaction and
  allocation came up null.
- **Effect sizes are modest** (single-digit accuracy points) on both tasks; neither is a
  high-signal regime for a strategy bake-off.
- **PGR denominator instability** for adjacent pairs — mitigated by raw-accuracy-first reporting.
- **SciQ is a targeted replication, not a full parallel sweep** (xent naive curve + de-confound
  control only; no logconf/gt_only/low-fractions/oracle).
- **Oracle bounds *error-targeting* allocation** specifically; a coverage/diversity axis
  orthogonal to weak-error location is not directly bounded (but is the weaker motivation).

---

## Bottom line & next step

At GPT-2 scale, with a near-zero baseline, the actionable picture is clear and cross-task:
**spend GT to get more of it (the curve is gradual, so small budgets buy little), trust that
weak labels carry real signal, and don't bother optimizing *where* the GT goes — it doesn't
move the needle.** The headline-grabbing hypotheses (a frugal "knee," supervision value scaling
with capability) are honest negatives here.

The binding constraint is **model scale, not method.** The rigorous next step is not more
strategies on GPT-2/BoolQ but to **re-run the fraction curve + the de-confound + the oracle on a
family spanning a much larger capability gap** (e.g. Pythia/Qwen across ~100×), where the
phenomena the original paper reports actually appear. Any GPT-2-scale Phase 2 (combination /
loss-dynamics / reliability methods) should be a **small, pre-registered probe framed as
"interesting even if it fails,"** not a broad sweep — because Component B already shows the
allocation axis is dead and neither task offers the signal a sweep would need.

---

## Rigor practices used (methodology slide)

Noise floor established first and every effect read against it · raw-accuracy-first with PGR
caveated · pre-registration committed to git before confirmation data · predictions scored
including refutations · a self-refuted headline (the knee) · an adversarially-handled bad seed
(reproduced, then variance-studied across 8 seeds, then excluded by an outcome-independent rule)
· a named confound (mixing>GT-only) gated behind a clean control · a power/MDE gate before any
strategy work · every cost/decision logged in `TIME_LOG.md`. Total GPU spend across all phases (S1–S8):
≈ **$515** (see TIME_LOG cost tables).
