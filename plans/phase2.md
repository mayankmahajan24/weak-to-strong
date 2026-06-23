# Phase 2 — Combination-Method Portfolio (how to combine weak + strong supervision)

**Status:** pre-registered plan. Written 2026-06-23, after Phase 1 + Phase 1b. **No runs begin
until the methods are implemented, unit-tested, and the predictions below are committed to git.**
Spine: `results/RESEARCH_PATH.md`. Findings so far: `results/FINDINGS.md`.

## Why this phase, and why *not* allocation

The interview rubric's #1 criterion is **"try as many different approaches/ideas as possible
(even if many fail) while maintaining rigor."** Phase 1b already triaged the idea space:

- **Allocation (Axis A) is dead.** The oracle (GT placed perfectly on the weak teacher's errors,
  100% error-coverage) **tied random placement** (oracle − naive = +0.0006 / −0.0049, within
  noise; confident, since MDE = 0.0071). So uncertainty/disagreement/diversity heuristics — which
  all approximate that oracle — are not worth running. *We present this as the reason we don't.*
- **Scale-up is out of scope** ("no models other than the GPT-2 family exist").

That leaves the orthogonal axis we have *not* tested: **how to combine the two signals per row**
(and how the loss treats them). This phase runs a portfolio of combination/loss/reliability
methods, each pre-registered, **reporting hits and misses**.

## Fixed universe

- Task **BoolQ**; models **GPT-2 family**; **3 seeds** (`gt_seed = seed`); EXCLUDE `(1,"gpt2-large")`.
- **Fractions {0.10, 0.25, 0.50}** — rationale: 0.10 = scarce/sample-efficiency regime (naive
  ≈ 0, max headroom, but ≥471 GT rows so methods are still *fittable*); 0.25 = the brief-mandated
  budget and where naive first clears noise; 0.50 = signal-rich anchor (naive +0.030, clearly
  resolvable) + ceiling-raising test + insurance against an all-null portfolio. **Not 1.0** (all-GT
  → every method is identical). **Not <0.10** (method-starvation risk).
- **6 strict weak<strong pairs**: `medium←gpt2`, `large←{gpt2,medium}`, `xl←{gpt2,medium,large}`
  (≈18 (pair,seed) cells/fraction → matches the power behind MDE 0.0071). Self-pairs excluded.
- **Loss:** xent for most; the GT-anchored method runs on logconf (it is a logconf repair).
- **Baseline is free:** naive mixing at these fractions/pairs already exists from Phase 1 — do
  **not** re-run it; every method is compared against it.
- **Noise floor = 0.014.** An effect counts only if the 3-seed median over pairs exceeds it.

## Success criteria (pre-registered, three readouts)

For each method, vs naive mixing at the matched (pair, fraction, seed):
1. **Beat-naive:** median Δacc(method − naive) > 0.014 at a given fraction.
2. **Left-shift (sample efficiency — the headline framing):** does method@0.10 reach naive@0.25,
   or method@0.25 reach naive@0.50? I.e., same generalization with less GT.
3. **Ceiling-raise:** at 0.50 (where naive works), does the method still beat it?

Report all three per method as a grid; a method can win on one and not others — that pattern is
itself a finding. **Honest prior:** given the oracle null and the modest Phase-1 effects, we
expect **most methods to be null; ≤1–2 to clear the floor.** Nulls are reported as results.

## The portfolio (ordered by priority / implementability)

Each entry: mechanism · code touch-point · pre-registered prediction. Implement + unit-test
each before the sweep (as we did for `oracle`/`random_labels`). Run as many as we can build
cleanly — breadth is the graded goal; compute is not the constraint (~54 runs/method).

**M1 — Weighted loss (GT upweighting).** Per-example loss weight: GT rows ×λ, weak rows ×1
(pilot λ ∈ {2,4,8}, then fix one). *Touch:* per-sample weight threaded into `train.py` loss
call; `label_source` already tags rows. *Predict:* small positive at 0.10 (amplifies scarce
clean signal), fading by 0.50; risk of overfitting the GT subset at high λ.

**M2 — Soft / confidence-weighted GT targets.** GT rows get a *soft* target (e.g. 0.9/0.1)
rather than one-hot, or blend weak-soft↔GT by weak confidence. *Touch:* `label_mixing.py`
(emit a soft target for GT rows). *Predict:* ≈neutral-to-slight-positive — reduces GT-row
overconfidence but adds little signal; mostly a regularization effect.

**M3 — GT-anchored logconf (rescue the logconf null).** Standard logconf blends the target with
the model's own hardened predictions (`target = labels·(1−c) + strong_preds·c`, `c→0.5`),
diluting even GT rows to ~50% self-prediction. **Anchor GT rows: exempt `label_source=="gt"`
from the auxiliary term** (hard GT), apply standard logconf only to weak rows. *Touch:*
`weak_to_strong/loss.py` (per-row mask on the confidence term). *Predict (strong, directional):*
moves logconf from deeply negative toward xent-like — i.e., the logconf null is **fixable** by
protecting the clean labels. A clean callback to the Phase-1 logconf finding.

**M4 — Teacher-reliability weighting (scalable-oversight-native, highest creativity).** Use the
GT subset to fit P(teacher correct | x) (logistic on pooled hidden states or weak-confidence
features over GT rows), predict reliability on weak rows, **downweight low-reliability weak
labels** in the loss. Reframes strong labels as a *calibration resource*, not just extra labels.
*Touch:* preprocessing that computes per-row weights → M1's weighted-loss path. *Predict:*
positive **iff** weak errors are feature-predictable from a few hundred GT rows; genuinely
uncertain — "interesting even if it fails."

**M5 — GT-as-early-stopping (reframe GT as evaluation, not training).** Train on pure weak labels;
use the GT subset as a clean validation set for checkpoint selection. *Touch:* `train.py` —
eval on GT subset every N steps, keep best checkpoint. *Predict:* small positive at low cost; a
sample-efficiency angle most candidates miss (uses the budget without ever training on it).

*(Stretch, if time: M6 relabel-and-retrain; M7 GT-guided weak-label filtering. Same 54-run unit.)*

## Run matrix & cost

Per method = 6 pairs × 3 fractions × 3 seeds = **54 runs**.

| | runs |
|---|---|
| M1–M5 (5 methods × 54) | 270 |
| HP pilots (λ, blend temp; 1 pair × 1 frac × few values) | ~30 |
| **Total** | **~300** |

Per-run times = naive mixing (full-data): gpt2 ~80s … xl ~625s on H200. ≈34 GPU-hr ÷ 8 ≈
**~5–6 h wall-clock, ~$150–180** on **8×H100 (preferred — same Hopper compute, ~30% cheaper;
80 GB fits gpt2-xl) or 8×H200**. Remaining budget ≈ $1,485 of $2,000 → ample;
scale to as many methods as implementable (~$30/method).

## Harness changes (more invasive than Phase 1b — flag in code, unit-test each)

1. **Per-sample loss weights** (M1, M4): thread an optional `sample_weight` (keyed off
   `label_source` or a precomputed vector) into the loss in `weak_to_strong/train.py`.
2. **GT-anchored logconf** (M3): per-row mask on the auxiliary confidence term in
   `weak_to_strong/loss.py` so GT rows are exempt.
3. **Soft GT targets** (M2): option in `label_mixing.py` to emit a soft/blended target for GT rows.
4. **GT-validation early-stopping** (M5): eval-on-GT-subset + best-checkpoint selection in `train.py`.
5. New flags (e.g. `--combination_method`, `--gt_loss_weight`) tagged into config/foldername
   **only when non-default**, so naive runs stay byte-identical.

**Validation (cheap, before the sweep):** unit-test each like `test_label_mixing.py` (synthetic
+ a real weak_labels arrow); confirm a `naive` run reproduces the existing Phase-1 number
bit-for-bit (default path unchanged).

## Execution

GPU-pool driver (reuse `run_ab_driver.py` pattern): job list over (method, pair, fraction, seed),
8 concurrent on 8×H100 (or H200), write to `results/data/phase2_<method>/...`, clean `pytorch_model*.bin`
+ `results.pkl` per run, arm the dead-man failsafe, pull slim, destroy on completion. Verify
N/N ok, 0 NaN (exclude any method that legitimately degrades-to-chance from the NaN gate, as we
did for `random_labels`).

## Deliverables

- `results/phase2/RESULTS_phase2.md`: the **portfolio grid** (method × {beat-naive, left-shift,
  ceiling-raise} × fraction), median Δacc ± seed range vs the 0.014 floor, and an explicit
  **negative-results section** (criterion #1).
- **Money figure:** each method's efficiency curve overlaid on the naive curve (the left-shift view).
- Pre-registered scorecard (predictions vs outcomes), updated `RESEARCH_PATH.md` + `TIME_LOG` (S9).

## Decision gate → presentation

Carry the 1–2 methods (if any) that clear the floor into the talk as positive results; present
the rest as honest negatives. Whether or not anything wins, the phase **completes the
three-question story**: *how much* (curve, no knee) · *where* (allocation null) · *how to
combine* (this portfolio). Then build the ≤20-min deck. If the whole portfolio is null, that is a
clean, defensible result — "at GPT-2 scale, beyond label quantity and weak-label informativeness,
neither *where* nor *how* you combine moves the needle" — and the honest next step is scale.
