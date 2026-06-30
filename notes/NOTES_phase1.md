# Phase 1 — Pre-Registration (predictions + frozen methodology)

**Registered:** 2026-06-22 ~23:00 UTC, against seed-1 data only.
**Anchor commit:** `59ee63e` (seed-1 mixing + GT-only captured; seeds 0 & 2 NOT yet
collected at registration time).
**Purpose:** Lock the analysis pipeline and quantitative predictions *before* seeds 0
and 2 are observed, so those seeds constitute an out-of-sample replication test rather
than an opportunity for post-hoc analysis choices. Nothing in the "Frozen analysis
decisions" or "Predictions" sections may be edited after seeds 0/2 land — deviations get
recorded in a dated "Deviations" section at the bottom, with rationale.

---

## 1. Frozen analysis decisions (the pipeline may not change after seeds 0/2)

1. **PGR formula:** `PGR = (transfer_acc − weak_GT_acc) / (strong_GT_acc − weak_GT_acc)`,
   GT accuracies = the blank-weak baseline runs (model fine-tuned on true labels),
   computed **per seed** (no cross-seed pooling of the denominator).
2. **Valid pair rule:** include a (strong, weak) pair only if the strong arch is strictly
   larger than the weak arch AND `strong_GT > weak_GT` **in that seed**. Self-supervision
   pairs (strong == weak) are excluded from PGR.
3. **Aggregation:** report the **median** PGR across valid pairs within a seed; across
   seeds report the median-of-medians and the min–max range as the band.
4. **Metric-by-regime rule (from seed-1 nuance):** at `gt_fraction ≥ 0.5`, mixing@1.0 can
   exceed the GT ceiling (it trains GT on the larger transfer split), pushing PGR > 1 and
   making the denominator a poor normalizer. Therefore the **headline curve is read in raw
   test accuracy**; PGR is reported as a secondary view and is interpreted only for
   `gt_fraction ≤ 0.25`.
5. **gpt2-large GT anomaly:** seed-1 gpt2-large GT (0.662) is a training pathology (below
   gpt2's 0.665) that invalidates every large-as-student pair. **Decision, made now:**
   regenerate gpt2-large GT for seed 1 alongside the seeds 0/2 GT runs (Milestone 1). A
   per-seed GT that still inverts (`large_GT < gpt2_GT`) after regeneration excludes that
   seed's large-as-student pairs by rule (2) — it is **not** hand-removed.
6. **Noise floor:** the cross-seed noise floor (M5) is the per-pair baseline (frac=0.0)
   accuracy range across the 3 seeds. Any mixing effect smaller than this band is reported
   as null. (The Phase-0 within-seed FP figure, 0.0018, is only a lower bound and is not
   used as the floor.)
7. **Scale interaction is read via the per-pair PGR zero-crossing fraction**, NOT via PGR
   magnitude across students — magnitude is confounded by the per-pair GT-gap denominator.

---

## 2. Seed-1 observed values (the basis for the predictions below)

Reference only — these are what we are predicting will (or won't) replicate.

- **xent median-PGR fraction curve:** 0.0:−0.21, 0.01:−0.21, 0.05:−0.20, 0.10:−0.13,
  0.25:**+0.27**, 0.50:+0.30, 1.0:+1.08. Knee at 0.25; all 4/4 valid pairs flip positive
  at 0.25; effect at 0.25 is 18× the within-seed FP floor.
- **xent raw median acc:** 0.657 → 0.657 → 0.662 → 0.664 → **0.693** → 0.683 → 0.743.
- **mixing − gt_only (raw acc, xent):** +0.039, +0.037, +0.042, +0.056, +0.060 for
  fracs 0.01–0.50; −0.001 at 1.0.
- **logconf raw median acc:** ~flat at 0.605, 0.601, 0.599, 0.597, 0.608, 0.601, 0.600.
- **scale (xent zero-crossing):** medium←gpt2 crosses ~0.03; xl←{small} crosses ~0.18.

---

## 3. Predictions (quantitative, falsifiable) — registered before seeds 0/2

**P1 — xent supervision knee reproduces.** The xent raw-accuracy curve is flat
(within the M5 noise band) for `gt_fraction ≤ 0.10` and shows a positive step at 0.25 in
**≥ 2 of 3 seeds**. Median-of-3 raw-acc delta(0.25 − 0.0) ≥ +0.015 and ≥ 2× the M5 floor.
*Refuted if* the 0.25 step is within the noise band in ≥ 2 seeds, or the knee location
moves below 0.10 or above 0.25 in the 3-seed median.

**P2 — sub-10% GT is inert (xent).** Median-of-3 raw-acc delta(0.10 − 0.0) is within the
M5 noise band (no reliable effect below 10% GT).
*Refuted if* delta(0.10 − 0.0) exceeds the noise band in ≥ 2 seeds.

**P3 — mixing beats GT-only at every fraction < 1.0 (xent).** Median-of-3
(mixing − gt_only) raw acc is > 0 at each of 0.01, 0.05, 0.10, 0.25, 0.50, and the gap
is non-decreasing in fraction up to 0.50; the gap is ≈ 0 (|·| < M5 floor) at 1.0.
This is the prediction we hold most strongly.
*Refuted if* GT-only matches or beats mixing at any fraction < 1.0 in the 3-seed median.

**P4 — logconf is null under mixing.** logconf raw median acc stays flat: median-of-3
|delta(frac − 0.0)| < 2× M5 floor at every fraction ≤ 0.50.
*Refuted if* logconf shows a monotone or stepped gain exceeding that band.
*Consequence if confirmed:* drop logconf from Phase 2 (per decision gate).

**P5 — scale interaction: larger capacity gap needs more GT (directional).** Using the
zero-crossing fraction per valid pair (3-seed median), pairs with a **larger GT gap**
cross to positive PGR at a **higher** gt_fraction. Concretely, xl←{gpt2,medium} crosses
at a higher fraction than medium←gpt2.
*Status:* **weakest prediction; flagged underpowered.** Conditional on Milestone-1
gpt2-large regeneration restoring large-as-student pairs in ≥ 2 seeds. If large stays
anomalous, P5 is reported as **not testable at GPT-2 scale this phase**, not as refuted.
Note this reframes Phase-0/plan prediction #4 ("larger *students* extract more"): seed-1
suggests the governing variable is **gap size**, not absolute student size.

**P6 — 0.25→0.50 plateau.** Marginal value of GT between 0.25 and 0.50 is small:
median-of-3 raw-acc delta(0.50 − 0.25) < delta(0.25 − 0.10). Seed-1 showed an actual dip
driven by a single pair (xl←large); we predict a **plateau, not a reliable dip**, in the
3-seed median.
*Refuted if* delta(0.50 − 0.25) ≥ delta(0.25 − 0.10) in the 3-seed median (no diminishing
returns), or a significant dip survives all 3 seeds.

---

## 4. Pre-stated confounds & threats to validity

- **Small denominators:** adjacent GPT-2 pairs have GT gaps < 0.03; PGR is hypersensitive
  there. Mitigated by raw-acc headline (decision 4) and median aggregation.
- **Few valid pairs:** xent has 4 valid pairs in seed 1 (2 for logconf). Seeds 0/2 may
  yield different valid-pair sets per rule (2); the curve is therefore reported with the
  per-seed valid-pair count shown, never hidden.
- **gt_seed = seed coupling:** GT-row selection is tied to the seed (per plan). We are not
  separating "which rows are GT" variance from "data split" variance this phase; noted.
- **mixing@1.0 > ceiling:** see decision 4 — do not over-read PGR at high fraction.
- **Single dataset (BoolQ):** SciQ replication is deferred to Phase 3; no cross-dataset
  claims are made from Phase 1.

---

## 5. Decision gate wiring (what each outcome triggers)

- P1 + P2 confirmed → headline "scarce supervision has a ~25% knee on BoolQ"; Phase 2
  strategies are tested at the 0.10–0.25 budget where the naive curve hasn't saturated.
- P3 confirmed → "weak labels provide coverage GT-only can't" becomes a standalone result.
- P4 confirmed → Phase 2 runs xent-only (drop logconf), saving ~50% compute.
- P5 confirmed → foreground "scarce supervision scales with capability gap"; if not
  testable → report honestly as underpowered and defer to a larger model family.

---

## Deviations (append-only; record any post-registration pipeline change here)

_(none yet)_
