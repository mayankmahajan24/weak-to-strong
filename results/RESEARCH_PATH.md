# Research Path — Ground-Truth Mixing in Weak-to-Strong Generalization

**What this document is.** The spine of the project and its decision log. Read it top to
bottom and you can reconstruct the whole study: the question, the operating rules, then for
each phase *what we did → what we saw → how it changed what we did next*. Detailed numbers
live in the per-phase `RESULTS_*.md` (linked); pre-registrations in `NOTES_*.md`; the full
timestamped/cost record in [`../TIME_LOG.md`](../TIME_LOG.md) (sessions S1–S11). This file is
deliberately honest about predictions that were wrong and choices we revised — that trail is
the scientific contribution as much as the numbers.

## The question

Weak-to-strong generalization (W2SG): can a strong model trained on a weaker model's labels
exceed its teacher? Our question is one layer up — **treat strong/ground-truth (GT)
supervision as a scarce budget**:
1. **How much** GT do you need? (the marginal-value / fraction curve, and does it scale with student size)
2. **Where** should you spend it? (allocation)
3. **How** should you combine weak + GT signals? (combination method)

Universe: GPT-2 family only (`gpt2`, `-medium`, `-large`, `-xl`). Primary task BoolQ;
secondary SciQ for headline replication. Fork of `openai/weak-to-strong`. Master plan:
[`../plans/w2sg_gt_mixing_plan.md`](../plans/w2sg_gt_mixing_plan.md).

## Operating doctrine (held fixed from run #1)

- **Must-ship core = Phase 0 + Phase 1**; everything later is depth trimmed under time/compute.
- **Compute doctrine:** one condition ≈ 60 runs (10 GPT-2 pairs × 2 losses × 3 seeds).
  Statistical power beats breadth; descriptive curves at 3 seeds, head-to-head claims at 5;
  second dataset = headline replication only, not a parallel sweep.
- **Seed environment** (recorded in [`../NOTES_phase0.md`](../NOTES_phase0.md)): `gt_seed`,
  weak labels, GT ceilings, and student init all **reseed per seed** so error bars capture the
  real variance sources. The correctness invariant — injected GT must come from the *transfer
  split*, never the weak model's training data — is asserted in code.
- **Raw accuracy is primary; PGR is secondary and caveated** (small denominators for adjacent
  pairs inflate it). Every effect is read against a measured **noise floor**.
- **Pre-register predictions before confirmation data; report hits and misses.**

---

## Phase 0 — Infrastructure, baseline reproduction, mixing harness
*(detail: [`phase0/RESULTS_phase0.md`](phase0/RESULTS_phase0.md); sessions S1–S4)*

- **Did.**
  - **M1** environment smoke tests.
  - **M2** reproduced the W2SG baseline across the full GPT-2 family on **BoolQ + SciQ, 3 seeds,
    xent + logconf** (a multi-instance GPU effort: Lambda + Vast A100/B200/H100; see S1–S2).
  - Built + validated the GT-mixing harness (`weak_to_strong/label_mixing.py`):
    **M3** identity check (gt_fraction=0 ≡ baseline; diff 0.0018 = FP noise — PASS),
    **M4** ceiling check (gt_fraction=1.0 → 0.742 vs 0.655 baseline, +0.088 — PASS),
    **M5** first real datapoint: 25% GT mix.
- **Saw.** Baseline W2SG signal is weak at this scale: **BoolQ xent median PGR ≈ −0.31,
  SciQ ≈ +0.17** (logconf deeply negative, as expected). 25% mix beat GT-only decisively
  (xent PGR −0.85 → +0.06). Flagged the **seed-1 gpt2-large GT anomaly** (0.662).
- **Informed.** A signal exists but is small and single-seed → the must-ship core needs a
  **full fraction curve × 3 seeds** with a noise floor. Chose **BoolQ as primary** (higher
  variance / more discriminating) — a choice Phase 1b later re-examines, since it is also the
  *lower-signal* task.

## Phase 1 — How much supervision? Fraction curve + scale interaction
*(detail: [`phase1/RESULTS_phase1.md`](phase1/RESULTS_phase1.md); pre-registration:
[`../NOTES_phase1.md`](../NOTES_phase1.md); sessions S5–S6)*

- **Did.** Pre-registered six predictions (P1–P6) against seed-1 **before seeds 0/2 existed**
  (git-anchored). Ran the full 3-seed sweep on BoolQ, `gt_fraction ∈ {0.01,0.05,0.1,0.25,0.5,1.0}`,
  mixing + GT-only, xent + logconf — **504 canonical runs** (336 of them the seeds-0+2 sweep on
  8× H200 in S6). Computed the **noise floor (M5 of the plan): 0.014**.
- **Saw.**
  1. GT mixing helps xent but is **back-loaded — no knee**: ≤0.10 within noise; detectable at
     0.25, robust at 0.50, largest at 1.0. **P1 (concave/diminishing-returns knee) REFUTED.**
  2. **Mixing > GT-only at every budget < 1.0** (largest effect, +0.05–0.07) — *but confounded
     by training-set size* (GT-only is data-starved at low fractions). Strongest *candidate*
     headline, not clean. **P3 held as stated but cannot yet claim weak-label informativeness.**
  3. **logconf inert at every budget incl. 100% GT** — clean null with a loss-function mechanism
     (the confidence term dilutes the target with self-predictions). **P2/P5 held.**
  4. **Scale interaction inconclusive / underpowered. P4 not supported.**
- **Honesty trail (two corrections).**
  - *Retracted our own headline:* a single-seed draft asserted a "knee at 0.25." Seed 0 erased
    it; a per-pair re-analysis showed it was a **PGR zero-crossing artifact** amplified by the
    seed-1 anomaly. The raw-accuracy curve has no elbow.
  - *Adversarial data handling (the seed-1 gpt2-large anomaly):* in **S5** we regenerated the
    exact run — it reproduced **0.662 bit-for-bit** (deterministic; weak labels' hard labels
    identical) → a *real bad-optimization outcome*, not corruption. Excluded under a
    **pre-stated, outcome-independent failed-ceiling rule** (`EXCLUDE={(1,"gpt2-large")}`), kept
    in the dataset elsewhere; the exclusion *tightened* the noise floor 0.018→0.014.
- **Informed.** The exciting "knee / scale-interaction" story did **not** survive 3 seeds; what
  survived is modest, null, or confounded. → Before any Phase 2 strategy work, insert a gate
  (Phase 1b). Decision-gate also set Phase 2 budget to {0.10, 0.25} and **dropped logconf**.

## Side-thread — Is the bad seed a fluke? gpt2-large GT variance study
*(detail: `phase0/gpt2large_variance/SUMMARY.md`; session S7)*

- **Did.** Ran **8 fresh seeds** (11–18) of gpt2-large GT on BoolQ (ceiling only, vary `--seed`),
  on 8× H100 (~6 min compute) — to test whether seed-1's 0.662 was a one-off.
- **Saw.** Across 11 seeds: mean **0.709**, std **0.030**, range [0.649, 0.743]; **3/11 (27%)**
  fall into a low mode <0.70 (below gpt2-medium's GT). seed-1 is a *representative draw from a
  recurring ~27% failure mode* — gpt2-large GT is **optimization-unstable** (likely lr=1e-5 too low).
- **Informed.** Independently confirms the failed-ceiling exclusion was principled (not
  cherry-picking) and *reinforces* "scale interaction inconclusive" — the mid-scale model's
  ceiling is intrinsically noisy here. A clean methods footnote and a rigor slide.

## Phase 1b — Testbed & premise validation (GATE before Phase 2)
*(plan: [`../plans/phase1b.md`](../plans/phase1b.md); detail:
[`phase1b/RESULTS_phase1b.md`](phase1b/RESULTS_phase1b.md))*

The intellectually honest move: don't tune strategies until we show the testbed can **detect**
the effects and the **premises hold**. Four tests, each with a pre-registered kill criterion.

- **Component 0 — Power/MDE (done ✅ PASS).** Paired (pair,seed) contrasts give a minimum
  detectable effect of **0.0071 acc** (1-sided, 80% power) — below the 0.02 gate and below
  naive's own movement. ⇒ a null in A/B will be a *genuine* null, not underpowered.
- **Component A — De-confound "mixing > GT-only" (done ✅ INFORMATIVE).** `random_labels`
  holds row/step count equal to mixing but replaces non-GT weak labels with noise.
  `mixing − random_labels = +0.079 / +0.081` (10%/25%), **15/15 pairs, both fractions** (~11×
  the floor). Ordering: `random 0.61 < gt_only 0.62 < mixing 0.67–0.69` — noise *hurts*, so
  mixing's win is **real weak-label information, not row count.** Result-2 confound resolved;
  P3 upheld cleanly.
- **Component B — Oracle-allocation ceiling (done ❌ NULL).** `oracle` spends GT on the weak
  teacher's wrong rows (verified 100% error-coverage). `oracle − naive = +0.0006 / −0.0049`,
  ~coin-flip sign — within noise at both budgets, and Component 0 says we'd have seen ≥0.007, so
  a **confident null.** Even perfect error-targeting gives no gain over random placement.
- **Component C — SciQ validity (done ◑).** 144 SciQ runs. The naive curve is **cleaner** on
  SciQ (PGR positive & monotonic from 0.10: +0.16→+0.94; BoolQ starts negative) but moves **less
  in raw accuracy** (+0.033 vs +0.055 at 1.0 — less headroom). The **no-knee/gradual shape and the
  weak-label-informativeness (A) both replicate** — A even stronger on SciQ (+0.16/+0.11, 18/18;
  random_labels collapses to ~chance). Testbed validated, but SciQ offers *smaller* effects, so
  it's not a higher-signal venue for Phase 2.

- **Informed.** A+B+C give a sharp, honest, **cross-task** story: *which* labels matters (weak
  labels informative, both tasks), *where* you place GT does not (allocation null), *how much* is
  gradual/back-loaded with no knee (both tasks). This **kills Phase-2 Axis A (allocation)** —
  uncertainty/disagreement/diversity all approximate the error-targeting the oracle bounded at
  zero. So Phase 2 is **focused, not a broad bake-off**: a pre-registered *combination*-axis
  portfolio (B/C/D) testing the one untested lever, *how* to combine weak + GT. The defensible
  contribution either way is the cross-task-replicated **characterization** + the allocation null
  + the gpt2-large instability + the logconf null; the **larger model gap** is the honest
  future-work lever (out of scope per the brief), not this phase.

---

## Forward roadmap (decided after Phase 1b + reading the interview brief)

Phase 1b triaged the original idea-space portfolio: **Axis A (allocation) is dead** (oracle
null), so we do **not** run allocation heuristics. The interview rubric's #1 criterion is breadth
of *plausible* approaches with rigor, so Phase 2 is **on** — but focused on the one untested axis.

- **Phase 2 — combination-method portfolio** (planned; *how* to combine weak + GT per row).
  5 pre-registered methods — weighted loss, soft-GT, **GT-anchored logconf** (rescue the logconf
  null), teacher-reliability weighting, GT-as-early-stopping — at {0.10, 0.25, 0.50}, 6 strict
  pairs, 3 seeds (~300 runs, ~$170 on 8×H100). Success = beat-naive / left-shift / ceiling-raise
  vs the 0.014 floor; report hits **and** misses. Plan: [`../plans/phase2.md`](../plans/phase2.md);
  execution spec (code, tests, invariants): [`../plans/PHASE2_PROMPT.md`](../plans/PHASE2_PROMPT.md).
- **Phase 3 — mechanism + robustness:** imitation analysis (does a winner fix weak-*error* rows?),
  weak-quality interaction, noisy-oversight robustness. (SciQ headline replication already done in 1b-C.)
- **Phase 4 — synthesis:** the ≤20-min talk on the three questions (how much / where / how), with
  the pre-registration scorecard + noise-floor + negative-results slides as the rigor backbone.
- **Out of scope (future work, not a deliverable):** the capability-gap scale-up to a ~100× family
  is the scientifically correct next lever, but the brief fixes the universe to GPT-2 ("no models
  other than the GPT-2 family exist") — so it lives in the talk as future work, not as compute spend.

## How we kept ourselves honest (the through-line for the talk)

- **Pre-registration, git-anchored** — P1–P6 committed before seeds 0/2 existed → later seeds
  were a real out-of-sample test, not post-hoc storytelling.
- **We refuted our own headline** — the "knee at 0.25" was retracted once a second seed + a
  per-pair re-analysis exposed it as a metric artifact.
- **Adversarial data handling** — the bad seed was *reproduced* (S5) and then *variance-studied
  across 8 seeds* (S7) to prove it was real, then excluded by an outcome-independent rule — not
  deleted because inconvenient.
- **We named our own confound** — the strongest effect (mixing > GT-only) is flagged as
  training-set-size-confounded and gated behind a clean control, not sold as-is.
- **We gate before we scale** — Phase 1b exists to avoid "rigor theater": power is checked
  before strategies are built; every branch has a pre-committed action, including "stop."

## Current status

| Phase | State | Headline |
|---|---|---|
| 0 | ✅ | Baseline reproduced; harness validated; BoolQ low-signal, SciQ higher |
| 1 | ✅ | No knee (P1 refuted); mixing>GT-only (confounded); logconf null; scale inconclusive |
| — | ✅ | gpt2-large GT ~27% unstable (S7) → exclusion principled, scale claim under-measured |
| 1b | ✅ 0 PASS; A informative (both tasks); B allocation null; C testbed validated, findings replicate | Which labels matters; where doesn't; gradual no-knee curve — cross-task |
| 2 | ▶ **running** (8×H200): gates passed — naive bit-for-bit (0.69257), M5 smoke (0 GT, acc 0.65); 270-run combination sweep in progress | Tests *how to combine*; Axis A dropped (oracle null) |
| 3–4 | ⏳ pending | Mechanism analysis + the ≤20-min synthesis talk |

**Immediate next:** implement Phase 2 per [`../plans/PHASE2_PROMPT.md`](../plans/PHASE2_PROMPT.md)
— Phase A plumbing (loss `**kwargs` + per-row `gt_mask`/`sample_weight` threading + the
`--combination_method` flag) with `tests/test_losses.py` and the naive-reproduction regression,
then M1 + M3 first; commit the pre-registered predictions; run ~300 on 8×H100; then build the deck.

## Artifact index
- **Synthesis writeup (start here for the findings): [`FINDINGS.md`](FINDINGS.md).**
- Narrative spine / decision log: this file.
- Phase results: `phase0/RESULTS_phase0.md`, `phase1/RESULTS_phase1.md`, `phase1b/RESULTS_phase1b.md`;
  variance study `phase0/gpt2large_variance/SUMMARY.md`.
- Pre-registrations: `../NOTES_phase1.md` (P1–P6), `../NOTES_phase2.md` (M1–M5), `../NOTES_phase0.md` (code map + seed table + anomaly adjudication).
- Plans: `../plans/w2sg_gt_mixing_plan.md` (master), `../plans/phase0.md`, `../plans/phase1.md`,
  `../plans/phase1b.md`, `../plans/phase2.md` (combination portfolio) + `../plans/PHASE2_PROMPT.md`
  (execution spec: code/tests/invariants).
- Interview brief (the rubric + scope constraints): `../docs/INTERVIEW_INSTRUCTIONS.pdf`.
- Code (by phase): `../scripts/phase1/` (consolidate/analyze/compare/robustness/plot), `../scripts/phase1b/` (power/analyze/drivers), `../scripts/phase0/` (baseline runners), `../tests/`.
- Data + outputs: `phase1/phase1_results.csv`, `phase*/RESULTS_*.md`, `plots/`, `data/`.
- Session / cost / lessons log: `../TIME_LOG.md` (S1–S8).
