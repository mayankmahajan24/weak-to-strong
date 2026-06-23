# Research Path — Ground-Truth Mixing in Weak-to-Strong Generalization

**What this document is.** The spine of the project. Read it top to bottom and you can
reconstruct the whole study: the question, what we did at each phase, what we saw, and how
each result *changed what we did next*. Detailed numbers live in the per-phase `RESULTS_*.md`
files (linked); this file is the narrative and the decision log. It is deliberately honest
about predictions that were wrong and choices we revised — that trail is the point.

## The question

Weak-to-strong generalization (W2SG) studies whether a strong model trained on a weaker
model's labels can exceed its teacher. Our question is one layer up: **if you can also inject
a little ground truth (GT), how much does it help, how should you spend it, and does that
change with model capability?** Concretely, on the GPT-2 family + BoolQ/SciQ, we vary the GT
fraction mixed into weak labels and measure transfer accuracy / PGR.

**Measurement discipline (constant throughout).** Raw accuracy is the primary readout; PGR is
secondary and explicitly caveated (its denominator, GT-ceiling − weak-floor, is tiny for
adjacent pairs and inflates noise). Every effect is read against a measured **noise floor**.
We **pre-register predictions before seeing confirmation data**, and we score them honestly —
including refutations.

---

## Phase 0 — Reproduce the baseline, build the mixing harness
*(detail: [`phase0/RESULTS_phase0.md`](phase0/RESULTS_phase0.md))*

- **Did.** Reproduced the W2SG baseline across the GPT-2 family on BoolQ + SciQ, 3 seeds, xent
  + logconf. Built and validated the GT-mixing harness (`label_mixing.py`), including
  identity (gt_fraction=0 ≡ baseline) and ceiling (gt_fraction=1.0) checks.
- **Saw.** Baseline W2SG signal at this scale is weak: **BoolQ xent PGR ≈ −0.31, SciQ ≈ +0.17**.
  A first 25%-mix datapoint beat GT-only. Noted the **seed-1 gpt2-large GT anomaly** (0.662).
- **Informed.** Signal exists but is small/single-seed → need a **fraction curve × multi-seed**.
  Chose BoolQ as primary because it is higher-variance / more discriminating. *(We revisit
  this choice in Phase 1b — it is also the lower-signal task, which cuts both ways.)*

## Phase 1 — Supervision-scaling fraction curve + scale interaction
*(detail: [`phase1/RESULTS_phase1.md`](phase1/RESULTS_phase1.md); pre-registration:
[`../NOTES_phase1.md`](../NOTES_phase1.md))*

- **Did.** Full 3-seed (0,1,2) fraction sweep on BoolQ, `gt_fraction ∈ {0.01…1.0}`, mixing +
  GT-only, xent + logconf — 504 canonical runs. Pre-registered six predictions (P1–P6) against
  seed-1 *before* seeds 0/2 existed (anchored to a git commit).
- **Saw.**
  1. GT mixing helps xent but is **back-loaded — no knee**: ≤0.10 is within the noise floor;
     detectable at 0.25, robust at 0.50, largest at 1.0. **P1 (concave/knee) REFUTED.**
  2. **Mixing > GT-only at every budget < 1.0** (largest effect) — *but confounded by
     training-set size* (GT-only is data-starved at low fractions). Strongest *candidate*
     headline, not a clean one.
  3. **logconf inert at every budget incl. 100% GT** (clean null, with a loss-function mechanism).
  4. **Scale interaction inconclusive / underpowered** — no claim.
- **Honesty trail.** An early single-seed draft asserted a "knee at 0.25." Seed 0 erased it; a
  per-pair re-analysis showed it was a **PGR zero-crossing artifact** amplified by the seed-1
  anomaly. We retracted it. Separately, the seed-1 gpt2-large anomaly was **adjudicated by
  regeneration** (it reproduced 0.662 bit-for-bit → real seed variance, not corruption) and
  excluded under a **pre-stated, outcome-independent** failed-ceiling rule — kept in the
  dataset, excluded only where it would divide by a degenerate ceiling. *(detail:
  [`../NOTES_phase0.md`](../NOTES_phase0.md) adjudication section.)*
- **Informed.** The exciting "knee / scale-interaction" story did **not** survive. What
  survived is modest, null, or confounded. So before building elaborate Phase 2 strategies, we
  must check (a) whether the strongest result is real or an artifact, (b) whether smart
  allocation can do anything here at all, and (c) whether we can even resolve such effects.
  → **Insert Phase 1b as a gate.** Decision-gate specifics also set Phase 2 budget to
  {0.10, 0.25} and drop logconf.

## Phase 1b — Testbed & premise validation (GATE before Phase 2)
*(plan: [`../plans/phase1b.md`](../plans/phase1b.md); detail:
[`phase1b/RESULTS_phase1b.md`](phase1b/RESULTS_phase1b.md))*

The intellectually honest move: do not tune strategies on a testbed until we show it can
**detect** the effects and that the **premises hold**. Four tests, each with a pre-registered
kill criterion.

- **Component 0 — Power/MDE (done ✅ PASS).** Paired (pair,seed) contrasts give a minimum
  detectable effect of **0.0071 acc** (1-sided, 80% power) — below the 0.02 gate and below
  naive's own movement. So a null in A/B would be a *genuine* null, not underpowered.
- **Component A — De-confound "mixing > GT-only"** *(harness ready, runs pending)*. New
  `random_labels` strategy holds row/step count fixed and removes weak-label information; if
  mixing still beats it, weak labels are informative *beyond quantity*.
- **Component B — Oracle-allocation ceiling** *(harness ready, runs pending)*. New `oracle`
  strategy spends GT on the rows the weak teacher got wrong. On real data, oracle@10% achieves
  **100% error-coverage vs naive's ~33%** — the lever exists; B tests whether it moves accuracy.
  If even the oracle ≈ naive, no realistic Phase 2 allocation strategy will help.
- **Component C — SciQ validity** *(pending)*. BoolQ is near-zero baseline PGR; SciQ is +0.17.
  Is GT-mixing larger/cleaner where the underlying phenomenon actually exists? Decides whether
  BoolQ was the right primary task.

- **Informed (decision tree).** Green-light full Phase 2 only if **B shows oracle headroom AND
  power adequate**, on whichever task **C** says is higher-signal. If **A** collapses → reframe
  the claim from "weak labels are valuable" to "GT-budget effects." If **B** is null → allocation
  is dead at this scale (pivot to combination methods or escalate model scale). Worst case
  (B null + C null) → write the **bounded-null** result honestly and specify the larger model
  gap that would be required, rather than tuning strategies on noise.

---

## How we kept ourselves honest (the through-line for the talk)

- **Pre-registration with git anchoring** — predictions (P1–P6) were committed before seeds
  0/2 existed, turning later seeds into a real out-of-sample test, not post-hoc storytelling.
- **We refuted our own headline.** The "knee at 0.25" was retracted once a second seed and a
  per-pair re-analysis exposed it as a metric artifact.
- **Adversarial data handling.** A bad seed was *reproduced* to prove it was real variance, then
  excluded by an outcome-independent rule — not deleted because it was inconvenient.
- **We named our own confound.** The strongest effect (mixing > GT-only) is flagged as
  training-set-size-confounded and gated behind a clean control rather than sold as-is.
- **We gate before we scale.** Phase 1b exists specifically to avoid "rigor theater" — running
  an elaborate Phase 2 on a testbed that may not resolve the effect. Power is checked first.
- **Every branch has a pre-committed action, including "stop."**

## Current status

| Phase | State | Headline |
|---|---|---|
| 0 | ✅ | Baseline reproduced; harness validated; BoolQ low-signal, SciQ higher |
| 1 | ✅ | No knee (P1 refuted); mixing>GT-only (confounded); logconf null; scale inconclusive |
| 1b | ▶ Component 0 PASS; A/B/C pending | Testbed can resolve ≥0.007; premises now under test |
| 2 | ⛔ gated | Begins only if Phase 1b clears its kill criteria |

**Immediate next:** run Phase 1b Components A + B on BoolQ (harness implemented + unit-tested),
then C on SciQ; score against the kill criteria; update `phase1b/RESULTS_phase1b.md` and this
file.

## Artifact index
- Narrative spine: this file.
- Phase results: `phase0/RESULTS_phase0.md`, `phase1/RESULTS_phase1.md`, `phase1b/RESULTS_phase1b.md`.
- Pre-registrations: `../NOTES_phase1.md` (predictions), `../NOTES_phase0.md` (code map + anomaly adjudication).
- Plans: `../plans/phase1.md`, `../plans/phase1b.md`.
- Data + reproduce: `phase1/phase1_results.csv`, `phase1/consolidate_phase1.py`, `phase1b/*.py`.
- Session/cost log: `../TIME_LOG.md`.
