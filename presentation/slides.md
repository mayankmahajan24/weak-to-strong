---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  /* Anthropic brand: Poppins headings / Lora body, warm palette, orange accent */
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Lora:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');
  section {
    font-family: "Lora", Georgia, serif;
    font-size: 25px;
    color: #141413;
    background: #faf9f5;
    padding: 48px 56px;
  }
  h1 { font-family: "Poppins", Arial, sans-serif; font-weight: 700; font-size: 44px; color: #141413; }
  h2 { font-family: "Poppins", Arial, sans-serif; font-weight: 600; font-size: 33px; color: #141413;
       border-bottom: 3px solid #d97757; padding-bottom: 6px; }
  strong { color: #d97757; font-weight: 600; }
  em { color: #6b6a63; }
  table { font-size: 21px; border-collapse: collapse; }
  th { background: #e8e6dc; color: #141413; }
  td, th { border: 1px solid #b0aea5; padding: 4px 10px; }
  a { color: #6a9bcc; }
  section.lead { justify-content: center; }
  section.lead h1 { font-size: 44px; line-height: 1.18; }
  footer { color: #b0aea5; font-size: 14px; }
  .small { font-size: 19px; color: #6b6a63; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# A small supervision budget barely improves weak-to-strong generalization

**On GPT-2 / BoolQ, mixing in a ground-truth budget helps only modestly — and *where* and *how* it's spent show little effect.**

A consistent picture emerges from the data: the student largely imitates the teacher's errors, and ground truth appears to repair mainly what it directly supervises — pointing to *volume* as the main lever.

<span class="small">GPT-2 family · BoolQ + SciQ · 3 seeds · paired per-(pair,seed) contrasts · pre-registered</span>

<!--
Open on the result, no background. The thesis sentence stands in for the intro.
-->

---

## Scope & setup

- **Setting:** weak-to-strong generalization — a strong *student* is trained on a weak *teacher's* labels.
- **Extension:** a *supervision budget* — mix a fraction of ground-truth ("strong") labels into the weak labels, and sweep that fraction.
- **Models:** **GPT-2 family only** — gpt2 / medium / large / xl, with within-family student–teacher pairs.
- **Tasks:** **BoolQ** (required) + **SciQ** (cross-task check).
- **Readout:** median **PGR** across the model sweep (raw accuracy primary; PGR secondary).

<span class="small">3 seeds · paired per-(pair,seed) contrasts · pre-registered · gpt2-large (seed 1) excluded by rule</span>

<!--
Scope card, not background — one glance. The interviewer knows W2SG; this just fixes the
experimental frame (family, tasks, the budget extension) before the results.
-->

---

## GPT-2 sweep at 25% ground truth

![bg right:50% fit](../results/plots/sweep_acc_boolq_gf025.png)

- GPT-2 family, BoolQ, **25% ground truth mixed into the weak labels** — paper-format sweep.
- **Median PGR (xent) = +0.30** across the sweep; logconf deeply negative (−1.3), so it's dropped.
- 25% GT moves the median from **−0.27 (0% GT) → +0.30**: real, but modest.
- Rest of the talk: *how* modest, where/how it does or doesn't help, and why.

<!--
This is the standardized readout. Lead with what it shows. logconf is already out of the picture.
-->

---

## How *much*? — no cheap knee; saturates by ~75%

![bg right:50% fit](../results/plots/phase1_fraction_curve.png)

- **≤10% GT sits within the noise floor (0.014)** — a small budget adds little above the pure-weak baseline.
- The response is **back-loaded**. Median xent PGR:
  −0.22 → **+0.30** (0.25) → +0.28 (0.50) → **+0.90** (0.75) → +1.04 (1.0).
- The 0.75 point shows it **saturates by ~75%** — the 0.75→1.0 step is within noise.
- A pre-registered "concave knee at 25%" prediction was **refuted** (retracted after the multi-seed view).

<!--
The brief's own suggestion was start-high-then-decrease to test sample efficiency. The answer is
that efficiency is poor: you need a real budget, and most of it pays off late.
-->

---

## *Where* it's spent — little effect (allocation null)

![bg right:50% fit](../results/plots/phase1b_B_allocation.png)

- A **perfect error-targeting oracle** places GT exactly on the teacher-wrong rows — an upper bound on any allocation heuristic.
- oracle − naive (random placement) = **+0.0006 / −0.0049** at 0.10 / 0.25 — inside the MDE (0.0071).
- Since even this oracle ties random placement, allocation heuristics have **little room to help here.**

<!--
The oracle is the strongest possible allocation strategy. A tie with random rules out the whole
class. This also replicates on SciQ.
-->

---

## *How* it's combined — little effect (combination null)

![bg right:50% fit](../results/plots/phase2_delta_bars.png)

- A pre-registered portfolio of **5 methods**: GT up-weighting, soft-GT, GT-anchored logconf, reliability-weighting, GT-early-stop.
- **None** left-shifts the curve or raises the ceiling — median Δ vs naive mixing ≈ 0.
- Only **gt-anchored** clears the floor (+0.040 @0.50), and it only *rescues* logconf — still below plain xent (0.642 vs 0.697).

<!--
A pre-registered portfolio that comes back null is a clean result. gt_anchored was the registered
"most likely positive"; it's the one partial hit, and even it loses to the simplest baseline.
-->

---

## Why: the student imitates the teacher's **errors**

![bg right:50% fit](../results/plots/mechanism.png)

- Mechanism probe (gpt2 → gpt2-xl), on per-example predictions.
- At 0% GT, the student copies the teacher's **wrong** answer **81% (BoolQ) / 70% (SciQ)** of the time.
- The failures look **largely inherited rather than independent** — the rows needing correction are mostly the teacher-wrong ones.

<!--
Sets up the payoff: the student isn't making fresh mistakes, so targeting teacher-wrong rows
*should* help — which makes the next slide the surprising part.
-->

---

## Why (hypothesis): recovery looks **diffuse** → volume may be the main lever

![bg right:50% fit](../results/plots/mechanism_recovery.png)

- GT repairs teacher-wrong rows **~linearly in budget** (42% / 55% of the gap at 50%) — rather than concave/targeted.
- A GT label's value appears **roughly independent of which row it lands on.**
- This is **consistent with all three results**: if recovery is volume-bound, *where* (allocation) and *how* (combination) have little leverage — leaving *how much* as the main lever.

<!--
The three separate negatives reduce to one property of the system. This is the central point.
-->

---

## Are the nulls real? — power and controls

- **Power:** paired per-(pair,seed) contrasts give **MDE = 0.0071**, below the 0.02 effect of interest — a null here is informative.
- **Pre-registration:** git-anchored predictions, hits *and* misses reported; the "knee" prediction was retracted.
- **Exclusion:** an 8-seed study found gpt2-large GT collapses **27%** of the time → excluded by rule, not by outcome.
- **De-confounding:** a random-label control shows weak labels are genuinely informative (mixing > GT-only isn't just row count).
- **Cross-task:** the findings replicate on **SciQ**.

<!--
The rigor half of the rubric. The MDE point is what licenses calling these "nulls" rather than
"couldn't tell."
-->

---

## Experiments across the three axes

| Axis | Tried | Outcome |
|---|---|---|
| How much | 8-point fraction sweep, 0 → 1 | back-loaded, saturates ~0.75 |
| Where | error-targeting oracle + random control | null — no signal on allocation |
| How | 5 combination / loss methods | null — 1 floor-clearer, still < xent |
| Loss | xent vs logconf | logconf inert → dropped |
| Tasks | BoolQ + SciQ | replicates on both |
| Mechanism | imitation-vs-correction probe | consistent with the nulls |
| Robustness | 8-seed variance, seeds 3–4 reserve | exclusion principled |

<span class="small">Most results are negative; together they point toward a common mechanism rather than leaving loose ends.</span>

<!--
Coverage across the axes. Frame the negatives as triangulation toward the mechanism.
-->

---

## Takeaways & the experiment that would change this

- Reproduced W2SG on the GPT-2 family and extended it to a supervision-budget setting.
- A **three-question decomposition** — *how much / where / how* — with a powered null on each.
- A single **hypothesis** (imitation + volume-bound recovery) is consistent with all three.
- **The data point to scale — rather than allocation or combination — as the main constraint here.**
  - The test that could change this: **larger student–teacher capacity gaps** (out of scope here — GPT-2 only).
  - Prediction: if recovery becomes **concave** at scale, *where* and *how* begin to matter.

<!--
Close on the synthesis and a concrete, falsifiable next experiment that respects the GPT-2 scope.
-->

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Appendix

<span class="small">frac=0 reproduction (acc + PGR) · transfer heatmap · per-method × pair detail · de-confound · cross-task</span>

---

## Appendix — frac=0 reproduction (the testbed)

![bg right:52% fit](../results/plots/sweep_pgr_boolq.png)

- Canonical W2SG sweep, GPT-2 family, BoolQ, **0% GT** — PGR axis.
- Median xent PGR **−0.27**: BoolQ is low-signal at zero strong supervision (several pairs sub-imitation).
- The baseline the 25%-GT plot is measured against.

---

## Appendix — transfer tracks the teacher more than student capacity

![bg right:52% fit](../results/plots/phase0_transfer_heatmap.png)

- Holding the teacher fixed, scaling the student up adds **~+0.003**; improving the teacher adds **~+0.04** (~10× here).
- Suggests the student's extra capacity is largely spent on imitation — the static counterpart of the mechanism.

---

## Appendix — combination portfolio, per-method detail

![bg right:52% fit](../results/plots/phase2_overlay.png)

- All five methods track naive mixing across {0.10, 0.25, 0.50}.
- gt-anchored (logconf) is the only floor-clearer; it rescues logconf but stays under xent.

---

## Appendix — weak labels are informative (de-confound)

![bg right:52% fit](../results/plots/phase1b_A_deconfound.png)

- Ordering: **random < gt_only < naive mixing** (15/15 BoolQ, 18/18 SciQ pairs).
- Replacing weak labels with noise *hurts* → mixing's gain reflects weak-label information, not just training-set size.
