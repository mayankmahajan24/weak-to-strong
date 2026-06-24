# Mechanism — imitation vs correction (the headline result)

*Tier-1 experiment (S13, separate 8×H200, since destroyed): `gpt2 → gpt2-xl` mixing at GT fractions
{0, 0.1, 0.25, 0.5, 1.0} × BoolQ+SciQ × 2 seeds, **keeping per-example test predictions.** Joins the
teacher's and student's test predictions to ask: on the rows where the teacher is **wrong**, does the
student recover the truth as GT rises, or re-imitate the teacher's error? Driver
`scripts/phase2/run_mechanism_driver.py`, analysis `scripts/phase2/analyze_mechanism.py`, data
`mechanism_summary.csv`, figure `plots/mechanism.png`.*

| ds | frac | student acc \| teacher WRONG | student imitates teacher's error | student acc \| teacher right |
|---|---|---|---|---|
| boolq | 0.0 | **0.187** | **0.813** | 0.893 |
| boolq | 0.25 | 0.278 | 0.722 | 0.876 |
| boolq | 0.5 | 0.342 | 0.658 | 0.868 |
| boolq | 1.0 | 0.556 | 0.444 | 0.865 |
| sciq | 0.0 | **0.301** | **0.699** | 0.877 |
| sciq | 0.5 | 0.461 | 0.539 | 0.868 |
| sciq | 1.0 | 0.594 | 0.406 | 0.846 |

**Three findings:**

1. **The strong student imitates the teacher's mistakes.** With pure weak supervision (f=0), on the
   rows where gpt2 is wrong, gpt2-xl reproduces the teacher's *exact wrong answer* **81% (BoolQ) /
   70% (SciQ)** of the time, and is correct only 19% / 30%. The strong model doesn't "know better" —
   it copies the teacher's specific errors. This is the W2SG imitation phenomenon, measured directly
   in our GT-mixing setup. (On rows the teacher gets *right*, the student is ~88% correct and stays
   there — the whole dynamic lives on the teacher-wrong rows.)

2. **GT recovers those errors only *diffusely* — recovery is ~linear in budget, not concave.**
   Teacher-wrong accuracy climbs as GT rises, but at f=0.5 it has recovered just **42% (BoolQ) / 55%
   (SciQ)** of the full f=0→1 gap — i.e., roughly proportional to spend. A *little* GT does **not**
   teach the student to broadly override an unreliable teacher; you have to pay for the corrections
   roughly one budget-unit at a time.

3. **This mechanistically explains the Phase-1b allocation-null.** If correction is diffuse and
   ~linear in budget, then *where* you spend GT can't matter — targeting the teacher-wrong rows (the
   oracle) buys nothing over random, because the student doesn't leverage a few targeted GT labels
   into broad override; it just needs volume. **The "perfect oracle ties random" result is exactly
   what a diffuse, non-generalizing correction mechanism predicts.** (It also explains why the
   combination methods are null: none of them change this volume-bound dynamic.)

*Caveat:* teacher-wrong rows are intrinsically hard — even GT-trained (f=1.0) the student only
reaches 0.56–0.59 on them vs ~0.86 on teacher-right rows. So part of W2SG's weakness here is that
the teacher errs precisely on the examples the strong model also finds hard.

---

# Mechanism-lite — weak-teacher error structure (BoolQ)

Computed with zero GPU from the preserved `weak_labels` (each teacher's per-example `gt_label`,
`hard_label`, `soft_label`). Script: `scripts/phase2/weak_label_analysis.py`. This grounds two
Phase-1b results (the M4 reliability premise, the oracle-null) in the actual error structure the
methods operate on — partially addressing the "no mechanism" gap.

## 1. Weak error rates fall with scale (and the gpt2-large seed-1 anomaly shows here too)

| teacher | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| gpt2 | 0.346 | 0.329 | 0.338 | 0.338 |
| gpt2-medium | 0.310 | 0.303 | 0.288 | 0.300 |
| gpt2-large | 0.260 | **0.332** | 0.275 | 0.289 |
| gpt2-xl | 0.225 | 0.237 | 0.262 | 0.242 |

Monotonic with size as expected; seed-1 gpt2-large (0.332) is the documented instability surfacing
in its *teaching* quality, not just its GT ceiling.

## 2. Weak confidence IS calibrated → the M4 reliability premise is real but noisy

Accuracy within bins of weak confidence `max(soft_label)` (seed 0):

| teacher | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0) |
|---|---|---|---|---|---|
| gpt2 | 0.54 | 0.55 | 0.67 | 0.80 | — |
| gpt2-medium | 0.54 | 0.57 | 0.65 | 0.80 | 0.92 |
| gpt2-large | 0.53 | 0.68 | 0.78 | 0.89 | — |
| gpt2-xl | 0.51 | 0.59 | 0.73 | 0.87 | — |

Accuracy rises monotonically with confidence — so a confident weak label **is** more likely
correct, which is exactly the signal M4 (teacher-reliability weighting) tries to exploit. *But* the
signal is weak where it matters: the low-confidence bins (where you'd most want to down-weight) sit
at ~0.51–0.55, barely above chance, and a large fraction of mass lives there. So reliability
weighting has a real but shallow lever — consistent with our prior that M4 is "uncertain."

## 3. Errors are *systematic and shared across scale* → context for the oracle-null

P(larger teacher wrong | smaller teacher wrong), seed 0 (same transfer split):

| | → medium | → large | → xl |
|---|---|---|---|
| gpt2 wrong | 0.63 | 0.52 | **0.39** |
| medium wrong | — | 0.61 | 0.47 |
| large wrong | — | — | 0.62 |

A larger teacher repeats **39–63%** of a smaller teacher's mistakes — the errors are *systematic*,
not random noise, and heavily shared between adjacent scales. **Why this matters for Phase 1b's
oracle-null:** if weak errors are systematic (the whole family struggles on the same examples),
then putting GT on exactly those rows doesn't durably help — the student re-derives the systematic
error from the rest of the (weak-labeled) data. That is a plausible mechanism for why *perfect
error-targeting allocation tied random*: the benefit of GT is diffuse/regularizing, not
per-example correction. (A full test needs the student's per-example predictions, which we'd keep
on a future run — see the deferred mechanism analysis.)
