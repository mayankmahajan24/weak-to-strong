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
