# Plan — Elicitation: spend the GT budget to *orient latent knowledge*, not as labels

**Status:** spec + advance predictions, **git-anchored before any elicitation run.**
Companion analysis lands in `results/elicitation/RESULTS_elicitation.md`. Reuses the Phase-0/1
testbed (GPT-2 family, BoolQ + SciQ, the GT ceilings, the `EXCLUDE={(1,"gpt2-large")}` rule, the
0.014 noise floor, the strict-pair PGR convention).

## Question
Our central result: a small GT budget barely helps **as training labels** (volume-bound mixing).
Elicitation tests a different lever — does a small budget help if spent **pointing at knowledge the
strong model already has** (a readout from its frozen representations)? And, per Anthropic's
automated-W2S-researcher result (elicitation dominates at *large* gaps), does any elicitable signal
exist at our *small* GPT-2 gaps?

PGR convention (unchanged): `PGR = (acc − weak_GT) / (strong_GT − weak_GT)`, strict weak<strong
pairs, median over (pair, seed), read against the **0.014 noise floor**; `EXCLUDE={(1,"gpt2-large")}`.

## Advance predictions (frozen before running)
From the volume-bound mechanism, elicitable signal should grow with the capability gap, so at GPT-2
scale it should be weak. Falsifiable:
- **EL1.** Elicited test accuracy rises monotonically with strong-model size (gpt2 < medium < large < xl).
- **EL2.** At a *tiny* matched budget (k ≤ 32 GT labels), elicitation ≥ naive mixing for the widest
  pair (gpt2→gpt2-xl); ≈ tie for adjacent pairs.
- **EL3.** Where it works at all, elicitation is far more **sample-efficient** than mixing — it
  reaches its own ceiling with k ≪ the budget mixing needs for the same accuracy.
- **EL4 (honest prior).** On BoolQ at GPT-2 scale, elicitation sits within the noise floor of the
  weak baseline (little to surface) — confirming volume-bound. SciQ may show more (more elicitable
  factual knowledge).
- **EL5.** The unsupervised CCS probe does **not** beat the k-shot supervised probe at GPT-2 scale
  (small models give CCS a weak, often non-truth direction); both are bounded by the
  full-supervised linear-probe ceiling.

## Methods (both read out a *frozen, pretrained* strong model — no fine-tuning)
The activation is the model's own classifier input: last-token hidden state (`transformer(input_ids)[0]`
at the last non-pad position), with a small **layer sweep** via `output_hidden_states`.
1. **k-shot supervised probe** (simple, robust). Fit a logistic probe on the strong model's frozen
   activations using only the **k GT labels** from train; evaluate on the test split. The clean
   "spend the budget to elicit" baseline.
2. **CCS + GT-orient** (Anthropic-style, novel here). Build contrast prompts (BoolQ: `…\nAnswer: Yes`
   vs `…\nAnswer: No`; SciQ: `…\nThe answer is correct/incorrect.`), extract φ⁺/φ⁻, run Contrastive
   Consistency Search **unsupervised** (consistency + confidence loss, N restarts, keep best by the
   unsupervised loss) to get a sign-ambiguous truth direction. Use the **k GT labels only** to fix
   the sign, pick the layer/restart, and set the threshold. No fine-tuning.

## Conditions (all at matched budget k)
- elicit-probe (M1) · elicit-ccs (M2)
- **naive mixing** at k · **GT-only** at k *(existing Phase-1 curves)*
- weak-only (frac 0) and strong-GT ceiling *(for PGR)*
- **Controls:** CCS with a random/shuffled direction (does CCS beat noise?); the k→full supervised
  probe as the **upper bound on linear readout** from these activations.

## Metrics
- **Accuracy (primary)** and **PGR** per strict (weak,strong) pair, median over pairs×seeds, vs the
  0.014 floor; `EXCLUDE` applied.
- **Sample-efficiency curve:** accuracy vs k ∈ {8, 16, 32, 64, 128, 256} for elicitation vs mixing.

## Run matrix + compute
4 strong models × 2 tasks × 3 seeds × ~3 layers × (CCS: 10 restarts). Activation extraction is
**forward-passes only** (no training); probe/CCS fitting is trivial → **near-free, a few GPU-hours,
< $20**. A fine-tuning variant (train the student on elicitation-refined labels) is a larger
follow-up *only if* M1/M2 show signal.

## What each outcome means
- **Elicitation beats mixing at small k** → reframes the negative: the budget isn't useless, it's
  mis-spent (orientation > labels). Strong positive.
- **Elicitation ≈ weak baseline** → confirms small-gap students have little latent knowledge → the
  mechanism holds, and the larger-gap (Anthropic-regime) test becomes the decisive next step.
- The full-supervised probe ceiling separates "the model doesn't know" from "the readout is weak."

## Threats / researcher-DOF controls
- CCS can find non-truth directions (Farquhar et al.) → the random-direction control + the supervised
  ceiling bound the interpretation; report the k needed to orient.
- Selection rules fixed up front: CCS restarts chosen by unsupervised loss; layer chosen by the k GT.
- Activation extraction replicates the model's own last-non-pad-token selection exactly.

## Implementation + tests (this PR)
- `scripts/elicitation/extract_activations.py` — frozen forward pass; `select_last_token_states()` is
  a pure, unit-tested function replicating the model's `input_lens-1` indexing.
- `scripts/elicitation/probe.py` — k-shot logistic probe (M1) + sign/threshold from GT.
- `scripts/elicitation/ccs.py` — CCS probe, consistency+confidence loss, GT-orient (M2).
- `scripts/elicitation/run_elicitation.py` — driver (extract → fit → eval across the k sweep).
- `scripts/elicitation/analyze_elicitation.py` — PGR/accuracy/sample-efficiency vs the floor.
- `tests/test_probe.py`, `tests/test_ccs.py`, `tests/test_extract_activations.py` — synthetic,
  CPU-only, no model download; verify the algorithmic core before any GPU spend.
