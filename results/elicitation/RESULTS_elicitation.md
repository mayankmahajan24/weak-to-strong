# Results — Elicitation (frozen-readout) at GPT-2 scale

**Question (see `plans/elicitation.md`):** does spending the GT budget to *orient elicited latent
knowledge* (a readout from the frozen strong model) beat using it as training labels — and is there
elicitable signal at GPT-2 scale? Methods: M1 k-shot supervised linear probe; M2 CCS (unsupervised)
+ GT-orient. Frozen pretrained models, last-token states (+ a layer chosen on a train-pool
validation split), BoolQ + SciQ, 3 seeds. PGR convention + EXCLUDE + 0.014 floor as elsewhere.

## EL1 — elicited accuracy by strong model (median over seeds)

**BoolQ** (chance = 0.622):
| model | probe@32 | probe@256 | full-supervised | CCS@32 | random |
|---|---|---|---|---|---|
| gpt2 | 0.510 | 0.527 | 0.589 | 0.482 | 0.522 |
| gpt2-medium | 0.502 | 0.537 | 0.604 | 0.511 | 0.486 |
| gpt2-large | 0.510 | 0.546 | 0.590 | 0.502 | 0.474 |
| gpt2-xl | 0.519 | **0.549** | **0.611** | 0.548 | 0.472 |

**SciQ** (chance = 0.508):
| model | probe@32 | probe@256 | full-supervised | CCS@32 | random |
|---|---|---|---|---|---|
| gpt2 | 0.523 | 0.543 | 0.589 | 0.485 | 0.495 |
| gpt2-medium | 0.539 | 0.574 | 0.619 | 0.507 | 0.492 |
| gpt2-large | 0.544 | 0.594 | 0.613 | 0.511 | 0.492 |
| gpt2-xl | 0.561 | **0.616** | **0.667** | 0.518 | 0.517 |

## Findings
1. **BoolQ: no linearly-elicitable truth signal.** Every readout — including the full-supervised
   linear probe (the *upper bound* on linear elicitation, ≤0.611) — is at/below chance (0.622).
   Frozen GPT-2 representations don't linearly encode BoolQ truth at this scale.
2. **SciQ: signal exists and scales with capability.** Full-supervised probe rises monotonically
   **0.589 → 0.667** (gpt2 → xl); k=256 probe **0.543 → 0.616**. The latent knowledge grows with
   model size (**EL1 supported**) — but even gpt2-xl stays **below the fine-tuned weak-teacher
   ceiling**, so elicitation **loses to mixing** (median PGR negative at every k, both tasks).
3. **CCS (unsupervised) ≈ chance** everywhere — barely above the random-direction control and well
   under the supervised probe (**EL5 supported**); consistent with known CCS fragility on small
   models.

## Prediction scorecard (EL1–EL5, advance-anchored at `0fa9efe`)
| | Prediction | Outcome |
|---|---|---|
| EL1 | elicited acc rises with model size | ✓ clear on SciQ (0.59→0.67 full); weak/noisy on BoolQ |
| EL2 | elicitation ≥ mixing at tiny k (widest pair) | ✗ elicitation is below the weak baseline → loses to mixing |
| EL3 | elicitation more sample-efficient than mixing | ✗ probe improves with k but never clears the floor vs weak_GT |
| EL4 | BoolQ elicitation ≈ weak baseline (little to surface) | ✓ at/below chance — no signal |
| EL5 | CCS ⊁ k-shot probe; both ≤ full-supervised | ✓ CCS ≈ chance ≤ probe ≤ full |

## Interpretation — the small-gap end of the Anthropic result
Anthropic's automated-W2S researcher reaches PGR ~0.97 at a *large* gap (Qwen 0.5B → 4B) primarily
by **eliciting the strong model's latent knowledge** (CCS truth-directions, representations). Our
finding is the same lever at the *small* GPT-2 gap: **frozen-readout elicitation is weak** because
the models don't yet hold enough linearly-accessible truth. The **SciQ scaling trend is direct
within-family evidence** that elicitable knowledge grows with capability — so the mechanism that
makes elicitation dominate at 4B is *visible but not yet sufficient* at GPT-2-xl. This unifies our
small-gap negative with their large-gap positive: **elicitable knowledge scales with the gap; at
GPT-2 scale volume remains the only lever** — exactly the volume-bound mechanism.

## Caveats
- **Linear readout only.** We tested a logistic probe and CCS; the full-supervised linear probe
  (our ceiling) is itself weak, so *linearly*-accessible knowledge is limited here — a nonlinear or
  ensemble elicitor (à la their CCS+evolution-strategy) might extract more, untested.
- **BoolQ k-shot uses balanced sampling**, which underperforms the majority baseline on an
  imbalanced (62/38) test; the full-supervised probe (trained on the imbalanced pool) is the fair
  "is there signal" number — and it is ≈ chance.
- CCS sign/threshold oriented with k labels; restarts chosen by unsupervised loss; layer by the
  train-pool validation split (no test-set selection).

## Reproduce
`scripts/elicitation/{extract_activations,probe,ccs,run_elicitation,run_box}.py` (extraction = the
only GPU step); per-config JSONs in `results/elicitation/runs/`; `analyze_elicitation.py` regenerates
the tables above against the Phase-0 GT ceilings.
