# Time Summary — Ground-Truth Mixing in Weak-to-Strong Generalization

**~27 hours of hands-on work.**

This is the executive view of my time spent on this project. Total *wall-clock time* was
~45–50h, much of which was GPU sweeps running in the background; the hands-on engineering +
analysis time below is the active subset (approximately 27h). It counts work that didn't make the
final result set as well as work that did.

## At a glance
| | |
|---|---|
| **Hands-on time** | **~27 hours** |
| Wall-clock time (incl. background compute) | ~45–50 hours |
| GPU runs | ~1,000+ training runs (GPT-2 family) |
| Compute spend | ~$1.3k (Vast.ai, 8×H200 / H100 / A100 / B200) |
| Result | Full 3-question result set + mechanism, pre-registered, 0-failure final sweeps |

*Hours count time actually spent, including work that didn't make the final result set
(explored datasets, reproducibility re-runs, hardware shakeout, a retracted headline) — see
"What was explored but didn't make the final cut" below.*

## Where the time went (by phase)
| Phase | Hands-on | What was produced |
|---|---|---|
| **0 — Infrastructure + baseline** | ~13h | Reproduced the W2SG baseline across the full GPT-2 family (BoolQ+SciQ, xent+logconf, 3 seeds); built + validated the GT-mixing harness (identity/ceiling checks). Includes scoping work that didn't make the cut: an initial multi-dataset sweep (amazon_polarity / anthropic_hh / cosmos_qa) before restricting to BoolQ+SciQ; cross-instance reproducibility re-runs of the baseline (Lambda → B200 → A100) to fix a single canonical environment; and GPU/OOM shakeout |
| **1 — How much? (fraction curve)** | ~3h | Pre-registered P1–P6; ran the full fraction sweep; found *no knee* (P1 refuted), logconf null; retracted a single-seed "knee" headline |
| **— Variance study** | ~0.5h | 8-seed gpt2-large study → confirmed ~27% optimization instability (exclusion principled) |
| **1b — Premise gate** | ~1.5h | Power/MDE analysis; de-confounded "weak labels informative"; **allocation-null** (oracle ties random); SciQ cross-task validation |
| **— Synthesis + Phase 2 design** | ~2.5h | FINDINGS writeup; read interview brief; pre-registered Phase 2 plan + exec spec |
| **2 — Implementation** | ~4h | M1–M5 methods + loss/train plumbing; 54 unit tests; repo reorg (scripts/ by phase); driver + NOTES_phase2 pre-reg |
| **2 — Portfolio run** | ~1.5h | 270-run combination sweep (0 fail); portfolio null/negative, M3 only floor-clearer |
| **— Robustness reserve** | ~0.5h | Generated seeds 3,4 baseline (5-seed extension capability) |
| **— Mechanism experiment** | ~0.75h | Imitation-vs-correction: explained *why* allocation + combination are null |
| | **~27h** | |

## What was explored but didn't make the final cut (still counted in the hours):

- **Multi-dataset baselines (Phase 0).** The first baseline sweep targeted more of the original
  paper's tasks (amazon_polarity / anthropic_hh / cosmos_qa alongside BoolQ + SciQ) before I
  deliberately narrowed scope to the two cleanest tasks. The extra-dataset runs were generated and
  then dropped from the canonical set.
- **Cross-instance reproducibility re-runs (Phase 0).** The BoolQ baseline was run on three
  separate environments (Lambda A100 → Vast B200 → Vast A100) to pin down a single reproducible
  canonical environment; the superseded copies were discarded once the canonical one was fixed.
- **Hardware / OOM shakeout.** GPU selection and out-of-memory debugging (e.g. RTX 5090 32 GB can't
  fit gpt2-xl on long-sequence BoolQ; transformers-version memory regressions; CUDA-driver
  mismatches that silently fell back to CPU), plus several aborted instances.
- **The retracted "knee" headline (Phase 1).** A single-seed result initially read as a sharp knee
  at 25% GT; the multi-seed re-analysis refuted it and I retracted it. The analysis effort still
  happened (and the retraction is itself part of the rigor story).
- **logconf throughout.** Carried the logconf loss across every phase; it proved inert/harmful and
  was dropped at the Phase 2 gate — a negative result that cost compute and analysis time.
- **Robustness reserve (seeds 3, 4).** Generated a 5-seed extension capability that the final
  writeup keeps in reserve rather than folding into the headline 3-seed numbers.

## Compute spend (itemized highlights)
| Phase | Instances | ~Cost |
|---|---|---|
| Phase 0 baseline | Lambda A100, Vast B200/A100/H100 | ~$355 |
| Phase 1 + variance + 1b | Vast H100/H200 | ~$160 |
| Phase 2 portfolio | 8×H200 | ~$115 |
| Seeds 3,4 baseline | 8×H200 | ~$27 |
| Mechanism experiment | 8×H200 | ~$20 |
| Provisioning / idle / setup / aborted-instance overhead | — | remainder |
| **Total (actual billed)** | | **~$1.3k** |

*The synthesis and Phase 2 implementation phases were local (no compute). The itemized run costs
above sum to ~$680; the difference to the ~$1.3k actual is provisioning, idle/setup time, and
aborted instances not captured per-run.*

## What ~27 hours produced
- **A complete, honest three-question result set** (how much / where / how) with a **mechanism** that
  explains the central negative, all cross-task (BoolQ + SciQ).
- **Pre-registration discipline** (git-anchored predictions; hits *and* misses reported) and a
  documented self-retraction (the "knee").
- **~1,000+ training runs**, the two largest sweeps (504 Phase-1, 270 Phase-2) finishing **0-failure**.
- A **reproducible repo**: per-phase scripts, 54 unit tests, drivers, and writeups from which the
  whole study rebuilds.

*Efficiency note: ~27 hands-on hours for a pre-registered, mechanism-backed, two-task result set —
most of the wall-clock was unattended GPU sweeps, not active work.*
