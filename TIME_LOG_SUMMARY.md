# Time Summary — Ground-Truth Mixing in Weak-to-Strong Generalization

**~25 hours of hands-on work.**

This is the executive view of [`TIME_LOG.md`](TIME_LOG.md) (the full record, sessions S1–S13). The
detailed log tracks *session wall-time* (~45–50h), much of which was GPU sweeps running in the
background; the hands-on engineering + analysis time below is the active subset (~25h).

## At a glance
| | |
|---|---|
| **Hands-on time** | **~25 hours** |
| Session wall-time (incl. background compute) | ~45–50 hours |
| GPU runs | ~1,000+ training runs (GPT-2 family) |
| Compute spend | ~$1.3k (Vast.ai, 8×H200 / H100 / A100 / B200) |
| Result | Full 3-question result set + mechanism, pre-registered, 0-failure final sweeps |

## Where the time went (by phase)
| Phase | Sessions | Hands-on | What was produced |
|---|---|---|---|
| **0 — Infrastructure + baseline** | S1–S4 | ~11h | Reproduced the W2SG baseline across the full GPT-2 family (BoolQ+SciQ, xent+logconf, 3 seeds); built + validated the GT-mixing harness (identity/ceiling checks) |
| **1 — How much? (fraction curve)** | S5–S6 | ~3h | Pre-registered P1–P6; ran the full fraction sweep; found *no knee* (P1 refuted), logconf null; retracted a single-seed "knee" headline |
| **— Variance study** | S7 | ~0.5h | 8-seed gpt2-large study → confirmed ~27% optimization instability (exclusion principled) |
| **1b — Premise gate** | S8 | ~1.5h | Power/MDE analysis; de-confounded "weak labels informative"; **allocation-null** (oracle ties random); SciQ cross-task validation |
| **— Synthesis + Phase 2 design** | S9 | ~2.5h | FINDINGS writeup; read interview brief; pre-registered Phase 2 plan + exec spec |
| **2 — Implementation** | S10 | ~4h | M1–M5 methods + loss/train plumbing; 54 unit tests; repo reorg (scripts/ by phase); driver + NOTES_phase2 pre-reg |
| **2 — Portfolio run** | S11 | ~1.5h | 270-run combination sweep (0 fail); portfolio null/negative, M3 only floor-clearer |
| **— Robustness reserve** | S12 | ~0.5h | Generated seeds 3,4 baseline (5-seed extension capability) |
| **— Mechanism experiment** | S13 | ~0.75h | Imitation-vs-correction: explained *why* allocation + combination are null |
| | | **~25h** | |

## Compute spend (itemized highlights)
| Phase | Instances | ~Cost |
|---|---|---|
| Phase 0 baseline (S1–S4) | Lambda A100, Vast B200/A100/H100 | ~$355 |
| Phase 1 + variance + 1b (S5–S8) | Vast H100/H200 | ~$160 |
| Phase 2 portfolio (S11) | 8×H200 | ~$115 |
| Seeds 3,4 baseline (S12) | 8×H200 | ~$27 |
| Mechanism experiment (S13) | 8×H200 | ~$20 |
| Provisioning / idle / setup / aborted-instance overhead | — | remainder |
| **Total (actual billed)** | | **~$1.3k** |

*Local sessions (S9, S10) used no compute. The itemized run costs above sum to ~$680; the difference
to the ~$1.3k actual is provisioning, idle/setup time, and aborted instances not captured per-run.*

## What ~25 hours produced
- **A complete, honest three-question result set** (how much / where / how) with a **mechanism** that
  explains the central negative, all cross-task (BoolQ + SciQ).
- **Pre-registration discipline** (git-anchored predictions; hits *and* misses reported) and a
  documented self-retraction (the "knee").
- **~1,000+ training runs**, the two largest sweeps (504 Phase-1, 270 Phase-2) finishing **0-failure**.
- A **reproducible repo**: per-phase scripts, 54 unit tests, drivers, and writeups from which the
  whole study rebuilds.

*Efficiency note: ~25 hands-on hours for a pre-registered, mechanism-backed, two-task result set —
most of the wall-clock was unattended GPU sweeps, not active work.*
