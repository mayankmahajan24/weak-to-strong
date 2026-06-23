# Time Log

> Each entry has a `sid` (session ID) for tracking concurrent sessions.
> Format: `S<number>` (e.g. S1, S2). New sessions should pick the next available number.

## Session Index
| sid | Date | Scope | Status |
|---|---|---|---|
| S1 | 2026-06-21/22 | M2 baseline: boolq+sciq, xent+logconf + seed1 fill | Complete |
| S2 | 2026-06-21/22 | Seed 1+2 sweep: sciq+boolq, xent+logconf, GPT-2 family | Complete |
| S3 | 2026-06-22 | M3-5: mixing harness + identity/ceiling checks + 25% mix sweep | Complete |
| S4 | 2026-06-22 | Seed 2 full sweep + seed 0 GT weak labels regeneration + RESULTS_phase0 writeup | Complete |
| S5 | 2026-06-22 | Recovery: capture seed-1 Phase 1 results + analysis + pre-registration | Complete |

---

## S1 — 2026-06-21 — Milestone 2: Reproduce Published Baseline

### Milestone 1: Environment Sanity (smoke tests)
- *Completed in prior sessions*

### Milestone 2: Reproduce Published Baseline

**Lambda A100 80GB instance (ubuntu@158.101.127.190)**
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S1 | 2026-06-21 | ~17:10 | Launched Lambda instance, started sweep.py for 4 datasets |
| S1 | 2026-06-21 | ~17:25 | Killed sweep.py, switched to optimized parallel script (run_optimized.sh) |
| S1 | 2026-06-21 | ~17:28 | Fixed Pillow, relaunched with dataset-aware minibatch sizes |
| S1 | 2026-06-21 | ~17:32 | OOM on gpt2-medium boolq (minibatch=32 too large). Relaunched with conservative sizes |
| S1 | 2026-06-21 | ~17:35 | Training confirmed: 4 GT models running in parallel |
| S1 | 2026-06-21 | ~17:58 | boolq GT done. Transfers started |
| S1 | 2026-06-21 | ~18:22 | boolq transfers batch 1 done |
| S1 | 2026-06-21 | ~18:45 | boolq 14/14 complete. Pulled locally |
| S1 | 2026-06-21 | ~18:50 | sciq also 14/14 (from earlier). Generated xent-only plots |
| S1 | 2026-06-21 | ~19:25 | Lambda instance terminated. boolq + sciq xent complete |

**Vast.ai B200 instance (logconf runs)**
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S1 | 2026-06-21 | ~21:20 | Spun up 8x B200 on Vast.ai for logconf runs |
| S1 | 2026-06-21 | ~21:23 | Fixed python→python3, save_pretrained shared tensor issue |
| S1 | 2026-06-21 | ~21:28 | Fixed disk full (32GB container). Deleted model weights between runs |
| S1 | 2026-06-21 | ~21:33 | Discovered logconf transfers need xent weak_labels (hardcoded in code) |
| S1 | 2026-06-21 | ~21:38 | Uploaded xent weak_labels from local, launched logconf transfers |
| S1 | 2026-06-21 | ~21:45 | sciq logconf ground truth complete |
| S1 | 2026-06-21 | ~22:14 | sciq logconf 14/14 complete |
| S1 | 2026-06-21 | ~22:40 | boolq logconf 13/14 (missing xl←xl due to missing xent xl weak_labels) |
| S1 | 2026-06-21 | ~22:52 | Ran xent gpt2-xl GT on Vast to generate weak_labels |
| S1 | 2026-06-21 | ~23:00 | Final logconf xl←xl transfer completed. All 28 logconf runs done |
| S1 | 2026-06-21 | ~23:02 | Pulled all results locally. Generated plots with xent + logconf |

**Vast.ai A100 80GB Minnesota instance (canonical rerun)**
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S1 | 2026-06-22 | 00:17 | Spun up 8x A100 SXM4 80GB for consistent single-env baseline |
| S1 | 2026-06-22 | 00:26 | Setup complete. Discovered OOM: newer transformers (4.57.6) uses ~75GB for gpt2-xl |
| S1 | 2026-06-22 | 00:34 | GT for gpt2, gpt2-medium, gpt2-large complete. gpt2-xl OOM at minibatch=1 |
| S1 | 2026-06-22 | 00:53 | Tried max_ctx=512 for xl — failed due to subagent process conflicts |
| S1 | 2026-06-22 | 01:03 | Cleaned all processes, relaunched xl GT with max_ctx=512 |
| S1 | 2026-06-22 | 01:13 | xl GT completed (mc=1024, ran alone on GPU without conflicts) |
| S1 | 2026-06-22 | 01:23 | xent transfers complete (batch 1 + 2) |
| S1 | 2026-06-22 | 01:35 | logconf transfers batch 1 complete |
| S1 | 2026-06-22 | 01:40 | xl transfers (4 xent + 4 logconf) launched on GPUs 0-3 |
| S1 | 2026-06-22 | 02:05 | xent xl transfers complete (20/24) |
| S1 | 2026-06-22 | 02:30 | logconf xl transfers: 23/24 (last one hit disk full on save) |
| S1 | 2026-06-22 | 02:50 | Freed disk, relaunched final run (logconf xl←medium) |
| S1 | 2026-06-22 | 03:43 | 24/24 boolq baseline complete. Pulled locally |
| S1 | 2026-06-22 | 03:49 | Launched sciq full baseline (24 runs) on same A100 instance |
| S1 | 2026-06-22 | ~04:40 | sciq baseline expected complete |
| S1 | 2026-06-22 | 05:07 | sciq 24/24 complete. Pulled to seed0 |
| S1 | 2026-06-22 | 05:10 | Reorganized results: baseline/seed0 (canonical), baseline/seed1 (S2), baseline/seed2 (B200) |
| S1 | 2026-06-22 | 05:23 | Spun up H100 SXM x8 (instance 42052755) to fill 2 missing seed1 boolq logconf xl transfers |
| S1 | 2026-06-22 | 05:26 | Setup complete. CUDA driver too old for torch 2.12.1 — downgraded to torch 2.5.1+cu124 |
| S1 | 2026-06-22 | 05:26 | Launched 2 missing runs (logconf xl←large, xl←xl) on GPUs 0-1 |
| S1 | 2026-06-22 | 05:40 | Both runs complete. Pulled to seed1. seed1 now 56/56 |
| S1 | 2026-06-22 | 05:40 | H100 instance stopped. A100 instance stopped earlier at ~05:10 |

### Time Summary
| sid | Task | Wall time | GPU-hours | Notes |
|---|---|---|---|---|
| S1 | Lambda boolq+sciq xent | ~2h | ~2h (4 GPU) | Includes debugging OOM, optimizing script |
| S1 | Vast B200 logconf | ~2h | ~2h (8 GPU) | Disk issues, weak_labels upload |
| S1 | A100 canonical boolq | ~3.5h | ~3.5h (8 GPU) | OOM debugging, subagent conflicts |
| S1 | A100 canonical sciq | ~1h | ~1h (8 GPU) | Clean run |
| S1 | H100 seed1 fill (2 runs) | ~0.3h | ~0.3h (2 GPU) | Torch downgrade needed |
| S1 | **Total (M2)** | **~9h** | **~9h** | |

### Cost Summary
| sid | Instance | $/hr | Duration | Cost |
|---|---|---|---|---|
| S1 | Lambda 8x A100 80GB | ~$7/hr | ~2h | ~$14 |
| S1 | Vast 8x B200 | $33/hr | ~2h | ~$66 |
| S1 | Vast 8x A100 80GB | $10.67/hr | ~4.5h | ~$48 |
| S1 | Vast 8x RTX 5090 (aborted) | $4.80/hr | ~0.1h | ~$1 |
| S1 | Vast 8x H100 SXM (seed1 fill) | ~$18/hr | ~0.3h | ~$5 |
| S1 | **Total** | | | **~$134** |

### Lessons Learned
| sid | Lesson |
|---|---|
| S1 | Newer transformers (4.57.6) SDPA attention uses ~2x memory vs older versions. Pin version or budget for larger VRAM |
| S1 | Cross-instance runs are nearly identical when same software stack, but not guaranteed. Use single environment for canonical results |
| S1 | 32GB GPUs (RTX 5090) insufficient for gpt2-xl with transformers 4.57.6. Need 80GB+ |
| S1 | Disk space on Vast containers (32-64GB) fills quickly with model checkpoints. Delete pytorch_model.bin between phases |
| S1 | logconf ground truth is not meaningful — ground truth should always use xent. Filter from plots |
| S1 | Vast.ai H100 instances may have older CUDA drivers (12.8) incompatible with latest torch (2.12.1/CUDA 13.0). Downgrade torch or check driver before installing |

---

## S2 — 2026-06-21/22 — Seed 1 Sweep (sciq + boolq, xent + logconf)

### Setup & Infrastructure
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S2 | 2026-06-21 | ~22:30 | Parameterized run_optimized.sh with SEED, LOSS, SWEEP vars. Organized results into seed-based dirs (full_results/seed_0, seed_1) |
| S2 | 2026-06-21 | ~22:45 | Evaluated Vast.ai GPU options. Decided on 8x RTX 5090 32GB ($5.35/hr) as cheapest option |

### Vast.ai 8x RTX 5090 (instance 42021522 — aborted)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S2 | 2026-06-21 | ~23:30 | Spun up 8x RTX 5090 (New Jersey). Cloned repo, installed deps |
| S2 | 2026-06-21 | ~23:40 | Fixed python→python3, hardcoded paths. Launched seed 1 xent |
| S2 | 2026-06-21 | ~23:48 | sciq xent GT: gpt2+medium done. gpt2-large and gpt2-xl running |
| S2 | 2026-06-21 | ~23:55 | OOM on boolq: gpt2-medium (mbs=4), gpt2-large (mbs=2), gpt2-xl (mbs=1) all fail. 32GB insufficient for boolq's long sequences |
| S2 | 2026-06-22 | ~00:00 | Aborted RTX 5090. Decided to upgrade to 8x A100 80GB ($10.46/hr) |

### Vast.ai 8x A100 SXM4 80GB (instance 42024051 — xent runs)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S2 | 2026-06-22 | 00:09 | Created 8x A100 80GB instance (Minnesota, $10.46/hr) |
| S2 | 2026-06-22 | 00:12 | SSH connected, cloned repo, installed deps, patched script |
| S2 | 2026-06-22 | 00:12 | Launched seed 1 xent in tmux |
| S2 | 2026-06-22 | 00:29 | sciq xent GT done (17 min). gpt2-xl using 38GB/80GB — no OOM |
| S2 | 2026-06-22 | 00:33 | sciq xent transfers batch 1 running |
| S2 | 2026-06-22 | 00:44 | Disk full (50GB overlay). sciq xent complete (14 runs) but boolq failed |
| S2 | 2026-06-22 | 00:57 | Moved HF cache + results to /dev/shm (503GB tmpfs). Restarted clean |
| S2 | 2026-06-22 | 01:01 | Clean restart of full xent sweep |
| S2 | 2026-06-22 | 01:17 | sciq xent GT done (16 min) |
| S2 | 2026-06-22 | 01:32 | sciq xent transfers batch 1 done |
| S2 | 2026-06-22 | 01:47 | sciq xent complete (14/14, 46 min total) |
| S2 | 2026-06-22 | 02:13 | boolq xent GT done. gpt2-xl used 80GB — tight fit |
| S2 | 2026-06-22 | 02:35 | boolq xent transfers batch 1 done |
| S2 | 2026-06-22 | 02:57 | boolq xent complete (14/14). All 28 xent runs done |
| S2 | 2026-06-22 | 03:02 | Packaged lightweight tarball (111MB, no model checkpoints). Downloaded locally |

### Vast.ai 4x H100 SXM 80GB (instance 42039643 — logconf runs)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S2 | 2026-06-22 | 03:10 | Destroyed A100 instance. Created 4x H100 SXM ($8.91/hr, US) |
| S2 | 2026-06-22 | 03:18 | SSH connected, cloned repo, installed deps, uploaded xent results |
| S2 | 2026-06-22 | 03:20 | Launched logconf-only script (4-GPU batched) in tmux |
| S2 | 2026-06-22 | 03:33 | sciq logconf GT done (12 min — faster than A100's 17 min) |
| S2 | 2026-06-22 | 03:44 | sciq logconf transfers batch 1 done (11 min) |
| S2 | 2026-06-22 | 03:56 | sciq logconf transfers batch 2 done (12 min) |
| S2 | 2026-06-22 | 04:08 | sciq logconf complete (14/14, 48 min) |
| S2 | 2026-06-22 | 04:22 | boolq logconf GT done (14 min) |
| S2 | 2026-06-22 | 04:35 | boolq logconf transfers batch 1 done |
| S2 | 2026-06-22 | 04:48 | boolq logconf transfers batch 2 done |
| S2 | 2026-06-22 | 05:00 | boolq logconf complete (14/14). All 56 seed 1 runs done |
| S2 | 2026-06-22 | 05:02 | Downloaded results (209MB). Extracted to results/data/baseline/seed1/ |
| S2 | 2026-06-22 | 05:03 | Destroyed H100 instance |

### Vast.ai 8x H100 SXM 80GB (instance 42055072 — seed 2 full run, Slovenia)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S2 | 2026-06-22 | 05:30 | Attempted 4x H100 Netherlands — CUDA driver too old (560.35), training fell back to CPU |
| S2 | 2026-06-22 | 05:40 | Destroyed Netherlands instance. Created 8x H100 SXM Slovenia ($18.69/hr, driver 570) |
| S2 | 2026-06-22 | 05:46 | Launched seed 2 full sweep (56 runs: xent + logconf) |
| S2 | 2026-06-22 | 05:58 | sciq xent GT done (12 min) |
| S2 | 2026-06-22 | 06:19 | sciq xent complete (14/14, 33 min) |
| S2 | 2026-06-22 | 06:33 | boolq xent GT done |
| S2 | 2026-06-22 | 06:58 | boolq xent complete (14/14). All 28 xent runs done |
| S2 | 2026-06-22 | 06:58 | Logconf auto-started — disk full on model save (100GB overlay). gpt2 + medium GT failed |
| S2 | 2026-06-22 | 08:28 | Moved cache + results to /dev/shm. Restarted logconf clean |
| S2 | 2026-06-22 | 08:40 | sciq logconf GT done |
| S2 | 2026-06-22 | 09:01 | sciq logconf complete (14/14) |
| S2 | 2026-06-22 | 09:15 | boolq logconf GT done |
| S2 | 2026-06-22 | 09:40 | boolq logconf complete (14/14). All 56 seed 2 runs done |
| S2 | 2026-06-22 | 09:56 | Downloaded results (130MB). Extracted to results/data/baseline/seed2/ |
| S2 | 2026-06-22 | 09:57 | Destroyed H100 instance |

### Time Summary
| sid | Task | Wall time | GPU-hours | Notes |
|---|---|---|---|---|
| S2 | RTX 5090 attempt (aborted) | ~0.5h | ~0.5h (8 GPU) | OOM on 32GB for boolq |
| S2 | A100 80GB seed 1 xent | ~2h | ~2h (8 GPU) | Disk full recovery added ~15 min |
| S2 | H100 4x seed 1 logconf | 1h 40m | 1h 40m (4 GPU) | ~3x faster per-run than A100 |
| S2 | H100 8x seed 2 full sweep | ~4h | ~4h (8 GPU) | Disk full again, +1.5h idle during fix |
| S2 | **Total** | **~8h** | **~8h** | 112 runs across seed 1 + seed 2 |

### Cost Summary
| sid | Instance | $/hr | Duration | Cost |
|---|---|---|---|---|
| S2 | Vast 8x RTX 5090 (aborted) | $5.35/hr | ~0.5h | ~$3 |
| S2 | Vast 8x A100 80GB | $10.46/hr | ~2h | ~$21 |
| S2 | Vast 4x H100 SXM (seed 1 logconf) | $8.91/hr | ~1.7h | ~$15 |
| S2 | Vast 4x H100 SXM Netherlands (aborted) | $9.60/hr | ~0.2h | ~$2 |
| S2 | Vast 8x H100 SXM Slovenia (seed 2) | $18.69/hr | ~4h | ~$75 |
| S2 | **Total** | | | **~$116** |

### Lessons Learned
| sid | Lesson |
|---|---|
| S2 | RTX 5090 (32GB) cannot run gpt2-xl on boolq even with mbs=1 + gradient checkpointing. A100 80GB is the minimum for the full GPT-2 family on long-sequence datasets |
| S2 | Vast.ai overlay disk fills fast with HF cache + model checkpoints. ALWAYS move to /dev/shm before training. Request ≥200GB disk or budget for shm migration |
| S2 | H100 ~3x faster than A100 per run but ~2x $/hr — net cheaper for total job cost due to shorter wall time |
| S2 | Lightweight results transfer (no pytorch_model.bin) works well for continuing across instances — only ~100-200MB vs 80GB+ full |
| S2 | Check CUDA driver compatibility before launching. Driver 560 too old for CUDA 13.2 toolkit — caused silent CPU fallback |
| S2 | 4 GPUs is sufficient for GPT-2 family sweeps. 8 GPUs leaves half idle during gpt2-xl bottleneck runs |

---

## S3 — 2026-06-22 — Milestones 3–5: Mixing Harness + First Real Datapoint

### Milestone 3: Identity Check (gt_fraction=0.0)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S3 | 2026-06-22 | ~10:00 | Implemented label_mixing.py, wired into train_simple.py, wrote NOTES_phase0.md |
| S3 | 2026-06-22 | ~10:10 | Created 1x A100 SXM4 40GB on Vast ($0.61/hr, instance 42077799) |
| S3 | 2026-06-22 | ~10:21 | First test run — gpt2-xl OOM'd at minibatch=2 on 40GB. Reduced to minibatch=4 for gpt2-medium |
| S3 | 2026-06-22 | ~10:39 | Launched proper A/B test: baseline (no flag) vs gt_fraction=0.0, same hardware |
| S3 | 2026-06-22 | ~14:16 | Baseline run complete: 0.6546 |
| S3 | 2026-06-22 | ~14:31 | gt_fraction=0.0 run complete: 0.6528. Diff=0.0018 (GPU FP nondeterminism). Step-0 losses identical |
| S3 | 2026-06-22 | — | **Milestone 3: PASS** — mixing seam is inert when off |

### Milestone 4: Ceiling Check (gt_fraction=1.0)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S3 | 2026-06-22 | ~14:38 | Launched ceiling check (gt_fraction=1.0, gpt2-medium←gpt2, boolq, xent, seed=1) |
| S3 | 2026-06-22 | ~14:50 | Complete: accuracy=0.7424 (vs baseline 0.6546, +0.088). gt_fraction_actual=1.0 verified |
| S3 | 2026-06-22 | — | **Milestone 4: PASS** — GT labels produce significantly higher accuracy |

### Milestone 5: 25% GT Mix Sweep (gt_fraction=0.25)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S3 | 2026-06-22 | ~15:10 | Destroyed 1x A100. Created 8x H100 SXM 80GB ($18.68/hr, instance 42105019, Slovenia) |
| S3 | 2026-06-22 | ~15:14 | Setup complete. Installed torch 2.5.1+cu124, transformers 4.57.6 |
| S3 | 2026-06-22 | ~15:16 | Launched 20 mixing runs (10 pairs × 2 losses, gt_fraction=0.25) |
| S3 | 2026-06-22 | ~15:23 | Batch 1 done (6 min). 6/8 non-xl xent runs complete. gpt2-xl OOM at minibatch=2 |
| S3 | 2026-06-22 | ~15:29 | Batch 2 done. 12/20 non-xl runs complete. All xl runs OOM |
| S3 | 2026-06-22 | ~15:34 | Batch 3 done. 12/20 complete. Launched xl fixup (minibatch=1, 8 runs on 8 GPUs) |
| S3 | 2026-06-22 | ~15:52 | All 8 xl runs complete. 20/20 total. gt_fraction_actual=0.2499 verified |
| S3 | 2026-06-22 | ~15:55 | Results pulled locally to results/data/naive_mixing/025/seed1/. Instance destroyed |

### Time Summary
| sid | Task | Wall time | GPU-hours | Notes |
|---|---|---|---|---|
| S3 | Code implementation (local) | ~0.5h | 0 | label_mixing.py, train_simple.py, NOTES_phase0.md |
| S3 | M3 identity check (1x A100 40GB) | ~4h | ~4h (1 GPU) | Includes instance setup, torch downgrade, A/B test |
| S3 | M4 ceiling check (1x A100 40GB) | ~0.2h | ~0.2h (1 GPU) | Single run |
| S3 | M5 25% mix sweep (8x H100 SXM) | ~0.7h | ~0.7h (8 GPU) | 20 runs, xl needed minibatch=1 fixup |
| S3 | **Total** | **~5.5h** | **~5h** | |

### Cost Summary
| sid | Instance | $/hr | Duration | Cost |
|---|---|---|---|---|
| S3 | Vast 1x A100 SXM4 40GB | $0.61/hr | ~4.5h | ~$3 |
| S3 | Vast 8x H100 SXM 80GB | $18.68/hr | ~0.7h | ~$13 |
| S3 | **Total** | | | **~$16** |

### Lessons Learned
| sid | Lesson |
|---|---|
| S3 | gpt2-xl OOMs at minibatch=2 on 80GB H100 with transformers 4.57.6. Always use minibatch=1 for xl |
| S3 | weak_labels_path must be passed explicitly (--weak_labels_path) when sweep_subfolder differs from baseline, since auto-constructed path uses the same subfolder |
| S3 | Identity checks should run both baseline and gt_fraction=0.0 on same hardware for valid comparison. Cross-env differences (~0.002) are normal GPU FP nondeterminism |

---

## S4 — 2026-06-22 — Seed 2 Sweep + Seed 0 GT Weak Labels + Results Writeup

### Seed 2 full sweep (8x H100 SXM, Slovenia)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S4 | 2026-06-22 | ~05:30 | Attempted 4x H100 Netherlands — CUDA driver 560 too old, training fell back to CPU |
| S4 | 2026-06-22 | ~05:40 | Destroyed NL instance. Created 8x H100 SXM Slovenia ($18.69/hr, driver 570) |
| S4 | 2026-06-22 | 05:46 | Launched seed 2 full sweep (56 runs: xent + logconf, sciq + boolq) |
| S4 | 2026-06-22 | 05:58 | sciq xent GT done (12 min) |
| S4 | 2026-06-22 | 06:19 | sciq xent complete (14/14, 33 min) |
| S4 | 2026-06-22 | 06:58 | boolq xent complete (14/14). All 28 xent runs done |
| S4 | 2026-06-22 | 06:58 | Logconf auto-started — disk full on model save (100GB overlay). gpt2 + medium GT failed |
| S4 | 2026-06-22 | 08:28 | Moved cache + results to /dev/shm. Restarted logconf clean |
| S4 | 2026-06-22 | 09:01 | sciq logconf complete (14/14) |
| S4 | 2026-06-22 | 09:40 | boolq logconf complete (14/14). All 56 seed 2 runs done |
| S4 | 2026-06-22 | 09:56 | Downloaded results (130MB). Extracted to results/data/baseline/seed2/. Instance destroyed |

### RESULTS_phase0.md writeup
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S4 | 2026-06-22 | ~10:00 | Wrote RESULTS_phase0.md — full baseline analysis with 3-seed GT tables, transfer matrices, PGR comparison vs published reference, variance analysis, seed 1 gpt2-large anomaly investigation |
| S4 | 2026-06-22 | ~10:30 | Revised for research rigor — added std devs, PGR denominator instability caveat, methodological appendix |
| S4 | 2026-06-22 | ~10:45 | Committed and pushed (5b3ceb3, 101f09c) |

### Seed 0 GT weak labels regeneration (4x H100 SXM)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S4 | 2026-06-22 | ~17:25 | Created 4x H100 SXM ($11.74/hr). Launched 8 GT runs (4 models × 2 datasets, seed=0) |
| S4 | 2026-06-22 | 17:35 | sciq GT done (10 min) |
| S4 | 2026-06-22 | 17:48 | boolq GT done (13 min). All 8 GT runs with weak_labels arrow files complete |
| S4 | 2026-06-22 | ~17:50 | First instance exited unexpectedly. Spun up replacement, re-ran 8 GT runs |
| S4 | 2026-06-22 | 19:02 | Replacement instance: launched 8 GT runs |
| S4 | 2026-06-22 | 19:12 | sciq GT done |
| S4 | 2026-06-22 | 19:25 | boolq GT done. Tarball created (12MB slim: config + log + results_summary + weak_labels arrows) |
| S4 | 2026-06-22 | 19:32 | Downloaded and extracted to results/data/baseline/seed0/. Instance destroyed |

### Time Summary
| sid | Task | Wall time | GPU-hours | Notes |
|---|---|---|---|---|
| S4 | Seed 2 full sweep (8x H100) | ~4h | ~4h (8 GPU) | Disk full added ~1.5h idle |
| S4 | RESULTS_phase0 writeup | ~1h | 0 | Local only |
| S4 | Seed 0 GT weak labels (4x H100, two attempts) | ~1h | ~0.5h (4 GPU) | First instance exited, re-ran |
| S4 | **Total** | **~6h** | **~4.5h** | |

### Cost Summary
| sid | Instance | $/hr | Duration | Cost |
|---|---|---|---|---|
| S4 | Vast 4x H100 NL (aborted, driver) | $9.60/hr | ~0.2h | ~$2 |
| S4 | Vast 8x H100 SXM Slovenia (seed 2) | $18.69/hr | ~4h | ~$75 |
| S4 | Vast 4x H100 SXM (seed 0 GT, attempt 1) | $11.74/hr | ~0.5h | ~$6 |
| S4 | Vast 4x H100 SXM (seed 0 GT, attempt 2) | $11.74/hr | ~0.5h | ~$6 |
| S4 | **Total** | | | **~$89** |

### Lessons Learned
| sid | Lesson |
|---|---|
| S4 | results.pkl files are massive (up to 5.9GB for gpt2-xl). Exclude from tarballs — results_summary.json has the accuracy |
| S4 | When tarring, use find + tar -T for precise file selection. --exclude with glob patterns is unreliable for nested dirs |
| S4 | Vast instances can exit unexpectedly. Download results immediately after completion — don't leave data on ephemeral instances |

---

## S5 — 2026-06-22 — Recovery: Capture Seed-1 Phase 1 Results

**Recovery session.** A prior session launched Phase 1 seed-1 mixing + GT-only runs on
8× H200 instance `42131402` (ssh5.vast.ai:11402) but was interrupted before the results
were pulled. Instance was found still running with no active jobs; run dirs timestamped
~20:12–21:24 UTC. This session reconnected and captured the completed results.

### Capture
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S5 | 2026-06-22 | ~22:00 | **Recovery start.** Confirmed VAST_API_KEY in env; v0 CLI deprecated, used v1 API to locate running instance 42131402 (8× H200) |
| S5 | 2026-06-22 | ~22:05 | SSH'd in. No jobs running. Inventoried results: 108 results_summary.json (naive_mixing 025/050/100, gt_only 001–100) |
| S5 | 2026-06-22 | ~22:10 | rsync'd naive_mixing + gt_only excluding *.pkl (pulled config.json + log.jsonl + results_summary.json only; ~1MB JSON vs 1.2GB with pkls) |
| S5 | 2026-06-22 | ~22:15 | Verified: 050/100 pulled with 0 pkl; gt_only all 6 fractions × 8 runs present. Instance-origin data is seed1, gt_seed=1 |
| S5 | 2026-06-22 | ~22:20 | Wrote consolidate_phase1.py → phase1_seed1_results.csv (196 rows, filtered to boolq/seed1/gt_seed=1; dropped Phase 0 gt_seed=42 artifacts) |
| S5 | 2026-06-22 | ~22:25 | Wrote results/phase1/CAPTURE_seed1.md provenance manifest. xent mixing curve verified monotonic-increasing (0.657→0.743) |
| S5 | 2026-06-22 | ~22:30 | Committed + pushed capture (59ee63e): CSV + consolidate script + manifest + plan + S5 log |

### Analysis (local, no GPU)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S5 | 2026-06-22 | ~22:40 | Wrote analyze_phase1.py. Computed PGR fraction curve, mixing-vs-GT-only, scale interaction (PGR = (xfer−weak_GT)/(strong_GT−weak_GT), median over valid pairs) |
| S5 | 2026-06-22 | ~22:45 | Key reads: xent knee at 0.25 (PGR −0.21→+0.27); mixing beats GT-only at every frac<1.0 (+0.04–0.06); logconf null (flat ~0.60); scale interaction underpowered (gpt2-large GT anomaly removes large-as-student pairs) |
| S5 | 2026-06-22 | ~22:55 | Wrote robustness_phase1.py — seed-independent checks. Knee unanimous (4/4 pairs flip at 0.25), survives metric choice, 18× FP noise floor. Validation clean (gt_fraction_actual OK; gt_only=1.0 by construction; 0 degenerate runs) |
| S5 | 2026-06-22 | ~23:00 | Wrote NOTES_phase1.md pre-registration: froze pipeline (7 decisions) + 6 falsifiable predictions (P1–P6) against seed-1 only, anchored to commit 59ee63e before seeds 0/2 |
| S5 | 2026-06-22 | ~23:05 | Committed + pushed pre-registration (773b2f2): NOTES_phase1.md + analyze + robustness scripts |

### Seed 0 integration + seed-1 gpt2-large GT regeneration
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S5 | 2026-06-23 | ~00:30 | Seed-0 Phase 1 data found local (120 mixing + 48 gt_only, gs=0). Generalized consolidate to multi-seed (phase1_results.csv, 388 rows); wrote compare_seeds_phase1.py |
| S5 | 2026-06-23 | ~00:45 | Scored P1–P6 across seeds 0+1: P2/P3/P4 hold (mixing≫gt_only replicates tightly); P1 knee SHAKY (seed0 gradual, crosses @0.50 not 0.25); P5/P6 not supported. Seed0 gpt2-large GT healthy (0.743) vs seed1 anomaly (0.662) |
| S5 | 2026-06-23 | ~01:00 | Decided to regenerate seed-1 gpt2-large GT to adjudicate anomaly (transient fail vs real seed variance). Existing 8×H200 (42131402) saturated by seed-2 sweep — GPU-share attempt OOM'd (orchestrator double-booked GPU 0) |
| S5 | 2026-06-23 | ~01:15 | Provisioned dedicated GPU. A100 (42160326) → swapped to H100 SXM (42160540, $2.13/hr) for speed. rsync'd code, installed deps (transformers 4.57.6, datasets 5.0.0) |
| S5 | 2026-06-23 | ~01:30 | Debugged: mbs=16 OOM on 80GB → mbs=4; pkill -f train_simple self-killed SSH shell (cmd contained the pattern) → launched without pkill |
| S5 | 2026-06-23 | ~01:32 | GT run complete: accuracy=0.6619761394921995 — **bit-identical** to original. Verified fresh run (294 steps). weak_labels: hard labels identical (0/4714 mismatch), soft ≤0.007 diff |
| S5 | 2026-06-23 | ~01:35 | Adjudication: anomaly is real/reproducible seed variance, not corruption. Keep as-is, validity rule excludes seed-1 large-as-student pairs. No downstream reruns. Pulled slim results; destroyed H100 (42160540) + confirmed A100 gone |

### Deeper analysis + methodology decisions (local)
| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S5 | 2026-06-23 | ~02:00 | Per-pair re-examination of the "knee": median Δacc-vs-baseline + #pairs-improved. P1 (knee at 0.25) looking **refuted**, not just shaky — seed0 shows ~0 gain at 0.25 (5/10 pairs), effect only consistent at 0.50 (9/10), large only at 1.0. Curvature convex/accelerating (opposite of predicted concave/diminishing-returns). The PGR "knee" was a small-denominator amplification + seed-1 artifact |
| S5 | 2026-06-23 | ~02:05 | Decision: keep seed 1 in the {0,1,2} set. Dropping a reproducible-but-unlucky seed = outcome-dependent selection; validity rule already handles it. Recommend a with/without-seed1-large sensitivity check in the writeup |
| S5 | 2026-06-23 | ~02:10 | Audited results.pkl need: holds per-example TEST predictions only. Not required for any planned work (metrics from results_summary.json; Phase 2 allocation needs weak_labels, preserved for all 3 seeds). Regenerable exactly for accuracy/hard-labels (code+config+weak_labels preserved), ~0.007-approximate for soft_labels under env change |

### Status after S5
- Seed-1 Phase 1: mixing (M2) + GT-only controls (M4) **complete** + analyzed; pre-registration locked.
- Seed-0 Phase 1: mixing + GT-only **complete**, integrated into phase1_results.csv; scored vs P1–P6.
- seed-1 gpt2-large GT anomaly **adjudicated** (regenerated → reproduced 0.66198 bit-for-bit → real seed variance, kept). See NOTES_phase0.md.
- Still pending: seed 2 sweep (M3, in progress on 42131402), noise floor + plots (M5/M6), final 3-seed scoring of P1–P6.
- Instance `42131402` (seed-2 sweep) left **running** at user request.

### Time / Cost Summary
| sid | Task | Wall time | GPU-hours | Notes |
|---|---|---|---|---|
| S5 | Recovery capture (local + rsync) | ~0.4h | 0 | No new compute; data pull + consolidation only |
| S5 | Seed-1 analysis + robustness + pre-registration | ~0.7h | 0 | Local only |
| S5 | Seed-0 integration + P1–P6 scoring | ~0.3h | 0 | Local only |
| S5 | seed-1 gpt2-large GT regeneration (1× H100 SXM) | ~0.5h | ~0.15h (1 GPU) | A100→H100 swap, OOM/pkill debugging; run itself ~6 min |

### Cost Summary
| sid | Instance | $/hr | Duration | Cost |
|---|---|---|---|---|
| S5 | Vast 1× A100 SXM (42160326, swapped off) | $0.73/hr | ~0.05h | <$1 |
| S5 | Vast 1× H100 SXM (42160540, regen) | $2.13/hr | ~0.5h | ~$1 |
| S5 | **Total (S5 provisioned)** | | | **~$2** |

Instance 42131402 (8× H200, seed-2 sweep) billing accrues separately while left running — not counted here.

### Lessons Learned
| sid | Lesson |
|---|---|
| S5 | Bundled vastai CLI (0.3.1) hits deprecated v0 API (410 error). Use v1 endpoint directly: `curl -H "Authorization: Bearer $VAST_API_KEY" https://console.vast.ai/api/v1/instances/` |
| S5 | Pull results immediately after a sweep — an interrupted session left a fully-computed instance billing idle. rsync --exclude='*.pkl' captures all the numbers (~1MB) without the 1.2GB prediction dumps |
| S5 | macOS has no `timeout` cmd; use ssh -o ConnectTimeout instead for connection guards |
| S5 | gt_fraction_actual is 1.0 for GT-only runs by construction (weak labels discarded → 100% of used labels are GT); only compare actual-vs-requested for mixing runs |
| S5 | Pre-register predictions + freeze the pipeline against the first seed BEFORE collecting confirmation seeds, anchored to a git commit. Converts later seeds into a real out-of-sample test instead of post-hoc analysis. Seed-1 robustness (per-pair unanimity, metric-invariance, effect-vs-noise) is checkable without the other seeds |
| S5 | `pkill -f <pattern>` over SSH self-terminates the launching shell when the remote command line itself contains `<pattern>` (e.g. pkill -f train_simple while launching train_simple). Kill by specific PID, or just don't pkill if the prior process already exited |
| S5 | Don't manually slot a job onto an instance running an active multi-GPU sweep — the sweep's orchestrator grabs idle GPUs and double-books, causing OOM. Provision a dedicated box (1× H100 SXM ~$2/hr, <$2 for a single run) for isolated one-offs |
| S5 | gpt2-large BoolQ at mbs=16/mc=1024 OOMs on 80GB; mbs=4 fits (~64GB). minibatch only chunks memory — effective batch_size=32 via grad accumulation, so results are identical |
| S5 | Training is effectively deterministic under fixed seed: seed-1 gpt2-large GT reproduced bit-for-bit (0.6619761394921995) across A100→H100, mbs, and datasets 4.x→5.0.0. Re-running a suspected-bad run is a valid, cheap way to distinguish transient failure from real seed variance |

---

## S6 — 2026-06-22 — Phase 1 Seeds 0 + 2 Fraction Sweep (8x H200)

Ran the full Phase 1 fraction sweep for seeds 0 and 2 (seed 1 already complete locally),
completing the 3-seed supervision-scaling curve. Also wrote the LLM-ready Phase 1 prompt
(`results/phase1/PHASE1_PROMPT.md`) with the per-(seed,fraction,condition) data-location map.

### Run matrix (per seed: 6 fractions × (20 mixing + 8 gt_only) = 168; ×2 seeds = 336)
Fractions {0.01, 0.05, 0.10, 0.25, 0.50, 1.0}, boolq, xent+logconf, gt_seed=seed.
Weak_labels (boolq xent, 4 models/seed) already existed locally — no GT regeneration needed
(plan's "deleted" note was stale). Reused existing 8× H200 instance (id 42131402).

| sid | Date | Time (UTC) | Event |
|---|---|---|---|
| S6 | 2026-06-22 | 22:31 | Cleared stale seed-1 data on instance; staged seed0/seed2 weak_labels to /dev/shm |
| S6 | 2026-06-22 | 22:34 | Launched 336-run sweep via GPU-pool driver (8 concurrent, outputs to /dev/shm, pkl/bin cleaned per-run) |
| S6 | 2026-06-22 | ~00:19 | Seed 0 complete (168/168); pulled + verified slim results locally |
| S6 | 2026-06-22 | 00:47 | Caught disk-cleanup gap: gpt2-xl saves SHARDED weights (pytorch_model-0000N-of-*.bin) that the exact-name cleaner missed → 407 GB accrued. Purged to 763 MB, added supplemental sharded-pattern cleaner |
| S6 | 2026-06-22 | 02:21 | Sweep complete: 336/336 ok, 0 failed (~3h48m). 240 mixing + 96 gt_only |
| S6 | 2026-06-22 | 02:22 | Pulled + verified seed 2 (0 NaN, 0 fidelity violations). Instance destroyed (billing stopped) |
| S6 | 2026-06-22 | ~02:25 | Consolidated all 3 seeds → results/phase1/phase1_results.csv (582 rows) |

### Headline result — 3-seed xent PGR vs gt_fraction (mixing, weak≠strong pairs, n=13/frac)
| gf | 0.0 | 0.01 | 0.05 | 0.10 | 0.25 | 0.50 | 1.0 |
|----|-----|------|------|------|------|------|-----|
| median PGR | -0.24 | -0.21 | -0.11 | -0.21 | **+0.25** | +0.27 | +1.03 |
| median acc | 0.656 | 0.667 | 0.670 | 0.671 | 0.693 | 0.693 | 0.746 |

- **No sharp knee** (corrected after scrutiny): gradual threshold-then-convex rise. ≤0.10 flat/within-noise; crossover to positive PGR is *not* seed-robust at 0.25 (seed 0 still −0.18); first all-seed-positive fraction is 0.50; only 1.0 clearly above noise. Raw accuracy is convex (back-loaded benefit) — refutes pre-reg prediction #1 (concave/front-loaded).
- **logconf flat/null** until full GT (median PGR bounces ~0 for fractions 0.01–0.50; n=2 valid pairs after denominator-stability filter — logconf compresses model spread, so ceiling−floor denominators are tiny/negative for most pairs). Supports dropping logconf at the Phase 2 gate.

### Time / Cost Summary
| sid | Instance | $/hr | Duration | Cost |
|---|---|---|---|---|
| S6 | Vast 8× H200 (reused, id 42131402) | $26.70/hr | ~3.9h sweep | ~$104 |

### Lessons Learned
| sid | Lesson |
|---|---|
| S6 | gpt2-xl save_pretrained(safe_serialization=False) writes SHARDED weights (pytorch_model-00001-of-00002.bin), NOT pytorch_model.bin. A cleaner matching only the exact name silently misses them → multi-100GB accrual. Match `pytorch_model-*.bin` too. Watch disk *trend*, not just absolute free space |
| S6 | gt_only at gt_fraction=1.0 trains on the full transfer split (nothing filtered), so those xl runs are as slow as full-data mixing (~625s), not fast like low-fraction gt_only — they gate sweep completion |
| S6 | gt_only `gt_fraction_actual` is 1.0 by construction (dataset filtered to GT rows), so it will "violate" a naive actual≈requested check — exclude gt_only from that gate |
| S6 | Resilience pattern for long remote sweeps: on-instance finalizer snapshots slim results to persistent disk at ALL DONE (survives /dev/shm RAM loss + local internet drops); kill stays gated on a verified local copy. Pull completed seeds incrementally rather than waiting for the whole run |
