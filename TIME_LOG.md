# Time Log

> Each entry has a `sid` (session ID) for tracking concurrent sessions.
> Format: `S<number>` (e.g. S1, S2). New sessions should pick the next available number.

## Session Index
| sid | Date | Scope | Status |
|---|---|---|---|
| S1 | 2026-06-21/22 | M2 baseline: boolq+sciq, xent+logconf + seed1 fill | Complete |
| S2 | 2026-06-21/22 | Seed 1+2 sweep: sciq+boolq, xent+logconf, GPT-2 family | Complete |

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
| S1 | **Total** | **~9h** | **~9h** | |

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
