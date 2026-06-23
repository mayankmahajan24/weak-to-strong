# Phase 1 — LLM-Ready Execution Prompt

This is a self-contained operational prompt derived from `plans/phase1.md`. Hand it to
an agent (with repo + Vast API access) to execute the Phase 1 fraction sweep for any
seed. It encodes the exact commands, the directory map, and the verification gates.

---

## Role

You are running the **supervision-scaling fraction sweep** for the weak-to-strong
ground-truth-mixing study. The infrastructure (`train_simple.py`,
`weak_to_strong/label_mixing.py`) was built and validated in Phase 0. **Do not modify
training code, losses, or model architectures.** Your job is to launch runs, save the
*key* results (not the multi-GB pickles), and place them in the canonical layout.

## Universe (fixed)

- **Models:** GPT-2 family only — `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- **Dataset:** `boolq` only (SciQ is Phase 3)
- **Losses:** both `xent` and `logconf`
- **Seeds:** 0, 1, 2. (Seed 1 is already complete; this prompt is parameterized by seed.)
- **Fractions:** `gt_fraction ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 1.0}` (6 points). `0.0` is the
  baseline and already exists under `results/data/baseline/seed{N}/`.

## Per-seed run matrix

For each seed, for each of the 6 fractions:

| Condition | Runs | What |
|---|---|---|
| **mixing** (`naive_mixing/`) | 20 | 10 transfer pairs × 2 losses. Weak labels + `gt_fraction` GT mixed in. |
| **gt_only** (`gt_only/`) | 8 | 4 strong models × 2 losses. Weak labels discarded, train on GT subset only. |

- 10 transfer pairs = the GPT-2 triangle (weak ≤ strong):
  `(gpt2→{gpt2,medium,large,xl})`, `(medium→{medium,large,xl})`, `(large→{large,xl})`, `(xl→xl)`.
- gt_only accuracy is independent of the weak source (labels are discarded), so only the
  **strong** model varies → 8 unique runs. Use the `gpt2` weak_labels dir as the loader
  source for all of them (matches the seed-1 convention: every gt_only folder is `wms=gpt2`).

**Per seed: 6 × (20 + 8) = 168 runs. Two seeds (0 and 2) = 336 runs.**

## Critical knobs (must match exactly)

- `--gt_seed = --seed` (GT-row selection is tied to the seed environment). Seed-1 used
  `gt_seed=1`; the old `gt_seed=42` Phase-0 artifacts are stale and filtered out downstream.
- **Do not pass `--lr`.** Let it default per model so the `l=` token in the folder name
  matches the baseline: `gpt2`/`gpt2-medium` → `5e-05`, `gpt2-large`/`gpt2-xl` → `1e-05`.
- Pass weak labels with `--weak_labels_path` pointing at the *baseline GT* run's
  `weak_labels/` dir (lesson from Phase 0 M5 — the auto-constructed path won't resolve
  because the mixing run lives in a different `sweep_subfolder`).
- Fraction string passed to `--gt_fraction` must `str()` to the folder token: pass
  `0.01, 0.05, 0.1, 0.25, 0.5, 1.0` → folders `gf=0.01/0.05/0.1/0.25/0.5/1.0`.

### Weak-labels path pattern

```
results/data/baseline/seed{N}/bs=32-dn=boolq-e=2-ee=1000000-lp=0-l=xent-l={LR}-ls=cosi_anne-mc=1024-ms={WEAK}-nd=20000-ntd=10000-o=adam-s={N}-twd=0/weak_labels
```
`LR = 5e-05` for weak `gpt2`/`gpt2-medium`, `1e-05` for `gpt2-large`/`gpt2-xl`.
These weak_labels (boolq xent, 4 models) **already exist locally** for all of seeds 0/1/2
(the plan's "deleted" note is stale) — upload them to the GPU box; no regeneration needed.

### Canonical command — mixing run

```bash
python train_simple.py \
  --model_size={STRONG} --ds_name=boolq --loss={xent|logconf} \
  --seed={N} --gt_seed={N} --gt_fraction={FRAC} \
  --minibatch_size_per_device={MBS} \
  --results_folder={OUT}/results/data \
  --sweep_subfolder=naive_mixing/{FDIR}/seed{N} \
  --weak_labels_path={OUT}/results/data/baseline/seed{N}/{WEAK_DIR}/weak_labels
```

### Canonical command — gt_only run (add `--gt_only=True`, weak source = gpt2)

```bash
python train_simple.py \
  --model_size={STRONG} --ds_name=boolq --loss={xent|logconf} \
  --seed={N} --gt_seed={N} --gt_fraction={FRAC} --gt_only=True \
  --minibatch_size_per_device={MBS} \
  --results_folder={OUT}/results/data \
  --sweep_subfolder=gt_only/{FDIR}/seed{N} \
  --weak_labels_path={OUT}/results/data/baseline/seed{N}/{GPT2_WEAK_DIR}/weak_labels
```

`FDIR` = zero-padded fraction dir (below). `MBS` per model on H100/H200:
`gpt2=16, gpt2-medium=8, gpt2-large=4, gpt2-xl=2` (minibatch only affects speed, not results).

## Where results go — data-location map

Fraction → directory token:

| `gt_fraction` | dir (`FDIR`) | `gf=` token |
|---|---|---|
| 0.01 | `001` | `gf=0.01` |
| 0.05 | `005` | `gf=0.05` |
| 0.10 | `010` | `gf=0.1`  |
| 0.25 | `025` | `gf=0.25` |
| 0.50 | `050` | `gf=0.5`  |
| 1.00 | `100` | `gf=1.0`  |

Final canonical layout (slim — JSON + log only, **no `results.pkl`, no model weights**):

```
results/data/
├── baseline/
│   ├── seed0/   ← exists (incl. boolq xent weak_labels for 4 models)
│   ├── seed1/   ← exists
│   └── seed2/   ← exists (incl. boolq xent weak_labels for 4 models)
├── naive_mixing/
│   ├── 001/  seed0/  seed1/  seed2/      # gt_fraction=0.01, 20 dirs each
│   ├── 005/  seed0/  seed1/  seed2/      # 0.05
│   ├── 010/  seed0/  seed1/  seed2/      # 0.10
│   ├── 025/  seed0/  seed1/  seed2/      # 0.25
│   ├── 050/  seed0/  seed1/  seed2/      # 0.50
│   └── 100/  seed0/  seed1/  seed2/      # 1.00
└── gt_only/
    ├── 001/  seed0/  seed1/  seed2/      # 8 dirs each
    ├── 005/  seed0/  seed1/  seed2/
    ├── 010/  seed0/  seed1/  seed2/
    ├── 025/  seed0/  seed1/  seed2/
    ├── 050/  seed0/  seed1/  seed2/
    └── 100/  seed0/  seed1/  seed2/
```

Each leaf run dir is named by `get_config_foldername()` and contains exactly:
`config.json`, `results_summary.json`, `log.jsonl`. The accuracy lives in
`results_summary.json` (`accuracy`, plus `gt_fraction_actual` / `gt_fraction_requested` /
`gt_seed` for mixing/gt_only). **This is all that's needed for PGR, plotting, and later phases.**

**This task delivers seeds 0 and 2** (seed 1 already present locally):
- `naive_mixing/{001,005,010,025,050,100}/{seed0,seed2}/`  — 20 dirs each = 240 dirs
- `gt_only/{001,005,010,025,050,100}/{seed0,seed2}/`        — 8 dirs each = 96 dirs

## Execution recipe

1. Provision / reuse an 8-GPU box (H100/H200). Confirm CUDA works (`torch.cuda.is_available()`,
   `device_count()==8`) and the repo + `weak_to_strong/label_mixing.py` are present.
2. Upload the boolq-xent `weak_labels/` (+ sibling `config.json`) for the target seeds into
   `results/data/baseline/seed{N}/...`.
3. Write run outputs to fast scratch (`/dev/shm`) and delete `results.pkl` +
   `pytorch_model.bin` after **every** run — pickles reach ~6 GB (gpt2-xl) and will fill disk.
4. Run 8 jobs concurrently, one per GPU (`CUDA_VISIBLE_DEVICES`), pulling from the job list.
   `gpt2-xl` runs single-GPU here (model_parallel is off when per-GPU memory > 35 GB).
5. On finish, `rsync`/`scp` only the slim JSON+log tree down into `results/data/...`.

## Verification gates (run after the sweep)

1. **Counts:** `naive_mixing/*/seed{0,2}` = 20 `results_summary.json` each;
   `gt_only/*/seed{0,2}` = 8 each. Totals: 240 mixing + 96 gt_only = 336.
2. **Fraction fidelity:** every mixing summary's `gt_fraction_actual ≈ requested` (±0.001).
3. **No degenerate runs:** no NaN, no exact-0.5 accuracies.
4. **Monotonicity (soft):** for xent, accuracy should trend up with fraction (not guaranteed
   for logconf). `gt_fraction=1.0` should sit near the GT ceiling (not identical — different split).
5. **Cross-seed sanity:** per fraction, the 3-seed mean within ±0.03 of any single seed.

## Downstream (local, no GPU) — Milestones 5–6 of the plan

- **Noise floor (M5):** per-pair baseline (`gt_fraction=0.0`) accuracy spread across the 3
  seeds — "what zero effect looks like."
- **Consolidate:** extend `results/phase1/consolidate_phase1.py` to all seeds → one CSV.
- **Plots (M6):** `phase1_fraction_curve.png`, `phase1_scale_interaction.png`,
  `phase1_accuracy_grid.png`. Write `results/phase1/RESULTS_phase1.md` (noise-floor table,
  fraction-curve table, scale-interaction read, decision gate for Phase 2).
