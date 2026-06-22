# Phase 1: Fraction Curve + Scale Interaction

## Context

Phase 0 is complete. We have:
- Baseline reproduced: 2 seeds (seed0: 48 runs, seed1: 56 runs) across BoolQ + SciQ, xent + logconf, full GPT-2 family
- Mixing harness validated: `--gt_fraction`, `--gt_seed`, `--gt_only`, `--weak_labels_path` all wired into `train_simple.py`
- First datapoint: 25% mix on BoolQ seed1 shows xent PGR improved from -0.12 → +0.06; logconf remains deeply negative
- GT-only control confirms mixing > replacement (weak labels add value at 25%)

**Phase 0 decision gate result:** Naive 25% mix beats GT-only decisively (PGR -0.85 → +0.06 for xent). It crosses from negative to positive PGR vs baseline (-0.12 → +0.06), but the effect is small and single-seed. The signal is there but needs the fraction curve to characterize and multi-seed to confirm.

**Phase 1 goal:** Two marquee scientific results from the same runs:
1. **Supervision scaling curve** — PGR vs gt_fraction, with GT-only overlay
2. **Scale interaction** — does the marginal value of GT grow with student size?

Phase 0 + Phase 1 is the **must-ship core**. Everything after is depth.

## Constraints

- Universe: GPT-2 family only (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- Dataset: BoolQ (primary). SciQ deferred to headline replication in Phase 3.
- Loss: both xent and logconf (logconf is expected to stay negative; include for completeness)
- Seeds: 3 seeds minimum (seeds 0, 1, 2). We have seed1 infrastructure; need to regenerate GT weak_labels for seeds 0 and 2.
- Do not modify model architectures, loss functions, or training dynamics in this phase.

## What we have vs what we need

### Existing assets

| Asset | seed0 | seed1 | seed2 |
|-------|-------|-------|-------|
| Baseline runs (boolq, xent+logconf) | 14/14 ✅ | 28/28 ✅ | 14/14 (xent only) |
| GT weak_labels (boolq, xent) | ❌ deleted | ✅ 4 models | ❌ deleted |
| Mixing runs (gt_fraction=0.25) | — | 20/20 ✅ | — |
| GT-only control (gt_fraction=0.25) | — | 19/20 ✅ | — |

### What must be generated

1. **GT weak_labels for seeds 0 and 2**: Re-run the 4 GT models (gpt2, medium, large, xl) on BoolQ xent for each seed. ~8 runs total, ~30 min on 4x H100. These produce the `weak_labels/` directories that all transfer runs read from.

2. **Mixing runs across fractions and seeds**: the bulk of Phase 1 compute.

## Run matrix

### Fractions

`gt_fraction ∈ {0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0}`

- `0.0` = baseline (already have for seeds 0, 1; exists as `baseline/` runs)
- `0.01` = 1% GT (~47 rows from 4714 transfer split)
- `0.05` = 5% GT (~235 rows)
- `0.10` = 10% GT (~471 rows)
- `0.25` = 25% GT (~1179 rows) — already have for seed1
- `0.50` = 50% GT (~2357 rows)
- `1.0` = 100% GT = full GT on transfer split (ceiling check, already validated in Phase 0 M4)

### Per-fraction run count

Each fraction requires:
- 10 transfer pairs (4+3+2+1 from the GPT-2 family triangle) × 2 losses = **20 runs**
- Plus GT-only control at same fraction: 10 pairs × 2 losses = **20 runs** (but GT-only accuracy is identical across weak model sources since labels are discarded — so effectively **8 unique runs**: 4 models × 2 losses)

### Total run count

| Component | Runs per seed | × 3 seeds | Notes |
|-----------|--------------|-----------|-------|
| GT regeneration (seeds 0, 2) | 4 | 8 | Only need xent GT for weak_labels |
| Mixing: 5 new fractions × 20 | 100 | 300 | 0.05, 0.10, 0.25, 0.50, 1.0 |
| GT-only: 5 fractions × 8 unique | 40 | 120 | 4 models × 2 losses × 5 fractions |
| **Total new runs** | | **428** | |
| Already completed (seed1 mix25 + gt_only) | | -39 | Reuse from Phase 0 |
| Baseline 0.0 (reuse existing) | | -0 | Already in baseline/ dirs |
| **Net new runs** | | **~389** | |

### Time estimate

- Non-xl run on H100 80GB: ~5 min (gpt2, medium) to ~7 min (large)
- xl run on H100 80GB (minibatch=1): ~15 min
- GT-only runs (25% data): ~2 min (non-xl), ~5 min (xl)

On 8x H100 SXM:
- Mixing runs: ~300 runs ÷ 8 GPUs × ~7 min avg = **~4.5 hours**
- GT-only runs: ~120 runs ÷ 8 GPUs × ~3 min avg = **~0.8 hours**
- GT regeneration: ~8 runs ÷ 8 GPUs × ~10 min = **~0.2 hours**
- **Total: ~5.5 hours** on 8x H100 at ~$18.70/hr = **~$103**

On 4x H100 SXM:
- **Total: ~11 hours** at ~$8/hr = **~$88**

**Recommendation:** 8x H100 for speed. Budget ~$110 including setup/teardown overhead.

## Critical implementation details

### Seed handling

The `--seed` flag controls:
1. Dataset shuffle and train/test split (`train_simple.py:233`)
2. The 50/50 weak/transfer split (`train_simple.py:242`)
3. Random head initialization (via PyTorch global seed)

The `--gt_seed` flag controls which rows receive GT labels. **Per the standing plan, gt_seed must vary per seed.** Use `gt_seed = seed` (i.e., `--gt_seed=0` for seed 0, `--gt_seed=1` for seed 1, etc.) so that GT selection is tied to the overall seed environment.

### Weak labels path

Transfer runs auto-construct the weak_labels path from `sweep_subfolder`. Since mixing runs use a different subfolder than baseline, we must pass `--weak_labels_path` explicitly pointing to the baseline GT run's weak_labels directory. This was the lesson from Phase 0 M5.

The path pattern is:
```
results/data/baseline/seed{N}/bs=32-dn=boolq-e=2-ee=1000000-lp=0-l=xent-l={LR}-ls=cosi_anne-mc=1024-ms={WEAK_MODEL}-nd=20000-ntd=10000-o=adam-s={N}-twd=0/weak_labels
```

Where LR = 5e-05 for gpt2/gpt2-medium, 1e-05 for gpt2-large/gpt2-xl.

### Directory structure

```
results/data/
├── baseline/seed{0,1,2}/        # Existing baseline runs + GT weak_labels
├── naive_mixing/
│   ├── 005/seed{0,1,2}/         # gt_fraction=0.05
│   ├── 010/seed{0,1,2}/
│   ├── 025/seed{0,1,2}/         # seed1 already exists from Phase 0
│   ├── 050/seed{0,1,2}/
│   └── 100/seed{0,1,2}/
└── gt_only_025/seed{0,1,2}/     # GT-only controls (only need 025 for Phase 1 comparison)
```

Wait — the GT-only controls at other fractions are needed for the fraction curve overlay. Revised:

```
results/data/
├── baseline/seed{0,1,2}/
├── naive_mixing/{005,010,025,050,100}/seed{0,1,2}/
└── gt_only/{005,010,025,050,100}/seed{0,1,2}/
```

### Minibatch sizes (H100 80GB)

| Model | Mixing runs (full data) | GT-only runs (subset data) |
|-------|------------------------|---------------------------|
| gpt2 | 16 | 16 |
| gpt2-medium | 8 | 8 |
| gpt2-large | 4 | 4 |
| gpt2-xl | 1 | 1 |

### Config and folder naming

When `gt_fraction > 0.0`, the config includes `gt_fraction` and `gt_seed`, which become part of the folder name via `get_config_foldername()`. When `gt_only=True`, `gt_only` is also added. This means:
- Different fractions → different folders (correct, no collisions)
- Different seeds → different folders (correct, `s={seed}` in name)
- Mixing vs GT-only → different folders (correct, `go=1` in GT-only names)

## Milestones

### Milestone 1 — Regenerate GT weak_labels for seeds 0 and 2

Run 8 GT models (4 models × 2 seeds) on BoolQ xent. These produce weak_labels directories needed by all transfer runs.

**Verification:**
- Each of the 8 runs produces a `weak_labels/data-00000-of-00001.arrow` file
- Spot-check: GT accuracy should be in the same ballpark as existing seed1 results (±0.03)
- Pull weak_labels locally before proceeding

**Compute:** ~10 min on 8x H100 (all 8 run in parallel)

### Milestone 2 — Seed 1 remaining fractions (mixing only)

Run mixing runs for seed 1 at fractions {0.05, 0.10, 0.50, 1.0} (0.25 already done). Each fraction = 20 runs (10 pairs × 2 losses).

**Total:** 80 runs. ~1 hour on 8x H100.

**Verification:**
- `gt_fraction_actual` in results_summary.json matches requested fraction (±0.001)
- Accuracy monotonically increases with fraction for xent (not guaranteed for logconf)
- No NaN or 0.5-accuracy results

### Milestone 3 — Seeds 0 and 2 full fraction sweep (mixing)

Run mixing runs for seeds 0 and 2 at all 5 fractions. Each seed × fraction = 20 runs. Total: 200 runs.

**Compute:** ~3 hours on 8x H100.

**Verification:** Same as M2, plus cross-seed consistency: for each fraction, the 3-seed mean should be within ±0.03 of any single seed.

### Milestone 4 — GT-only controls across fractions and seeds

Run GT-only controls at each fraction for all 3 seeds. Since GT-only accuracy doesn't depend on the weak model source (labels are discarded), we only need 4 models × 2 losses = 8 unique runs per fraction per seed.

For fractions {0.05, 0.10, 0.25, 0.50, 1.0}: 5 × 8 × 3 = **120 runs** (but many will be very fast since low fractions train on few rows).

Subtract seed1 gt_fraction=0.25 (already have ~19 runs): net ~101 new runs.

**Compute:** ~0.5 hours on 8x H100.

### Milestone 5 — Noise floor establishment

Compute the **run-to-run noise floor**: for the baseline condition (gt_fraction=0.0), report the per-pair accuracy spread across 3 seeds. This is "what zero effect looks like." Any mixing delta smaller than this spread is null.

The data already exists (baseline runs for seeds 0, 1, 2). This is a local analysis step, no GPU needed.

**Deliverable:** A table in RESULTS_phase1.md:
```
Noise floor (baseline xent, 3-seed range):
  Mean per-pair range: X.XXX
  Max per-pair range: X.XXX (model_pair)
  Median per-pair range: X.XXX
```

### Milestone 6 — Generate plots and write results

All local work, no GPU.

**Plot 1: Supervision scaling curve** (`results/plots/phase1_fraction_curve.png`)
- x-axis: gt_fraction (0.0 to 1.0)
- y-axis: median PGR across valid pairs
- Lines: xent mixing (solid blue), logconf mixing (dashed red), xent GT-only (dotted blue), logconf GT-only (dotted red)
- Error bands: min-max range across 3 seeds
- Shaded region for noise floor
- Inset or annotation: knee location (if visible)

**Plot 2: Scale interaction** (`results/plots/phase1_scale_interaction.png`)
- x-axis: strong model size (gpt2 → gpt2-xl, or GT accuracy as proxy)
- y-axis: ΔPGR per unit gt_fraction (slope of PGR vs fraction, estimated from the curve)
- One line per loss function
- Error bars from seed spread
- Key question answered: does the slope increase with model size?

**Plot 3: Per-fraction accuracy grid** (`results/plots/phase1_accuracy_grid.png`)
- Reproduce the Phase 0 mix25 plot format, but one subplot per fraction
- Shows the raw accuracy data behind the PGR summary

**RESULTS_phase1.md:**
- Noise floor table
- Fraction curve data table (fraction × loss × median PGR ± range)
- Scale interaction analysis
- 3-4 sentence headline read
- Decision gate outcome for Phase 2

### Validation checks (apply after each milestone)

1. After each batch: count `results_summary.json` files, verify expected total
2. Spot-check `gt_fraction_actual` ≈ `gt_fraction` requested
3. Sanity: gt_fraction=1.0 accuracy should be near GT ceiling (but not identical — different data split)
4. Sanity: gt_fraction=0.05 should show smaller effect than 0.25
5. Before plotting: load all results into DataFrame, check for NaN/missing, verify seed × fraction × model × loss matrix is complete

---

## Implementation plan

### Step 1: Write batch runner script

Create `run_phase1.sh` that:
- Takes `SEED`, `FRACTION`, `SWEEP_TYPE` (mixing or gt_only) as parameters
- Auto-constructs `--weak_labels_path` from the baseline directory
- Runs all 10 transfer pairs × 2 losses (or 4 models × 2 losses for gt_only)
- Batches across available GPUs with correct minibatch sizes
- Cleans up model weights between runs

### Step 2: Regenerate weak_labels (Milestone 1)

On 8x H100:
```bash
# Seed 0: 4 GT models on 4 GPUs
CUDA_VISIBLE_DEVICES=0 python3 train_simple.py --model_size=gpt2 --ds_name=boolq --seed=0 ...
# ... (4 parallel runs)

# Seed 2: 4 GT models on remaining 4 GPUs (or second batch)
```

Pull `weak_labels/` directories locally. Verify arrow files exist.

### Step 3: Run seed 1 remaining fractions (Milestone 2)

```bash
for frac in 0.05 0.10 0.50 1.0; do
    bash run_phase1.sh --seed=1 --fraction=$frac --type=mixing
done
```

### Step 4: Run seeds 0 and 2 (Milestone 3)

```bash
for seed in 0 2; do
    for frac in 0.05 0.10 0.25 0.50 1.0; do
        bash run_phase1.sh --seed=$seed --fraction=$frac --type=mixing
    done
done
```

### Step 5: Run GT-only controls (Milestone 4)

```bash
for seed in 0 1 2; do
    for frac in 0.05 0.10 0.25 0.50 1.0; do
        bash run_phase1.sh --seed=$seed --fraction=$frac --type=gt_only
    done
done
```

### Step 6: Analysis and plotting (Milestones 5-6)

Local Python script that:
1. Loads all results from baseline/ + naive_mixing/ + gt_only/
2. Computes PGR for each run using per-seed GT baselines
3. Computes noise floor from baseline seed spread
4. Generates the three plots
5. Writes RESULTS_phase1.md

---

## Files to modify

| File | Action |
|------|--------|
| `run_phase1.sh` | **New** — batch runner for fraction sweep |
| `plot_phase1.py` | **New** — generates the three Phase 1 plots |
| `results/phase1/RESULTS_phase1.md` | **New** — results writeup |
| `NOTES_phase1.md` | **New** — pre-registered predictions + methodology |
| `TIME_LOG.md` | Add S4 session entries |
| `plans/phase1.md` | This file (mark milestones as done) |

**Unchanged:** `train_simple.py`, `weak_to_strong/label_mixing.py`, `weak_to_strong/train.py` — all infrastructure was built in Phase 0.

## Pre-registered predictions (write to NOTES_phase1.md before running)

1. **Fraction curve shape (xent):** monotonically increasing, concave (diminishing returns). The knee is expected around 0.10-0.25 — most of the benefit comes from correcting the worst weak-label errors, which a small GT budget captures.

2. **Fraction curve shape (logconf):** flat or slightly increasing. Logconf's confidence mechanism fights the GT signal at small fractions. May only help at gt_fraction ≥ 0.50 where GT dominates.

3. **GT-only vs mixing:** mixing beats GT-only at every fraction. Weak labels always add value because they provide coverage over the full transfer split; GT-only is data-starved.

4. **Scale interaction (xent):** larger students extract more value from GT (positive slope of ΔPGR vs size). Larger models have more capacity to leverage the cleaner signal. The smallest models (gpt2) may show no benefit because they can't even fit the GT signal in limited capacity.

5. **Scale interaction (logconf):** no clear trend. Logconf's auxiliary signal dominates at all scales for the GPT-2 family, masking the GT effect.

## Decision gate (after Phase 1)

1. **Knee location:** if the knee is at 0.05-0.10, Phase 2 strategies should be tested at that budget (strategies only matter where the naive curve hasn't saturated). If no knee (linear), test at 0.10 or 0.25.

2. **Scale interaction sign:** if ΔPGR/∂fraction increases with model size, foreground "scarce supervision scales with capability" as a headline finding. If flat or negative, this is a null result (report honestly).

3. **Logconf interaction:** if logconf shows no improvement at any fraction, drop it from Phase 2 runs to save compute (report the null, run Phase 2 xent-only).

## Out of scope

Non-naive allocation strategies (uncertainty, disagreement, diversity). Non-naive combination methods (soft GT, weighted loss, curriculum). These are Phase 2. Second-dataset replication (SciQ). This is Phase 3.
