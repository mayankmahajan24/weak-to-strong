# Phase 0: Baseline + GT-mixing infrastructure

## Context

This is a fork of openai/weak-to-strong. We're extending it to study how mixing a small fraction of ground-truth (GT) labels with weak-teacher labels affects weak-to-strong generalization (W2SG). The full research program will compare many label-mixing strategies. This first step reproduces the published baseline and builds the seams that later strategies plug into. Do not over-build: implement the naive mix as the one and only strategy now, but structure the code so adding a strategy later is a localized change.

## Constraints

- Universe is the GPT-2 family only: gpt2, gpt2-medium, gpt2-large, gpt2-xl.
- Dataset: BoolQ (the repo's default config for it).
- Target mix: 25% ground truth / 75% weak labels — but make the fraction a parameter, default 0.0.
- Don't modify model architectures in this step.
- Preserve the repo's existing file/output formats and the notebooks/Plotting.ipynb plotting path.

## Directory structure

```
results/
├── plots/                 # Saved figures with descriptive names
└── data/
    └── baseline/          # Baseline sweep run outputs (sweep_subfolder="baseline")
```

- Training scripts write to `results/data/` with `--sweep_subfolder` controlling the subdirectory name (default: `"baseline"`).
- Future mixing runs use e.g. `--sweep_subfolder=mix25` → `results/data/mix25/`.
- `plot_smoke_test.py` reads from `results/data/baseline/` and writes to `results/plots/`.
- Plans live in `plans/`.

## Critical data schema

In saved weak labels (produced by `eval_model_acc` in `weak_to_strong/eval.py:50-65`):
- **`gt_label`** = original ground-truth label (int, 0 or 1)
- **`hard_label`** = weak model's **predicted** hard label (NOT ground truth!)
- **`soft_label`** = weak model's predicted probabilities (list of 2 floats)
- **`logits`** = weak model's raw logits

The mixing function must use **`gt_label`** to reconstruct GT soft labels as `[1 - gt_label, gt_label]`.

## The critical correctness invariant

The strong (student) model is trained on the transfer split — the portion of training data the weak model did not train on. `train_simple.py:242` splits train data 50/50 → `train1_ds` (weak model trains on first half), `train2_ds` (weak model infers on second half to produce weak labels). In transfer runs, the saved weak labels loaded as `train1_ds` ARE the transfer split (second half). GT mixing draws from within this same split — no leak possible since the weak model's training portion (first half) is never loaded in transfer runs. Verify this explicitly and assert it in code.

---

## Milestones

### ~~Milestone 1 — Environment sanity~~ ✅ DONE
Smoke tests completed (sciq, gpt2/gpt2-medium, small n_docs).

### ~~Milestone 2 — Reproduce the published baseline~~ ✅ DONE
Full-size GPT-2-family BoolQ sweep completed for both xent and logconf losses. Plots saved as `results/plots/boolq_baseline_xent.png`, etc. SciQ baseline also completed. All run data in `results/data/baseline/`.

### Milestone 3 — Mixing harness, identity check
Run one transfer pair (e.g. gpt2→gpt2-medium, BoolQ, xent) with `--gt_fraction=0.0`. Confirm accuracy is numerically identical to the milestone 2 baseline. This proves the seam is inert when off. A single pair suffices.

### Milestone 4 — Mixing harness, ceiling check
Run same pair with `--gt_fraction=1.0`. **Important nuance**: this trains on the transfer split (second half) with GT labels restored. The standard GT run trains on the first half. Different data → accuracies will be in the same ballpark but NOT identical. Verify accuracy is significantly better than `gt_fraction=0.0` and comparable to the GT ceiling run.

### Milestone 5 — First real datapoint
Run `--gt_fraction=0.25`, both losses, full GPT-2 family on BoolQ.
- Use `--sweep_subfolder=mix25`
- Also run 25%-GT-only control (train on just the 25% GT rows, no weak labels)
- Generate plot → `results/plots/mix25_gpt2_boolq.png`
- Write `RESULTS_phase0.md` with PGR comparison

---

## Implementation plan

### Step 1: Write `NOTES_phase0.md`

Document the label flow:
- `train_simple.py:242` splits train 50/50
- GT run: trains on first half, infers on second half, saves weak labels
- Transfer run: loads weak labels from disk as `train1_ds` (the transfer split)
- Labels consumed at `weak_to_strong/train.py:119` as `ex["soft_label"]`
- Call out the correctness invariant

### Step 2: Create `weak_to_strong/label_mixing.py`

New file, ~30 lines. Single function:

```python
def apply_label_mixing(ds, gt_fraction, gt_seed, strategy="naive"):
```

- If `gt_fraction == 0.0`: add `label_source = "weak"` to all rows, return
- Use `random.Random(gt_seed).sample(range(len(ds)), k=round(gt_fraction * len(ds)))` to select GT indices
- `ds.map(fn, with_indices=True)`: for selected indices, set `soft_label = [1 - gt_label, gt_label]`, `hard_label = gt_label`, `label_source = "gt"`; others get `label_source = "weak"`
- Assert `strategy == "naive"` (extension point for later)

### Step 3: Wire into `train_simple.py`

**Parameters** — add to `main()` signature:
```python
gt_fraction: float = 0.0,
gt_seed: int = 42,
```

**Validation**: `assert 0.0 <= gt_fraction <= 1.0`

**Config dict** — only add when `gt_fraction > 0.0` (preserves backward compat, directory names unchanged for baseline):
```python
if gt_fraction > 0.0:
    config["gt_fraction"] = gt_fraction
    config["gt_seed"] = gt_seed
```
Must go before `get_config_foldername(config)` is called (line 263 in transfer branch).

**Call mixing** — after line 258 (`train1_ds = load_from_disk(weak_labels_path)`), before tokenization:
```python
from weak_to_strong.label_mixing import apply_label_mixing
train1_ds = apply_label_mixing(train1_ds, gt_fraction, gt_seed)
```

**Provenance in results** — add mixing stats to `res_dict` after training:
```python
if gt_fraction > 0.0:
    gt_count = sum(1 for ex in train1_ds if ex.get("label_source") == "gt")
    res_dict["gt_fraction_actual"] = gt_count / len(train1_ds)
    res_dict["gt_fraction_requested"] = gt_fraction
    res_dict["gt_seed"] = gt_seed
```

### Step 4: Update `run_optimized.sh`

Extend `run_model()` to accept optional gt_fraction/gt_seed args.

### Step 5: GPU selection for Vast AI

Before running milestones 5+ and Phase 1, determine the optimal GPU to rent on Vast AI.

**Workload profile:**
- GPT-2 family: 124M → 1.5B params. All fit single-GPU VRAM easily.
- 295 steps × 2 epochs, batch_size=32, max_ctx=1024
- Bottleneck is throughput, not VRAM capacity

**Decision criteria (ranked):**
1. Lowest total cost for the full sweep (140 runs for Phase 1) — $/run, not $/hr
2. Multi-GPU node availability — need 4-8 GPUs on one machine
3. Startup overhead — avoid exotic configs with long setup times

**Likely sweet spot:** RTX 4090 or A100 40GB multi-GPU nodes. These models are small enough that H100/B200 cost premium isn't offset by proportionally faster training.

**Action:** Quick Vast AI pricing check, then a single timing test to calibrate minibatch sizes.

---

## Files to modify

| File | Action |
|------|--------|
| `weak_to_strong/label_mixing.py` | **New** — `apply_label_mixing()` |
| `train_simple.py` | Add params, validation, config, mixing call, provenance |
| `run_optimized.sh` | Extend `run_model()` for gt_fraction |
| `NOTES_phase0.md` | **New** — code map + invariant doc |

**Unchanged**: `sweep.py` (kwargs pass-through works), `weak_to_strong/train.py`, `loss.py`, `datasets.py`, `plot_smoke_test.py`.

## Deliverables

- `NOTES_phase0.md` (code map + invariant verification)
- Mixing seam code with strategy extension point and `label_source` provenance
- Milestone 3–5 runs, milestone 5 plot under `results/plots/`
- `RESULTS_phase0.md`: median PGR for baseline, 25%-mix, and 25%-GT-only, plus 3–4 sentence read
- `TIME_LOG.md` with timestamped entries per milestone

## Out of scope

Loss reweighting, curriculum/two-stage training, GT-anchored logconf, disagreement-targeted GT allocation, label-noise injection. The milestone-5 read determines which to build first.
