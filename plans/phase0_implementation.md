# Phase 0 (remaining) + Phase 1 Implementation Plan

## Context

We're extending the weak-to-strong generalization repo to study how mixing ground-truth (GT) labels with weak-teacher labels affects W2SG. Phase 0 milestones 1-2 (baseline reproduction) are done. This plan covers:
- Writing `NOTES_phase0.md` (the code map)
- Building the mixing seam (milestones 3-5)
- Setting up infrastructure for Phase 1 (GT-fraction × model-scale sweep)

## Critical Data Schema Finding

In saved weak labels (produced by `eval_model_acc` in `weak_to_strong/eval.py:50-65`):
- **`gt_label`** = original ground-truth label (int, 0 or 1)
- **`hard_label`** = weak model's **predicted** hard label (NOT ground truth!)
- **`soft_label`** = weak model's predicted probabilities (list of 2 floats)
- **`logits`** = weak model's raw logits

The mixing function must use **`gt_label`** to reconstruct GT soft labels as `[1 - gt_label, gt_label]`.

## Step 1: Write `NOTES_phase0.md`

Document the label flow with the verified schema above. Key points:
- `train_simple.py:242` splits train data 50/50 → `train1_ds` (weak model trains), `train2_ds` (inference/transfer)
- GT run: trains on `train1_ds`, runs inference on `train2_ds`, saves weak labels to disk
- Transfer run: loads weak labels from disk as `train1_ds` (this IS the transfer split). `train2_ds = None`
- Labels consumed at `weak_to_strong/train.py:119` as `ex["soft_label"]`
- **Correctness invariant**: In transfer runs, `train1_ds` is already the transfer split (second half). GT mixing draws from within this same split — no leak possible since the weak model's training portion (first half) is never loaded in transfer runs.

## Step 2: Create `weak_to_strong/label_mixing.py`

New file, ~30 lines. Single function:

```python
def apply_label_mixing(ds, gt_fraction, gt_seed, strategy="naive"):
```

Logic:
- If `gt_fraction == 0.0`: add `label_source = "weak"` to all rows, return
- Use `random.Random(gt_seed).sample(range(len(ds)), k=round(gt_fraction * len(ds)))` to select GT indices
- `ds.map(fn, with_indices=True)`: for selected indices, set `soft_label = [1 - gt_label, gt_label]`, `hard_label = gt_label`, `label_source = "gt"`; others get `label_source = "weak"`
- Assert `strategy == "naive"` (extension point for later)

## Step 3: Wire into `train_simple.py`

**Parameters** — add to `main()` signature (after `sync_command`):
```python
gt_fraction: float = 0.0,
gt_seed: int = 42,
```

**Validation** — early in `main()`:
```python
assert 0.0 <= gt_fraction <= 1.0
```

**Config dict** — only add when `gt_fraction > 0.0` (preserves backward compat, directory names unchanged for baseline):
```python
if gt_fraction > 0.0:
    config["gt_fraction"] = gt_fraction
    config["gt_seed"] = gt_seed
```
This must go before `get_config_foldername(config)` is called (line 263 in transfer branch).

**Call mixing** — after line 258 (`train1_ds = load_from_disk(weak_labels_path)`), before tokenization:
```python
from weak_to_strong.label_mixing import apply_label_mixing
train1_ds = apply_label_mixing(train1_ds, gt_fraction, gt_seed)
```

**Provenance in results** — after line 306, add mixing stats to `res_dict`:
```python
if gt_fraction > 0.0:
    gt_count = sum(1 for ex in train1_ds if ex.get("label_source") == "gt")
    res_dict["gt_fraction_actual"] = gt_count / len(train1_ds)
    res_dict["gt_fraction_requested"] = gt_fraction
    res_dict["gt_seed"] = gt_seed
```

## Step 4: Update `run_optimized.sh`

Extend `run_model()` to accept optional gt_fraction/gt_seed:
```bash
run_model() {
    local gpu=$1 ds=$2 model=$3 mbs=$4 weak_model=$5 gt_frac=${6:-} gt_seed=${7:-42}
    ...
    if [ -n "$gt_frac" ]; then
        gt_arg="--gt_fraction=$gt_frac --gt_seed=$gt_seed"
    fi
    ...
}
```

## Step 5: Milestone 3 — Identity check

Run one transfer pair (e.g. gpt2→gpt2-medium, BoolQ, xent) with `--gt_fraction=0.0`.
- Config dict won't include gt_fraction → same directory name → will load cached results if they exist
- To force a fresh run, use `--force_retrain` or a different `--sweep_subfolder`
- Verify accuracy matches existing baseline transfer run exactly

## Step 6: Milestone 4 — Ceiling check

Run same pair with `--gt_fraction=1.0`.
- **Important nuance**: this trains on the transfer split (second half) with GT labels restored. The standard GT run trains on the first half. Different data → accuracies will be in the same ballpark but NOT identical. This is expected and correct.
- Verify accuracy is significantly better than gt_fraction=0.0 and comparable to (not necessarily equal to) the GT ceiling run.

## Step 7: Milestone 5 — First real datapoint

Run `--gt_fraction=0.25`, both losses (xent, logconf), full GPT-2 family on BoolQ.
- Use `--sweep_subfolder=mix25`
- Also run 25%-GT-only control (train on just the 25% GT rows, no weak labels) — this requires a small addition to support subsampling in GT-only runs, or can be done by setting `--n_docs` appropriately
- Generate plot → `results/plots/mix25_gpt2_boolq.png`
- Write `RESULTS_phase0.md` with PGR comparison

## Step 8: GPU selection for Vast AI

Before running milestones 5+ and Phase 1, determine the optimal GPU to rent on Vast AI. Key considerations:

**Workload profile:**
- GPT-2 family models: 124M (gpt2), 355M (gpt2-medium), 774M (gpt2-large), 1.5B (gpt2-xl)
- All fit in single-GPU VRAM — no model parallelism needed (gpt2-xl is ~3GB fp16)
- Training is 295 steps × 2 epochs, batch_size=32, max_ctx=1024
- Bottleneck is throughput, not VRAM capacity

**Analysis approach:**
1. Check Vast AI current pricing for candidate GPUs (RTX 3090, RTX 4090, A100 40GB, A100 80GB, H100, L40S, B200)
2. For each, estimate $/run based on: (a) known per-step throughput for GPT-2 scale models, (b) hourly rental rate
3. Factor in multi-GPU availability — we want 4-8 GPUs on a single node for parallelism
4. Prefer GPUs where we can run ALL models without OOM at reasonable minibatch sizes

**Decision criteria (ranked):**
1. **Lowest total cost for the full sweep** (140 runs for Phase 1) — not $/hr but $/run
2. **Multi-GPU node availability** — need 4-8 GPUs on one machine
3. **Startup overhead** — avoid exotic configs with long setup times

**Likely sweet spot:** RTX 4090 or A100 40GB multi-GPU nodes. These models are small enough that the extra bandwidth/compute of H100/B200 is wasted — the per-hour cost premium won't be offset by proportionally faster training. RTX 3090s are cheapest but slower; 4090s hit the price/performance sweet spot for sub-2B parameter models.

**Action:** Before launching milestone 5, do a quick Vast AI search to confirm pricing and availability, then run a single timing test on the chosen GPU to calibrate `run_optimized.sh` minibatch sizes.

## Phase 1 setup (after Phase 0 milestones + GPU selection)

The GT-fraction sweep: 0%, 1%, 5%, 10%, 25%, 50%, 100% × full GPT-2 family × both losses on BoolQ.
- 7 fractions × 10 transfer pairs × 2 losses = 140 runs
- Each in its own sweep_subfolder (e.g. `mix01`, `mix05`, `mix10`, etc.)
- On a multi-GPU node, parallelizable to complete in ~1-2 hours total

## Files to modify

| File | Action |
|------|--------|
| `weak_to_strong/label_mixing.py` | **New** — `apply_label_mixing()` |
| `train_simple.py` | Add params, validation, config, mixing call, provenance |
| `run_optimized.sh` | Extend `run_model()` for gt_fraction |
| `NOTES_phase0.md` | **New** — code map + invariant doc |

## Files NOT modified
- `sweep.py` — kwargs pass-through already works
- `weak_to_strong/train.py` — only reads `soft_label`, ignores extra columns
- `weak_to_strong/loss.py` — untouched
- `weak_to_strong/datasets.py` — untouched
- `plot_smoke_test.py` — works as-is for baseline; Phase 1 plotting is a later concern

## Verification

1. **Identity**: `gt_fraction=0.0` produces bit-identical accuracy to baseline
2. **Ceiling**: `gt_fraction=1.0` accuracy ≈ GT ceiling (same ballpark, different data split)
3. **Provenance**: `label_source` column present in dataset, mixing stats in `results_summary.json`
4. **No regression**: existing baseline results still loadable by `plot_smoke_test.py`
