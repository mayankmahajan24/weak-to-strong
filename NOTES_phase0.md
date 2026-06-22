# Phase 0 — Code Map & Correctness Invariant

## Label flow through the codebase

### Ground-truth (GT) runs
1. `train_simple.py:main()` called with NO `weak_model_size`
2. Dataset loaded and split 50/50 at line 242: `train1_ds` (first half), `train2_ds` (second half)
3. Model trains on `train1_ds` (GT labels)
4. After training, model infers on `train2_ds` → saved as `weak_labels/` on disk
5. Saved fields per example: `gt_label` (int), `hard_label` (weak prediction), `soft_label` (weak probabilities), `logits`

### Transfer runs
1. `train_simple.py:main()` called with `weak_model_size=<name>`
2. Weak labels path constructed at line 228: `{results_folder}/{sweep_subfolder}/{weak_config_foldername}/weak_labels`
3. **Critical**: line 222 forces `weak_model_config["loss"] = "xent"` — logconf transfers still load xent GT's weak labels
4. `train1_ds = load_from_disk(weak_labels_path)` — this IS the transfer split (second half of training data)
5. `train2_ds = None` — no inference pass needed
6. Labels consumed at `weak_to_strong/train.py:119` as `ex["soft_label"]`

### Label mixing (new, Phase 0)
- Inserted between steps 4 and 5 of transfer runs
- `weak_to_strong/label_mixing.py:apply_label_mixing(ds, gt_fraction, gt_seed)`
- For selected indices: `soft_label` ← `[1 - gt_label, gt_label]`, `hard_label` ← `gt_label`
- All rows tagged with `label_source` = "gt" or "weak"
- `gt_fraction=0.0` is a no-op (adds `label_source="weak"` only) — identity with baseline

## Correctness invariant: no data leakage

The strong model trains on the **transfer split** (second half of training data). The weak model trained on the **first half**. When we mix GT labels back in, we're restoring ground truth for examples the weak model *inferred on*, not examples it trained on. The weak model's training portion is never loaded during transfer runs.

Specifically:
- GT run: `train_test_split(test_size=0.5, seed=seed)` → weak model trains on `split["train"]`, infers on `split["test"]`
- Transfer run: loads `weak_labels` = the `split["test"]` portion with weak predictions attached
- Mixing replaces `soft_label`/`hard_label` with GT for a fraction of these — same data, different labels
- No information from the weak model's training set leaks into the strong model's training

## Key config/folder name details

- `get_config_foldername(config)` at line 127 creates folder names from sorted config keys
- `gt_fraction` and `gt_seed` are added to config ONLY when `gt_fraction > 0.0`
- This means `gt_fraction=0.0` produces identical folder names to baseline (no new directories)
- `minibatch_size_per_device` is commented out of config (line 210) — doesn't affect folder names
