# Task brief: Weak-to-strong GT-mixing — Phase 0 (baseline + infrastructure)

## Context for you, the agent

This is a fork of openai/weak-to-strong. We're extending it to study how mixing a small fraction of ground-truth (GT) labels with weak-teacher labels affects weak-to-strong generalization (W2SG). The full research program will compare many label-mixing strategies. This first step does not implement those strategies. It reproduces the published baseline and builds the seams that later strategies will plug into. Do not over-build: implement the naive mix as the one and only strategy now, but structure the code so adding a strategy later is a localized change.

## Constraints (read carefully)

- Universe is the GPT-2 family only: gpt2, gpt2-medium, gpt2-large, gpt2-xl. No other models exist for our purposes.
- Dataset: BoolQ (the repo's default config for it).
- Target mix: 25% ground truth / 75% weak labels — but make the fraction a parameter, default it to 1.0 first (see milestones).
- Don't modify model architectures in this step.
- Preserve the repo's existing file/output formats and the notebooks/Plotting.ipynb plotting path. Our plots must come out in the original format.

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

## Before writing code

Read README.md, sweep.py, train_simple.py, and the weak_to_strong/ package (especially the dataset assembly, the training loop, and loss.py).
Produce a short written map (in a file NOTES_phase0.md) of: where weak labels are loaded, the exact dataframe schema of weak labels (column names for soft/hard labels and the example id/index), where the train split is divided into the weak-model's training portion vs. the strong-model's transfer portion, and where labels feed into the loss. Do not start coding until this map is written — the correctness of everything downstream depends on it.

## The critical correctness invariant

The strong (student) model is trained on the transfer split — the portion of training data the weak model did not train on. When we inject GT labels, the GT subset must be drawn from within that same transfer split, replacing weak labels on those rows with their true labels. It must never pull in rows from the weak model's own training portion (that would leak and inflate results). Verify this explicitly and assert it in code. Call this out in NOTES_phase0.md.

## What to build

Add a label-mixing seam in the strong-model training path. Concretely:

- A parameter `--gt_fraction` (float in [0,1]) threaded from sweep.py → train_simple.py. 0.0 = pure weak labels (original W2SG behavior). 1.0 = pure GT (ceiling). Default 0.0 so existing behavior is unchanged when the flag is absent.
- A parameter `--gt_seed` (int) controlling which rows are selected as GT, so selection is reproducible and independent of the training seed.
- A single function, e.g. `apply_label_mixing(transfer_df, gt_df, gt_fraction, gt_seed, strategy="naive") -> df`, that returns the training dataframe with the chosen rows' labels swapped to GT. Implement only `strategy="naive"` now (random selection of gt_fraction of transfer rows, hard-swap their soft+hard labels to ground truth). Leave the strategy argument in the signature as the extension point — later work adds branches here, nothing else moves.
- A per-row provenance column (e.g. `label_source ∈ {"weak","gt"}`) carried through training and saved into the run's output. This is cheap now and invaluable later for the mechanism analyses (e.g. "on rows where the weak label was wrong, did the student follow GT or weak?").

## Milestones — run and verify in this order

### ~~Milestone 1 — Environment sanity~~ ✅ DONE
Smoke tests completed (sciq, gpt2/gpt2-medium, small n_docs). Smoke test results have been cleaned up.

### ~~Milestone 2 — Reproduce the published baseline~~ ✅ DONE
Full-size GPT-2-family BoolQ sweep completed for both xent and logconf losses. Plots saved as `results/plots/boolq_baseline_xent.png` and `results/plots/boolq_baseline_logconf.png`. SciQ baseline also completed (`results/plots/sciq_baseline_xent.png`, `results/plots/sciq_baseline_logconf.png`). All run data in `results/data/baseline/`.

### Milestone 3 — Mixing harness, identity check
With the new code, run `--gt_fraction=0.0` and confirm results are numerically identical (same seed) to milestone 2. This proves the seam is inert when off. A single (weak, strong) pair with one loss suffices — no need for a full sweep.

### Milestone 4 — Mixing harness, ceiling check
Run `--gt_fraction=1.0` and confirm the transfer models now match the GT-trained ceiling models. This proves the GT swap actually works. Again, a single pair suffices for verification.

### Milestone 5 — First real datapoint
Run `--gt_fraction=0.25`, both losses, full GPT-2 family, generate the plot in the exact required format → `results/plots/mix25_gpt2_boolq.png`. Also produce, on the same axes or alongside, the 25%-GT-only baseline (a model trained on just the 25% GT rows, no weak labels) as a control reference, since we need to know whether weak labels add anything on top of the GT.

## Deliverables for this step

- `NOTES_phase0.md` (the code map + the invariant verification).
- The mixing seam code, with the strategy extension point and `label_source` provenance.
- The milestone 3–5 runs, with the milestone 5 plot saved under `results/plots/`.
- A short `RESULTS_phase0.md`: the median PGR numbers for baseline, 25%-mix, and 25%-GT-only, plus a 3–4 sentence read on whether the naive mix beats both the pure-weak baseline and the GT-only control. This read is what determines which strategies we prioritize next — flag any surprises.
- Keep a running `TIME_LOG.md` with timestamped entries per milestone (this is a required deliverable for the overall project; start it now).

## Explicitly out of scope for this step

Loss reweighting, curriculum/two-stage training, GT-anchored logconf, disagreement-targeted GT allocation, label-noise injection. Do not build these yet. The point of stopping here is that the milestone-5 read tells us which of them is worth building first.
