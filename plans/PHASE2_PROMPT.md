# Phase 2 — Execution Spec (combination-method portfolio)

LLM/engineer-ready implementation spec for `plans/phase2.md`. Build in the order below;
**every method ships with a unit test and must preserve the global invariants.** Do not start
the GPU sweep until Phase A+B land, the naive-reproduction regression passes, and the
pre-registered predictions in `plans/phase2.md` are committed.

## Preconditions
- Branch off `main`. Python 3 + the pinned stack on the GPU box: `transformers==4.57.6`,
  `torch 2.5.1+cu124`, `datasets`, `fire`, `torch_optimizer`, `wandb` (`WANDB_MODE=disabled`).
- Files in scope: `weak_to_strong/loss.py`, `weak_to_strong/train.py`,
  `weak_to_strong/label_mixing.py`, `train_simple.py`. **Do not** change model code, the
  optimizer, the data split, or `eval_model_acc` semantics.
- Regression anchor (naive path must stay byte-identical): a `naive` xent run of
  `gpt2-medium←gpt2, boolq, gt_fraction=0.25, seed=1` must reproduce the Phase-0/1 value
  (≈ **0.673**) bit-for-bit; and `gt_fraction=0.0` ≡ baseline (the Phase-0 M3 identity check).

---

## GLOBAL INVARIANTS (assert in tests + code; check after every method)
1. **Naive path unchanged.** With `--combination_method=naive` (default), config foldernames,
   training math, and outputs are identical to current `main`. New flags/columns are emitted
   **only when non-default**.
2. **Effective batch size is constant (=32).** Methods change *targets/weights/loss*, never the
   number of optimizer updates, the data, the split, or grad-accumulation totals. Minibatch size
   affects memory only, never results.
3. **Determinism preserved.** Fixed `(seed, gt_seed)` ⇒ identical run (we rely on this for
   regression + reproduction). No new RNG without a fixed seed derived from `gt_seed`.
4. **GT provenance honored.** Any per-row treatment keys off `label_source ∈ {gt, weak, random}`
   (set by `label_mixing`) or a precomputed dataset column — never off row index or order.
5. **Loss reduction.** Weighted/masked losses normalize so that the all-`weak`, weight-1 case
   reduces *exactly* to `cross_entropy(logits, labels).mean()` (numerically, not just in spirit).

---

## PHASE A — Shared plumbing (enables M1, M3, M4)

### A1. Loss signatures accept-and-ignore aux kwargs
`weak_to_strong/loss.py`: add `**kwargs` to `xent_loss.__call__`, `logconf_loss_fn.__call__`,
`product_loss_fn.__call__` so the loop can always pass aux tensors without breaking existing
losses. **Invariant test:** `xent_loss()(logits, labels, step_frac=0.5, gt_mask=m, sample_weight=w)`
== `xent_loss()(logits, labels, step_frac=0.5)` for any `m,w` (xent ignores them).

### A2. Thread per-row aux through the training loop
`weak_to_strong/train.py` (the loop around line 104–127): alongside `all_labels`, collect per
minibatch row:
- `gt_mask`: bool tensor, `True` where `ex["label_source"] == "gt"`.
- `sample_weight`: float tensor, `ex.get("sample_weight", 1.0)` (new optional dataset column;
  defaults to 1.0 ⇒ no-op).
Stack to batch tensors and call `loss_fn(all_logits, all_labels, step_frac=step/nsteps,
gt_mask=all_gt_mask, sample_weight=all_sample_weight)`.
**Invariant test:** a `naive`/xent minibatch produces the same loss value with vs without the new
args (ties to Invariant 1 + 5).

### A3. Method/flag plumbing
`train_simple.py`: add `combination_method: str = "naive"` and method params
(`gt_loss_weight: float = 1.0`, `soft_gt_eps: float = 0.0`, `reliability: bool = False`,
`gt_early_stop: bool = False`). Select the loss object from `(loss, combination_method)`:

| `--loss` | `--combination_method` | loss object |
|---|---|---|
| xent | naive | `xent_loss` |
| logconf | naive | `logconf_loss_fn` |
| xent | weighted | `WeightedXentLoss(gt_weight=gt_loss_weight)` |
| logconf | gt_anchored | `GTAnchoredLogconfLoss()` |
| xent | reliability | `WeightedXentLoss` + precomputed `sample_weight` column (M4) |

Add `combination_method` (+ its non-default params) to `config` **only when ≠ naive** ⇒ distinct
foldernames, naive untouched. Suggested token: `cm={method}` (+ `glw=`, `sge=` as needed).

---

## PHASE B — Methods (each: mechanism · code · test · prediction)

### M1 — Weighted loss (GT upweighting)  [xent]
- **Mechanism:** per-row CE with weight `w_i = gt_weight` if GT else `1.0`; normalize by Σw:
  `loss = Σ_i w_i·CE_i / Σ_i w_i`. Pilot `gt_weight ∈ {2,4,8}` on 1 pair×1 frac×1 seed, fix one.
- **Code:** new `WeightedXentLoss(LossFnBase)` in `loss.py`; uses `gt_mask` from A2 (`w = 1 + (gt_weight-1)*gt_mask`), `cross_entropy(..., reduction="none")`, weighted mean.
- **Test (`tests/test_losses.py`):** (a) `gt_weight=1` ⇒ equals `xent_loss` to 1e-6 [Invariant 5];
  (b) all-GT batch ⇒ equals unweighted mean (weights cancel); (c) a hand-computed 2-row example
  with one GT row matches the closed-form weighted mean.
- **Predict:** small + at 0.10 (amplifies scarce clean signal), fading by 0.50; high `gt_weight`
  risks overfitting the GT subset (watch train-vs-test gap on GT rows).

### M2 — Soft / confidence-weighted GT targets  [xent, no loss change]
- **Mechanism:** GT rows get a softened target `[ε, 1-ε]` (label-smoothed GT) instead of one-hot,
  `soft_gt_eps=ε` (e.g. 0.1); optionally blend with the weak soft label by weak confidence.
- **Code:** branch in `label_mixing.apply_label_mixing` for GT rows when `soft_gt_eps>0`:
  `soft_label = [ε,1-ε] if gt==1 else [1-ε,ε]`, `hard_label=gt`, `label_source="gt"`.
- **Test:** GT-row soft_label equals the smoothed target; `ε=0` reproduces the current one-hot GT
  exactly [Invariant 1]; non-GT rows untouched; selection set identical to naive.
- **Predict:** ≈ neutral / slight + (reduces GT-row overconfidence; mostly regularization).

### M3 — GT-anchored logconf (rescue the logconf null)  [logconf]
- **Mechanism:** standard logconf sets `target = labels·(1-coef) + strong_preds·coef`, diluting
  *all* rows incl. GT. **Anchor:** for `gt_mask` rows set `coef_row = 0` (keep hard GT); weak rows
  keep the normal `coef`. So `target = labels·(1-coef·(~gt_mask)) + strong_preds·(coef·(~gt_mask))`.
- **Code:** new `GTAnchoredLogconfLoss(LossFnBase)` in `loss.py` — copy `logconf_loss_fn`, make
  `coef` a per-row vector zeroed on `gt_mask`, broadcast into the target blend.
- **Test:** (a) with `gt_mask` all-False ⇒ identical to `logconf_loss_fn` [regression];
  (b) with `gt_mask` all-True ⇒ pure CE on hard labels (coef→0); (c) GT rows' target == their
  `labels` row exactly.
- **Predict (strong, directional):** moves logconf from deeply negative toward xent-like — the
  Phase-1 logconf null is *fixable* by protecting clean labels. Callback to Phase-1 Result 3.

### M4 — Teacher-reliability weighting  [xent + preprocessing]
- **Mechanism:** fit `P(teacher correct | x)` on the GT subset (where we know `weak.hard==gt`),
  predict reliability `r_i` on weak rows, set `sample_weight = r_i` for weak rows, `1.0` for GT.
  Reframes strong labels as a calibration resource. Reliability model = logistic regression on a
  cheap feature (start with weak `soft_label` confidence `max(p)`; optionally pooled hidden state).
- **Code:** preprocessing fn `add_reliability_weights(ds, gt_seed)` → writes a `sample_weight`
  column; reuse `WeightedXentLoss` path (A2 reads the column). Seed the LR fit from `gt_seed`.
- **Test:** (a) weights ∈ [0,1], GT rows == 1.0; (b) deterministic given `gt_seed`; (c) on a
  synthetic set where confidence perfectly predicts correctness, high-confidence weak rows get
  `r≈1`, low-confidence `r≈0`.
- **Predict:** + iff weak errors are feature-predictable from a few hundred GT rows; genuinely
  uncertain — "interesting even if it fails."

### M5 — GT-as-early-stopping (GT as eval, not training)  [xent]
- **Mechanism:** train on **pure weak labels** (`gt_fraction` rows held out from training), use the
  GT subset as a clean validation set; pick the checkpoint with best GT-subset accuracy.
- **Code:** in `train_simple.py` when `gt_early_stop`: split the transfer set into weak-train
  (1-frac) and a GT-eval subset (frac, with true labels); pass GT-eval as `eval_ds`, set
  `eval_every` to a sane cadence; in `train.py` track best eval-acc checkpoint and return it.
  *No GT in the training loss.*
- **Test:** training set contains 0 GT-labeled rows (all `label_source=="weak"`); eval set size ==
  `round(frac·N)`; checkpoint selected == argmax GT-eval acc over evals.
- **Predict:** small + at low cost — uses the budget without ever training on it; a
  sample-efficiency angle. (Mechanically distinct; lower priority if time-constrained.)

*(Stretch: M6 relabel-and-retrain, M7 GT-guided weak-label filtering — same 54-run unit.)*

---

## IMPLEMENTATION STATUS (as of S9/S10)

| Item | Status | Test |
|---|---|---|
| A1 loss `**kwargs` | ✅ done | `tests/test_losses.py` (A1 ignore-invariants) |
| A2 loop threads `gt_mask`/`sample_weight` | ✅ done | covered via loss tests + py_compile |
| A3 `--combination_method` dispatch + tagging | ✅ done | naive byte-identical by construction |
| M1 weighted loss (`WeightedXentLoss`) | ✅ done | `test_losses.py` (Inv5, closed forms) |
| M2 soft-GT (`soft_gt_eps`) | ✅ done | `tests/test_soft_gt.py` (7/7) |
| M3 GT-anchored logconf (`GTAnchoredLogconfLoss`) | ✅ done | `test_losses.py` (regression + reconstruction) |
| M4 reliability weighting (`reliability.py`) | ✅ done | `tests/test_reliability.py` (11/11) |
| M5 GT-as-early-stopping | ✅ implemented (gated) | `tests/test_select_step.py` (pure selector 6/6); **active path needs on-box smoke run** (loop only executes in a real training run) |

M5 wiring: `train_model`/`train_and_save_model` take an optional `gt_val_ds`; when present, each
eval step scores the held-out GT subset and the run returns the test results at the best-GT-val
step (`_select_best_step`, ties→earliest). Strictly gated on `gt_val_ds is not None`, so naive/
M1–M4 paths are byte-identical. `train_simple.py` builds the held-out GT-val set (true labels),
trains on the weak remainder, and sets a ~6-eval cadence.

Local suite: 13 + 7 + 11 + 6 (+ Phase-1b 17) = **54 checks pass**; `py_compile` clean. Remaining
pre-sweep gates on the box: (1) naive-reproduction (~0.673 bit-for-bit) and (2) an M5 smoke run
(confirm it trains on 0 GT rows, selects a checkpoint, yields a sane accuracy). **All 5 methods
implemented; M5 enters the sweep only after its smoke run passes.**

## PHASE C — Pre-register, run, analyze

1. **Commit predictions** (the M1–M5 directional calls + the honest "most null" prior) to git
   before any sweep — `git rev-parse HEAD` is the pre-registration anchor.
2. **Run matrix:** per method 6 strict pairs × {0.10, 0.25, 0.50} × 3 seeds = 54;
   5 methods + ~30 HP-pilot runs ≈ **300 runs**. Naive baseline reused from Phase 1 (no re-run).
3. **Driver:** clone `scripts/phase1b/run_ab_driver.py` → `run_c2_driver.py`. Job =
   (method, pair, frac, seed); 8 concurrent on **8×H100 (preferred) or 8×H200**; outputs to
   `results/data/phase2_<method>/{fdir}/seed{N}`; clean `pytorch_model*.bin` + `results.pkl`
   per run; arm the 3h dead-man; pull slim; **destroy on completion + verify via API.**
4. **Analyze (`results/phase2/analyze_phase2.py`):** for each method, median Δacc(method−naive)
   per (pair,seed) at each fraction vs floor 0.014; the three readouts (beat-naive / left-shift /
   ceiling-raise); EXCLUDE `(1,"gpt2-large")`; exclude any degrade-to-chance method from the
   acc≈0.5 NaN gate (as with `random_labels`).

## VERIFICATION GATES (before trusting results)
- Naive regression passes (≈0.673 reproduction; `gt_fraction=0` identity).
- All unit tests green (`tests/test_losses.py`, extend `tests/test_label_mixing.py`).
- `cm=` token present only on non-naive folders; no collision with Phase-1 dirs.
- N/N runs ok, 0 unexpected NaN; `gt_fraction_actual ≈ requested` for training-on-GT methods
  (M5 exempt — it trains on 0 GT by design).

## DELIVERABLES
`results/phase2/RESULTS_phase2.md` (portfolio grid: method × {beat/left-shift/ceiling} × frac,
median Δacc ± seed range, explicit negative-results section) · the **left-shift overlay figure**
(method curves on the naive curve) · pre-registered scorecard · updated `RESEARCH_PATH.md` +
`TIME_LOG.md` (S9) + `FINDINGS.md`.
