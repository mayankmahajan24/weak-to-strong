# Phase 2 — Combination-method portfolio (RESULTS)

> **STATUS: PRELIMINARY (skeleton).** Numbers below are from the partial sweep (M1–M3 final at
> 54/54; M4 reliability near-final; **M5 gt_early_stop still running**). Finalize by re-pulling at
> ALL DONE and re-running `scripts/phase2/analyze_phase2.py` + `scripts/phase2/plot_phase2.py`.
> Pre-registration: [`../../NOTES_phase2.md`](../../NOTES_phase2.md) (anchor f3acd25, predictions frozen).

## Setup
- **Question (Axis "how to combine"):** given a fixed GT budget, does *how* you combine weak + GT
  per row beat naive mixing? (Allocation/Axis-A was killed in Phase 1b: oracle tied random.)
- **Matrix:** 5 methods × 6 strict weak<strong pairs × {0.10, 0.25, 0.50} × 3 seeds = **270 runs**, BoolQ.
- **Baseline:** Phase-1 naive mixing at the same (pair, fraction, seed), **loss-matched**
  (gt_anchored = logconf method → vs naive logconf; the rest → vs naive xent).
- **Read against:** noise floor **0.014**; `EXCLUDE = {(1, "gpt2-large")}`. Raw accuracy primary.
- **Methods:** M1 weighted (λ=4) · M2 soft_gt (ε=0.1) · M3 gt_anchored (logconf, GT rows un-blended) ·
  M4 reliability (weight weak rows by P(correct|conf)) · M5 gt_early_stop (GT = checkpoint-selection val).

## Results — median Δacc(method − naive) over valid (pair, seed)   _(PRELIMINARY)_

| method | Δ@0.10 | Δ@0.25 | Δ@0.50 | verdict |
|---|---|---|---|---|
| **gt_anchored** (vs logconf) | +0.004 | +0.011 | **+0.040** (15/15) | **clears floor @0.50** |
| soft_gt (vs xent) | +0.000 | −0.002 | +0.003 | within noise (null) |
| reliability (vs xent) | −0.003 | −0.004 | −0.006 | within noise (null) |
| weighted (vs xent) | −0.012 | −0.036 | −0.018 | **negative everywhere** |
| gt_early_stop (vs xent) | _pending_ | _pending_ | _pending_ | ⏳ |

Overlay figure: `../plots/phase2_overlay.png` (method curves vs naive xent/logconf references).

## Pre-registration scorecard (M1–M5)   _(PRELIMINARY)_

| # | method | prediction (NOTES_phase2) | outcome | hit? |
|---|---|---|---|---|
| M1 | weighted | "small + at 0.10, fading by 0.50" | **negative at every budget** (GT-upweighting hurts) | ❌ MISS (informative) |
| M2 | soft_gt | "≈ neutral / within noise" | neutral (±0.003) | ✅ HIT |
| M3 | gt_anchored | "STRONG: rescue logconf toward xent-like" | beats naive-logconf, monotone, clears floor @0.50 | ✅ HIT (mechanism) |
| M4 | reliability | "+ iff weak errors feature-predictable (uncertain)" | null/slightly negative — premise didn't pay | ◐ consistent |
| M5 | gt_early_stop | "small + (sample-efficiency)" | _pending_ | ⏳ |

## The headline + the honest caveat
**The one method pre-registered as the strong bet (M3) is the one that clears the floor**, exactly as
predicted: anchoring GT rows out of the confidence blend rescues logconf, monotonically with budget
(+0.4 → +1.1 → +4.0 pp), 15/15 pairs positive at 0.25 and 0.50. **Mechanism confirmed.**

**Caveat (do not over-claim):** gt_anchored rescues logconf *relative to naive logconf* but **does not
beat xent.** gt_anchored@0.50 median acc ≈ **0.642** vs naive-xent mixing ≈ **0.697**. So M3 fixes the
logconf pathology (clean GT rows shouldn't be diluted by the model's own predictions) — a real
*understanding* result — but the repaired logconf is still a worse method than plain xent. **No method
beats naive xent mixing.**

## Net (so far)
Portfolio is **"mostly null, one mechanism confirmed"** — matching the honest "most null" prior. No
*how-to-combine* method beats naive xent; the only positive is M3 rescuing the logconf failure mode
(below the xent baseline). Combined with Phase-1b's allocation null, this says: **at GPT-2 scale,
beyond label quantity, neither where nor how you spend GT moves the needle past naive mixing.**

## Threats / caveats
- 3-seed cells near a 0.014 floor; no multiple-comparisons correction across 5 methods × 3 fractions.
  → any claimed winner (only M3 qualifies) must be **promoted to 5 seeds** (seeds 3,4 ready) and the
  selection stated openly.
- gt_anchored's win is *within the logconf family*; framed against xent it is not a new best method.
- M1's λ=4 was an untuned default; "weighted hurts" is for that λ (a λ-sensitivity check would sharpen it).

## Decision → next
- **Promote M3 to 5 seeds** (`run_portfolio_driver.py --only=gt_anchored --seeds=3,4`) and **replicate on
  SciQ** (`--ds=sciq --only=gt_anchored`) — confirm the logconf-rescue mechanism cross-seed + cross-task.
- Everything else: report as honest negatives. _(Finalize this doc once M5 lands.)_
