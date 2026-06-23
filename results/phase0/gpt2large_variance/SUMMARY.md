# gpt2-large GT variance on BoolQ — instability characterization

**Question.** Seed-1 `gpt2-large` trained on ground truth to 0.662 (below `gpt2-medium`), an
apparent anomaly. Was it a rare unlucky draw, or is `gpt2-large` GT training genuinely unstable?
(The S5 regen reran *seed 1* and got 0.662 again — but that only confirms determinism under a
fixed seed; it says nothing about cross-seed variance.) So: run 8 fresh seeds (11–18), GT
ceiling only, same config, vary only `--seed`.

**Setup.** `gpt2-large`, BoolQ, xent, lr=1e-5, bs=32, mbs=4, 2 epochs — identical to the
baseline GT runs. 8× H100, one seed per GPU. Stored: `results_summary.json` per seed here.

## Result — recurring low mode, ~27% of seeds

All 11 `gpt2-large` GT BoolQ accuracies (seeds 0–2 existing + 11–18 new), sorted:

```
0.6485  0.662  0.6913 | 0.703  0.7149  0.7183  0.7247  0.7265  0.7284  0.7336  0.7433
 (s12)   (s1)  (s15)  |  ...the high cluster (8 seeds)...
n=11   mean=0.7086   std=0.0301   range=[0.649, 0.743]
```

- **Low mode (<0.70): 3/11 seeds (27%), mean 0.667** — {s12=0.6485, s1=0.662, s15=0.6913}.
- **High mode (≥0.70): 8/11 (73%), mean 0.724.**
- **3/11 seeds fall below `gpt2-medium` GT (0.700)** — i.e., gpt2-large trains *worse than a
  smaller model* about a quarter of the time.

**Verdict: not a fluke.** seed-1's 0.662 is a representative draw from a recurring failure
mode — seed 12 (0.6485) is even lower. `gpt2-large` GT on BoolQ is **optimization-unstable**:
~1 in 4 seeds converges to a poor basin. Most likely cause is lr=1e-5 being on the low/unstable
edge for gpt2-large on BoolQ's long sequences (a higher LR might collapse the low mode — untested).

## Implications

1. **The seed-1 Phase-1 exclusion was justified**, but the better framing isn't "drop an
   outlier" — it's "the `gpt2-large` GT ceiling is bimodal, so any single draw (and any 3-seed
   mean) is unreliable." The pre-stated failed-ceiling filter (drop ceilings below the next
   smaller model) cleanly catches all 3 low-mode seeds.
2. **`gpt2-large` PGR / scale-interaction cannot be trusted at 3 seeds** — with a 27% low-mode
   rate, a 3-seed sample has ~58% chance of containing ≥1 low draw, badly distorting the mean.
   This independently supports reporting Phase-1 scale interaction as **inconclusive**.
3. Phase 1 used `gpt2-large` at seeds 0 and 2 (both high mode, 0.743/0.715) after excluding
   seed 1 — so the numbers used are from the stable mode, but the ceiling's instability should
   be stated as a caveat wherever gpt2-large appears.

**Cost:** 8 runs, ~6 min, 8× H100 SXM (~$20/hr), instance destroyed immediately. ~$8 incl. setup.
