# Weak-to-Strong GT-Mixing: Phased Experimental Plan (Execution-Ready)

This plan is written to be executed in stages by an LLM coding agent. Each phase is self-contained: it states its goal, what to build, the exact runs to launch, the deliverables, and a decision gate that determines what the next stage should prioritize. Hand the agent one phase at a time. Do not let it run ahead into later phases — the gates exist because results reshape priorities.

---

## Standing context (applies to all phases — keep in the agent's context every stage)

Fork of `openai/weak-to-strong`. Goal: study how mixing a small fraction of ground-truth (GT) labels with weak-teacher labels affects weak-to-strong generalization (W2SG), and how that interacts with student scale, allocation, and combination method.

- **Universe:** the GPT-2 family only — `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`. No other models exist for this project.
- **Primary dataset:** BoolQ (required deliverable). **Secondary dataset:** one additional task (e.g. SciQ or CosmosQA), used for *targeted replication of headline findings only*, not parallel full sweeps (see Compute Doctrine).
- **Required figure:** the W2SG sweep restricted to the GPT-2 series, original repo format, with the median-PGR inset, via `notebooks/Plotting.ipynb`. Every result plot must match this format.
- Maintain `TIME_LOG.md` (timestamped, per-milestone), `NOTES_*.md` (code maps + pre-registered predictions), and `RESULTS_*.md` (numbers + reads) per phase. The time log is a required project deliverable.

### The correctness invariant (assert in code, every phase)

The strong student trains on the *transfer split* — data the weak model did **not** train on. Any injected GT must be drawn from within that transfer split, never from the weak model's own training portion. Add an explicit assertion; a violation silently inflates results.

### The seed environment (decide once, in Phase 0, then hold fixed)

Seeds replicate *within* a condition; they do not provide *coverage across* conditions. Every new `gt_fraction` or strategy is a new condition that must be run inside the established seed environment. For each artifact, decide and record in a table whether it is **fixed apparatus** (cached once, reused everywhere) or a **reseeded variance source** (one instance per seed):

- Weak model + weak labels: **reseeded** for headline comparisons (one weak model per seed), so error bars capture the dominant variance source. May be **fixed** (cached from seed 0) for purely descriptive curves — but if so, state that bands are conditional on one weak supervisor and do NOT claim they capture weak-model variance.
- GT ceiling models: **reseeded** (one per seed) — they are the PGR denominator and must carry the same noise as the rest.
- `gt_seed` (which examples receive GT labels): **always varies per seed**, in every case. For allocation studies this is often the single largest noise source; never hold it fixed across "seeds."
- Student training randomness (head init, data order): **always varies per seed.**

Write this artifact × (fixed | reseeded) table into `NOTES_phase0.md`. It is also a rigor slide in the final deck.

### Compute doctrine (the binding constraint — ~20 hrs, GPT-2-xl is the long pole, baseline PGR near zero so effects are noise-dominated)

Price every condition before launching it. One condition (one `gt_fraction` or one strategy, one dataset) ≈ 10 transfer runs (the GPT-2 family pairs) × 2 losses × 3 seeds ≈ **60 runs**. Across both datasets that doubles. This arithmetic, not enthusiasm, sets breadth.

Priority order when compute is tight (cut from the bottom):
1. **Statistical power beats breadth.** Fewer conditions at adequate seeds > many conditions at one seed. On this benchmark an unreplicated "win" is not a result. Breadth gets cut before seeds do.
2. **Seed tiering:** descriptive curves (fraction sweep, scale interaction) at **3 seeds**; the 1–2 head-to-head claims you will actually rank at **5 seeds**. Report any difference smaller than the noise floor as *null*, not as a ranking.
3. **Second dataset = replication of the headline only**, not a parallel sweep. "The winning finding reproduces on a second task" buys most of the credibility of a full second sweep at a fraction of the cost.
4. **Develop/debug at small `n_docs` on `gpt2`/`gpt2-medium`**; launch full-size xl runs only when confident, batched and unattended so wall-clock isn't person-hours.
5. **Budget 2–3 hrs for plotting/results-wrangling.** It always overruns.

### Standing rigor practices (bake in from run #1 — cannot be retrofitted)

- **Establish a noise floor first.** Run one identical condition across your seeds and report that spread as "this is what zero effect looks like here." Every later comparison is read against it. This single number does more for perceived rigor than any seed count.
- **Multiple seeds + variance on every reported number**, per the tiering above. State explicitly when a difference is within noise.
- **Pre-register each idea.** Before running a strategy, write its hypothesis and expected direction in `NOTES_*.md`. Report hits and misses honestly. A failed idea with a recorded prediction and a post-hoc explanation is worth more than an unexplained success — and directly serves the "even if many fail" grading criterion.
- **Hold confounds constant.** When comparing strategies, match GT count, total labels seen, optimizer updates, epochs, and token count. State what's held constant in each comparison.
- **At least one sanity-inverter** (an experiment designed to fail in a predictable direction — see Phase 3). If it fails as predicted, the mechanism is real; if not, there's a confound.
- **Generalization checks, not just in-distribution accuracy.** Report the weak-disagreement subset separately (the paper's diagnostic), and watch the train-vs-test gap on GT rows to catch overfitting of the tiny GT subset.

### Time guardrails

- **Must-ship core = Phase 0 + Phase 1.** A complete story on its own. Everything after is depth added until time runs out; an overrun trims depth, never the narrative.
- **10-hour checkpoint** (record in `TIME_LOG.md`): if Phase 0 + Phase 1 aren't done and plotted, stop adding conditions and write up what exists.
- **Minimum viable deck** always presentable: motivation → baseline → fraction curve → scale-interaction → GT-only-control insight → threats → next steps.

---

## Phase 0 — Infrastructure, reproduction, and the seed environment

**Goal:** a trustworthy baseline + the reusable spine + a validated seed framework, before innovating.

### Build

- `--gt_fraction` (float [0,1], default `0.0`) and `--gt_seed` (int), threaded `sweep.py` → `train_simple.py`.
- `apply_label_mixing(transfer_df, gt_df, gt_fraction, gt_seed, strategy="naive") -> df`. Implement `strategy="naive"` only (random selection, hard-swap soft+hard labels to GT). Leave `strategy` as the sole extension point — later phases add branches here and nowhere else.
- A `label_source ∈ {"weak","gt"}` provenance column carried through training and saved per run (needed for the Phase 3 imitation analysis).
- Config-driven runs + automatic plotting wired to `notebooks/Plotting.ipynb` so every later condition produces an identical-format figure with the median-PGR inset.

### Cache the shared spine (once)

Train the GT ceiling models and generate the weak labels; persist to disk; all later runs **load** rather than recompute. **Save full soft predictions / logits, not just hard labels** — allocation and blending strategies need per-example weak confidence, and regenerating it across the family later is expensive.

### Reconcile the run footprint before trusting it

The reproduction footprint is 2 datasets × 2 losses × (4 GT + 10 transfer) = 56 runs/seed × 3 seeds = 168. **Check the GT-under-logconf runs:** GT ceiling models are trained on clean labels, where the logconf confidence loss has nothing to bootstrap against, so the 4 GT × logconf runs/seed are likely redundant with the 4 GT × xent runs. Confirm this. If redundant, drop them (~24 runs reclaimed) and reinvest in an extra fraction or seed. If NOT redundant, document why — it's a repo subtlety you must understand before trusting any ceiling.

### Milestones (compute-ordered)

1. **Environment sanity:** install; run the unmodified sweep at small `n_docs` on the GPT-2 family / BoolQ; confirm completion. (README warns the repo is "not well tested" — budget friction.)
2. **Cache spine:** train GT ceilings + generate/persist weak labels (with full soft predictions), per the seed environment table.
3. **Small-scale correctness bracket** (both must pass before any full-size run): identity (`gt_fraction=0.0` ⇒ bit-identical to unmodified, same seed) **and** ceiling (`gt_fraction=1.0` ⇒ matches cached GT models).
4. **Full-size reproduction**, new code at `gt_fraction=0.0`, both losses, both datasets, 3 seeds → matches the published GPT-2-only shape (small positive xent PGR, near-zero/negative logconf). Save → `results/baseline_<dataset>.png`. **Compute and record the noise floor** from the seed spread here.
5. **First real datapoint:** `gt_fraction=0.25`, naive, both losses, both datasets, 3 seeds → `results/mix25_<dataset>.png`, plus the **25%-GT-only control** (reuses cached ceilings, nearly free).

### Deliverables

`NOTES_phase0.md` (code map + invariant verification + seed-environment table + GT-logconf reconciliation), the mixing seam, milestones 1–5 with plots, `RESULTS_phase0.md` (baseline / 25%-mix / 25%-GT-only median PGR with variance + the noise floor + a 3–4 sentence read).

### Decision gate

Does naive 25% mix beat **both** the pure-weak baseline and the GT-only control, *outside the noise floor*? If mix ≈ GT-only, weak labels add nothing and the project's question becomes "which strategies recover their value." If mix beats both, you have a positive headline. This read sets Phase 2's priority.

---

## Phase 1 — How much strong supervision? (the fraction curve + scale interaction)

**Goal:** the two marquee scientific results. Both reuse the cached spine; cost is transfer runs only.

### Runs

`gt_fraction ∈ {0.0, 0.05, 0.10, 0.25, 0.50, 1.0}`, full GPT-2 family, both losses, BoolQ at 3 seeds. (Second dataset deferred to headline replication.)

### Result 1 — supervision scaling curve

Median PGR vs. `gt_fraction`, overlay GT-only control at each fraction (answers "do weak labels add anything on top of this much GT?"). Look for the **knee** — does a small budget capture most of the benefit?

### Result 2 — scale interaction (the most OpenAI-researcher-sounding result)

For each GPT-2 size, estimate ΔPGR per unit strong-label fraction. **Does the marginal value of strong supervision grow with student size?** Plot ∂PGR/∂fraction vs. student size. If larger students extract more value from scarce strong supervision, that is a headline finding in its own right. This is computed from the *same* runs as Result 1 — do not run a separate sweep for it.

### Deliverables

`RESULTS_phase1.md` + the fraction-curve plot + the scale-interaction plot.

### Decision gate

The knee location tells you the budget regime where strategy differences will be visible (strategies only matter where the naive curve hasn't saturated) — fix Phase 2's budget there. The scale-interaction sign/magnitude tells you whether to foreground "scarce supervision scales with capability" as a headline. **Phase 0 + Phase 1 is the must-ship core.**

---

## Phase 2 — Where to spend it, and how to combine it (the idea-space portfolio)

**Goal:** this is where "try as many approaches as possible" is graded. Present the full idea space as a grid (axis of attack × instantiation), run as many cells as compute allows in priority order at the Phase-1-chosen budget, and **present the grid itself as a deliverable** — the map of ideas is the artifact, not every run.

All runs: fixed budget, full GPT-2 family, both losses, BoolQ, 3 seeds (headline cells → 5). Each cell = one branch in `apply_label_mixing` or the loss site → one identical-format plot + one median-PGR ± variance. Hold GT count and total labels constant across cells.

### Axis A — Allocation: *where* to spend the budget (the scalable-oversight question)
- A1. Random (baseline).
- A2. Highest weak-uncertainty (entropy) examples.
- A3. **Lowest** weak-uncertainty — the **sanity-inverter**: if targeting uncertain examples helps, targeting confident ones should hurt; the expected ordering is a rigor signal.
- A4. Weak/student-disagreement examples (train a preliminary student, request GT where they disagree).
- A5. Diversity/coverage-based (spread GT across embedding clusters).
- Headline target: "non-random allocation beats random at equal budget, and the ordering confirms the mechanism."

### Axis B — Combination: *how* to combine the two signals
- B1. Hard GT swap (= naive).
- B2. Soft GT (confident-but-not-one-hot targets).
- B3. Confidence-weighted blending: per-example interpolate weak and GT targets by weak confidence (trust weak where confident, lean on GT where not).
- B4. Weighted loss: L = L_weak + λ·L_strong, sweep λ ∈ {0.5, 1, 2, 4, 8}.
- B5. Two-stage curriculum: weak-pretrain → GT-finetune, **and** GT-first → weak (latter likely worse but diagnostic). Test both orders.

### Axis C — Loss / training dynamics (interacts with logconf)
- C1. GT-anchored logconf: GT rows are hard anchors exempt from confidence down-weighting; weak rows get standard confidence treatment.
- C2. Scheduled confidence loss: trust weak early, ramp self-confidence only after GT has calibrated the student.
- C3. GT subset as clean validation signal for early-stopping / checkpoint selection — reframes GT as *evaluation* rather than training signal. A realistic use of a tiny budget most candidates miss.

### Axis D — Higher-creativity, scalable-oversight-native (pick 1–2; "interesting even if it fails")
- D1. **Teacher-reliability modeling:** use the GT subset to learn P(teacher correct | x), then reweight weak labels by predicted reliability. Reframes strong labels as a *calibration resource* rather than extra labels — the most scalable-oversight-aligned idea here.
- D2. **Active oversight loop:** train student → find uncertain examples → reveal GT there → retrain, iterated. The closest thing to *realistic* budget allocation (you discover where to spend rather than knowing in advance).
- D3. Relabel-and-retrain: train a corrector on the GT subset, fix weak labels on the rest, train on the corrected set.
- D4. GT-guided filtering: drop the most-distrusted weak labels rather than overwrite — does *removing* bad weak supervision beat *adding* good GT at equal budget?

### Deliverables

`RESULTS_phase2.md` with: the idea-space grid (which cells ran, which didn't, and why — unrun cells are deliberate prioritization, state so explicitly), one comparison table (strategy × loss × median PGR ± variance), each run's plot, and an explicit **negative-results section** framed as findings.

### Decision gate

Pick the 1–2 winning strategies (outside the noise floor) to carry into mechanism + replication.

---

## Phase 3 — Why it works, and does it hold (mechanism + robustness + replication)

**Goal:** the depth signal. Drop to one analysis if time-bound.

- **Imitation analysis** (uses `label_source` + saved weak predictions): on rows where the weak label was *wrong*, does the winning strategy move the student toward truth or toward weak-agreement, vs. naive? Disproportionate correction of weak-error cases is the most satisfying mechanistic result.
- **Weak-quality interaction:** vary weak teacher strength (smaller vs. larger weak GPT-2) at fixed budget — does GT matter more when the teacher is worse (corrects gross errors) or better (final refinement)? Substitutes vs. complements.
- **Noisy-oversight robustness:** inject label noise into the GT subset to simulate imperfect amplified labels (futureproofs the "GT as stand-in for scalable oversight" framing). Does the winner degrade gracefully?
- **Second-dataset replication of the headline** (NOT a full sweep): run the 1–2 winning strategies + the fraction curve on the secondary dataset, 3 seeds. "It reproduces on a second task" is the credibility multiplier; cross-task consistency partly compensates for thin within-task seeds.

### Deliverable

`RESULTS_phase3.md` + mechanism figures + the replication plot.

---

## Phase 4 — Synthesis & presentation (≤20 min)

Organize the talk around three questions (the cleanest spine):

1. **How much strong supervision is needed?** → fraction-scaling curve + the scale-interaction result.
2. **Where should it be allocated?** → random vs. uncertainty vs. disagreement (Axis A).
3. **How should weak and strong supervision be combined?** → mixing vs. weighting vs. two-stage vs. reliability modeling (Axes B/C/D).

Suggested timing (it runs long; your end-state problem is cutting, not padding): motivation + thesis + setup ~3 min ("strong supervision is a budget; the questions are the marginal-value curve and how to spend it"); Q1 ~5 min; Q2 ~4 min; Q3 ~4 min; mechanism + negative-results slide + threats to validity + next steps ~4 min (threats: small-model NLP may not transfer to frontier scale; real amplified labels are noisy). Include the seed-environment table as a rigor slide. Finalize `TIME_LOG.md` and the private forked repo link.

---

## Why this structure is robust

Phase 0 front-loads all expensive reusable computation (GT ceilings, weak predictions with full logits), all correctness gating, **and** the noise floor + seed-environment decisions — so every downstream number is interpretable and every downstream condition is cheap relative to a cold start. Each phase ends in a decision gate that uses results-in-hand to prioritize the next, which is the only honest way to run this given you can't pre-pay all conditions. The compute doctrine makes the breadth-vs-power tradeoff explicit and resolves it toward power, because on this near-zero-PGR benchmark an unreplicated result is indistinguishable from noise. The plan answers the grading rubric clause-by-clause: **breadth** via the Phase 2 grid across four axes; **failure tolerance** via pre-registration + the negative-results section; **generalization** via the weak-disagreement subset, overfitting checks, and second-dataset replication; **rigor** via the noise floor, seed tiering, held-constant confounds, and the sanity-inverter. The object being graded is whether you think like a researcher — the structure is built to make that thinking visible in 20 minutes.
