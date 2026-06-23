#!/usr/bin/env python3
"""Phase 2 combination-method portfolio — GPU-pool driver (run ON the 8-GPU box).

5 methods × 6 strict weak<strong pairs × {0.10,0.25,0.50} × 3 seeds = 270 runs, BoolQ.
Naive baseline is reused from Phase 1 (NOT re-run here). Methods (see plans/phase2.md):
  weighted       --combination_method=weighted --gt_loss_weight=4.0    (xent)    M1
  soft_gt        --combination_method=soft_gt  --soft_gt_eps=0.1        (xent)    M2
  gt_anchored    --combination_method=gt_anchored                       (logconf) M3
  reliability    --combination_method=reliability                       (xent)    M4
  gt_early_stop  --combination_method=gt_early_stop                     (xent)    M5

Hyperparameters (gt_loss_weight, soft_gt_eps) are fixed at sensible, *untuned* defaults
(consistent with the repo's "LRs not particularly tuned" ethos and the "interesting even if it
fails" framing). Override at the top if a pilot motivates it.

PRE-SWEEP GATES (run before the full sweep — see plans/PHASE2_PROMPT.md):
  1. naive reproduction (gpt2-medium<-gpt2 @0.25 seed1 ~= 0.673 bit-for-bit)
  2. M5 smoke: one gt_early_stop run trains on 0 GT rows, selects a checkpoint, sane acc.
Use --only=<method[,method]> to run a subset (e.g. the M5 smoke), --pairs / --fracs / --seeds
to subset further. Cleans model weights + results.pkl per run.

Usage: python run_portfolio_driver.py [OUT] [--only=m1,m2] [--fracs=0.1] [--seeds=1] [--pairs=1]
"""
import glob
import os
import queue
import subprocess
import sys
import threading
import time

# ---- args ----
OUT = None
opts = {}
for a in sys.argv[1:]:
    if a.startswith("--"):
        k, _, v = a[2:].partition("=")
        opts[k] = v
    else:
        OUT = a
if OUT is None:
    OUT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NGPU = int(opts.get("ngpu", 8))
DS = opts.get("ds", "boolq")  # boolq (default) or sciq (winner replication / confirm)
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
MBS = {"gpt2": 16, "gpt2-medium": 8, "gpt2-large": 4, "gpt2-xl": 2}
LR = {"gpt2": "5e-05", "gpt2-medium": "5e-05", "gpt2-large": "1e-05", "gpt2-xl": "1e-05"}
FDIR = {0.1: "010", 0.25: "025", 0.5: "050"}

METHODS = [
    dict(name="weighted", loss="xent", flags=["--combination_method=weighted", "--gt_loss_weight=4.0"]),
    dict(name="soft_gt", loss="xent", flags=["--combination_method=soft_gt", "--soft_gt_eps=0.1"]),
    dict(name="gt_anchored", loss="logconf", flags=["--combination_method=gt_anchored"]),
    dict(name="reliability", loss="xent", flags=["--combination_method=reliability"]),
    dict(name="gt_early_stop", loss="xent", flags=["--combination_method=gt_early_stop"]),
]
STRICT_PAIRS = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]  # 6
FRACS = [0.1, 0.25, 0.5]
SEEDS = [0, 1, 2]

# ---- optional subsetting (for smoke tests / pilots) ----
if opts.get("only"):
    keep = set(opts["only"].split(","))
    METHODS = [m for m in METHODS if m["name"] in keep]
if opts.get("fracs"):
    FRACS = [float(x) for x in opts["fracs"].split(",")]
if opts.get("seeds"):
    SEEDS = [int(x) for x in opts["seeds"].split(",")]
if opts.get("pairs"):  # cap number of strict pairs (smallest-strong-model first)
    STRICT_PAIRS = STRICT_PAIRS[: int(opts["pairs"])]


def weak_labels_path(weak, seed):
    d = (f"bs=32-dn={DS}-e=2-ee=1000000-lp=0-l=xent-l={LR[weak]}-ls=cosi_anne-mc=1024-"
         f"ms={weak}-nd=20000-ntd=10000-o=adam-s={seed}-twd=0")
    return f"{OUT}/results/data/baseline/seed{seed}/{d}/weak_labels"


jobs = []
for m in METHODS:
    for strong, weak in STRICT_PAIRS:
        for frac in FRACS:
            for seed in SEEDS:
                # boolq → phase2_<method> (unchanged); sciq → phase2_sciq_<method>
                prefix = "phase2" if DS == "boolq" else f"phase2_{DS}"
                jobs.append(dict(m=m, strong=strong, weak=weak, frac=frac, seed=seed,
                                 sub=f"{prefix}_{m['name']}/{FDIR[frac]}/seed{seed}"))

print(f"OUT={OUT}  methods={[m['name'] for m in METHODS]}  pairs={len(STRICT_PAIRS)} "
      f"fracs={FRACS} seeds={SEEDS}  total jobs={len(jobs)}", flush=True)

q = queue.Queue()
for j in jobs:
    q.put(j)
done, failed, lock = [], [], threading.Lock()


def clean(p):
    for pat in ("pytorch_model*.bin", "*.safetensors", "results.pkl"):
        for f in glob.glob(os.path.join(p, "**", pat), recursive=True):
            try:
                os.remove(f)
            except OSError:
                pass


def worker(gpu):
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        m = j["m"]
        cmd = ["python", "train_simple.py",
               f"--model_size={j['strong']}", f"--ds_name={DS}", f"--loss={m['loss']}",
               f"--seed={j['seed']}", f"--gt_seed={j['seed']}", f"--gt_fraction={j['frac']}",
               *m["flags"],
               f"--minibatch_size_per_device={MBS[j['strong']]}",
               f"--results_folder={OUT}/results/data", f"--sweep_subfolder={j['sub']}",
               f"--weak_labels_path={weak_labels_path(j['weak'], j['seed'])}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE="disabled",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        tag = f"{m['name']}:{j['strong']}<-{j['weak']}:f{j['frac']}:s{j['seed']}"
        t0 = time.time()
        r = subprocess.run(cmd, cwd=OUT, env=env, capture_output=True, text=True)
        clean(f"{OUT}/results/data/{j['sub']}")
        with lock:
            if r.returncode == 0:
                done.append(tag)
                print(f"[gpu{gpu}] OK {tag} ({time.time()-t0:.0f}s)  {len(done)}/{len(jobs)}", flush=True)
            else:
                failed.append(tag)
                print(f"[gpu{gpu}] FAIL {tag}\n{r.stderr[-1000:]}", flush=True)


ts = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(f"\nALL DONE: {len(done)} ok, {len(failed)} failed", flush=True)
if failed:
    print("FAILED:", failed, flush=True)
with open(f"{OUT}/phase2_status.txt", "w") as f:
    f.write(f"done={len(done)} failed={len(failed)}\nFAILED:\n" + "\n".join(failed) + "\n")
