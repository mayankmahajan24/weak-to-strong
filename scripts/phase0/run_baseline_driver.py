#!/usr/bin/env python3
"""Baseline generation driver — reproduce the phase-0 baseline for new seeds (default 3,4).

Two phases on an 8-GPU box:
  A. GT runs: 4 models × {boolq,sciq} × seeds → train on ground truth, emit weak_labels + ceilings.
  B. frac=0 transfers: 10 weak<=strong pairs × {boolq,sciq} × {xent,logconf} × seeds, loading the
     GT weak_labels from phase A.
Writes to OUT/results/data/baseline/seed{N}/... (same layout as seeds 0/1/2). Keeps weak_labels;
cleans model weights + results.pkl per run. Phase B waits for ALL of phase A (transfers need the
weak_labels).

Usage: python run_baseline_driver.py [OUT] [--seeds=3,4]
"""
import glob, os, queue, subprocess, sys, threading, time

OUT, opts = None, {}
for a in sys.argv[1:]:
    if a.startswith("--"):
        k, _, v = a[2:].partition("="); opts[k] = v
    else:
        OUT = a
if OUT is None:
    OUT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NGPU = int(opts.get("ngpu", 8))
SEEDS = [int(x) for x in opts.get("seeds", "3,4").split(",")]
DATASETS = ["boolq", "sciq"]
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
MBS = {"gpt2": 16, "gpt2-medium": 8, "gpt2-large": 4, "gpt2-xl": 2}
LR = {"gpt2": "5e-05", "gpt2-medium": "5e-05", "gpt2-large": "1e-05", "gpt2-xl": "1e-05"}


def wlp(weak, ds, seed):
    d = (f"bs=32-dn={ds}-e=2-ee=1000000-lp=0-l=xent-l={LR[weak]}-ls=cosi_anne-mc=1024-"
         f"ms={weak}-nd=20000-ntd=10000-o=adam-s={seed}-twd=0")
    return f"{OUT}/results/data/baseline/seed{seed}/{d}/weak_labels"


def clean(sub):
    for pat in ("pytorch_model*.bin", "*.safetensors", "results.pkl"):
        for f in glob.glob(os.path.join(OUT, "results/data", sub, "**", pat), recursive=True):
            try:
                os.remove(f)
            except OSError:
                pass


def run_pool(jobs, label):
    print(f"\n=== {label}: {len(jobs)} jobs ===", flush=True)
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    done, failed, lock = [], [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE="disabled",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            t0 = time.time()
            r = subprocess.run(j["cmd"], cwd=OUT, env=env, capture_output=True, text=True)
            clean(f"baseline/seed{j['seed']}")
            with lock:
                if r.returncode == 0:
                    done.append(j["tag"])
                    print(f"[gpu{gpu}] OK {j['tag']} ({time.time()-t0:.0f}s)  {len(done)}/{len(jobs)}", flush=True)
                else:
                    failed.append(j["tag"])
                    print(f"[gpu{gpu}] FAIL {j['tag']}\n{r.stderr[-800:]}", flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"=== {label} done: {len(done)} ok, {len(failed)} failed ===", flush=True)
    return failed


def base_cmd(model, ds, seed, extra):
    return ["python", "train_simple.py", f"--model_size={model}", f"--ds_name={ds}",
            f"--seed={seed}", f"--minibatch_size_per_device={MBS[model]}",
            f"--results_folder={OUT}/results/data", f"--sweep_subfolder=baseline/seed{seed}", *extra]


# ---- Phase A: GT runs (produce weak_labels + ceilings) ----
gt_jobs = []
for ds in DATASETS:
    for m in ORDER:
        for seed in SEEDS:
            gt_jobs.append(dict(seed=seed, tag=f"GT:{m}:{ds}:s{seed}",
                                cmd=base_cmd(m, ds, seed, [])))  # no weak_model -> GT; loss defaults xent

# ---- Phase B: frac=0 transfers (10 weak<=strong pairs × 2 losses) ----
pairs = [(s, w) for s in ORDER for w in ORDER if RANK[w] <= RANK[s]]  # 10 incl self
tr_jobs = []
for ds in DATASETS:
    for strong, weak in pairs:
        for loss in ["xent", "logconf"]:
            for seed in SEEDS:
                tr_jobs.append(dict(seed=seed, tag=f"TR:{strong}<-{weak}:{ds}:{loss}:s{seed}",
                                    cmd=base_cmd(strong, ds, seed,
                                                 [f"--loss={loss}", f"--weak_labels_path={wlp(weak, ds, seed)}"])))

print(f"OUT={OUT} seeds={SEEDS}  GT jobs={len(gt_jobs)}  transfer jobs={len(tr_jobs)}", flush=True)
fa = run_pool(gt_jobs, "PHASE A (GT)")
if fa:
    print(f"WARNING: {len(fa)} GT runs failed — transfers depending on them will fail too: {fa}", flush=True)
fb = run_pool(tr_jobs, "PHASE B (transfers)")
print(f"\nALL DONE. GT failed={len(fa)} transfers failed={len(fb)}", flush=True)
with open(f"{OUT}/baseline_status.txt", "w") as f:
    f.write(f"gt_failed={len(fa)} tr_failed={len(fb)}\n{fa}\n{fb}\n")
