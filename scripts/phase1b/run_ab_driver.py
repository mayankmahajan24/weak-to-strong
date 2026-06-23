#!/usr/bin/env python3
"""Phase 1b Components A+B GPU-pool driver (run ON the 8-GPU instance).

A (random_labels): 4 strong x {0.10,0.25} x 3 seeds = 24, weak source = gpt2.
B (oracle):        6 strict weak<strong pairs x {0.10,0.25} x 3 seeds = 36.
Total 60 runs, xent, BoolQ. Writes to OUT/results/data/<sweep_subfolder>/...;
cleans model weights + results.pkl per run (incl. sharded pytorch_model-*.bin).

Usage:  python run_ab_driver.py [OUT]   (OUT defaults to this repo root on the box)
"""
import glob
import os
import queue
import subprocess
import sys
import threading
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NGPU = 8
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
MBS = {"gpt2": 16, "gpt2-medium": 8, "gpt2-large": 4, "gpt2-xl": 2}
LR = {"gpt2": "5e-05", "gpt2-medium": "5e-05", "gpt2-large": "1e-05", "gpt2-xl": "1e-05"}
FDIR = {0.1: "010", 0.25: "025"}
FRACS = [0.1, 0.25]
SEEDS = [0, 1, 2]


def weak_labels_path(weak, seed):
    d = (f"bs=32-dn=boolq-e=2-ee=1000000-lp=0-l=xent-l={LR[weak]}-ls=cosi_anne-mc=1024-"
         f"ms={weak}-nd=20000-ntd=10000-o=adam-s={seed}-twd=0")
    return f"{OUT}/results/data/baseline/seed{seed}/{d}/weak_labels"


# ---- build job list ----
jobs = []
# A: random_labels, weak source = gpt2, vary strong model
for strong in ORDER:
    for frac in FRACS:
        for seed in SEEDS:
            jobs.append(dict(strat="random_labels", strong=strong, weak="gpt2", frac=frac,
                             seed=seed, sub=f"phase1b_random/{FDIR[frac]}/seed{seed}"))
# B: oracle, strict weak<strong pairs
strict_pairs = [(s, w) for s in ORDER for w in ORDER if RANK[w] < RANK[s]]  # 6 pairs
for strong, weak in strict_pairs:
    for frac in FRACS:
        for seed in SEEDS:
            jobs.append(dict(strat="oracle", strong=strong, weak=weak, frac=frac,
                             seed=seed, sub=f"phase1b_oracle/{FDIR[frac]}/seed{seed}"))

print(f"OUT={OUT}  total jobs={len(jobs)} (A random_labels=24, B oracle=36)", flush=True)

q = queue.Queue()
for j in jobs:
    q.put(j)
done, failed, lock = [], [], threading.Lock()


def clean(save_glob):
    for pat in ("pytorch_model*.bin", "*.safetensors", "results.pkl", "model.safetensors*"):
        for f in glob.glob(os.path.join(save_glob, "**", pat), recursive=True):
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
        cmd = [
            "python", "train_simple.py",
            f"--model_size={j['strong']}", "--ds_name=boolq", "--loss=xent",
            f"--seed={j['seed']}", f"--gt_seed={j['seed']}", f"--gt_fraction={j['frac']}",
            f"--mixing_strategy={j['strat']}",
            f"--minibatch_size_per_device={MBS[j['strong']]}",
            f"--results_folder={OUT}/results/data",
            f"--sweep_subfolder={j['sub']}",
            f"--weak_labels_path={weak_labels_path(j['weak'], j['seed'])}",
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE="disabled",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        tag = f"{j['strat']}:{j['strong']}<-{j['weak']}:f{j['frac']}:s{j['seed']}"
        t0 = time.time()
        r = subprocess.run(cmd, cwd=OUT, env=env, capture_output=True, text=True)
        clean(f"{OUT}/results/data/{j['sub']}")
        with lock:
            if r.returncode == 0:
                done.append(tag)
                print(f"[gpu{gpu}] OK {tag} ({time.time()-t0:.0f}s)  {len(done)}/{len(jobs)}", flush=True)
            else:
                failed.append(tag)
                print(f"[gpu{gpu}] FAIL {tag}\n{r.stderr[-800:]}", flush=True)


threads = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"\nALL DONE: {len(done)} ok, {len(failed)} failed", flush=True)
if failed:
    print("FAILED:", failed, flush=True)
with open(f"{OUT}/phase1b_ab_status.txt", "w") as f:
    f.write(f"done={len(done)} failed={len(failed)}\n")
    f.write("FAILED:\n" + "\n".join(failed) + "\n")
