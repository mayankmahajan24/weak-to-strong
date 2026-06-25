#!/usr/bin/env python3
"""Elicitation box-driver (run ON the GPU box).

Phase 1 (GPU pool): extract frozen activations for every (model, ds, seed) × {train, test} with the
CCS contrast prompts — forward passes only.
Phase 2 (CPU): run_elicitation.py per (model, ds, seed) -> results/elicitation/runs/<...>.json.

Only the small JSONs need pulling back (the .npz stay on the box). Usage:
  python run_box.py /workspace/w2s [--only_model=gpt2-xl] [--ngpu=8]
"""
import glob
import os
import queue
import subprocess
import sys
import threading
import time

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
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
DSETS = opts.get("ds", "boolq,sciq").split(",")
SEEDS = [int(s) for s in opts.get("seeds", "0,1,2").split(",")]
MBS = {"gpt2": 32, "gpt2-medium": 16, "gpt2-large": 8, "gpt2-xl": 4}  # forward-only batch sizes
N_TRAIN = int(opts.get("n_docs", 2000))     # train pool (k-shot + layer selection + upper bound)
N_TEST = int(opts.get("n_test_docs", 5000))  # test eval split
LAYERS = opts.get("layers", "last,half")
ACTS = os.path.join(OUT, "results/elicitation/acts")
RUNS = os.path.join(OUT, "results/elicitation/runs")
ELDIR = os.path.dirname(os.path.abspath(__file__))

models = [opts["only_model"]] if opts.get("only_model") else ORDER

ext_jobs = []
for m in models:
    for ds in DSETS:
        for seed in SEEDS:
            for split in ("train", "test"):
                ext_jobs.append(dict(m=m, ds=ds, seed=seed, split=split))

print(f"OUT={OUT}  models={models} ds={DSETS} seeds={SEEDS}  extraction jobs={len(ext_jobs)}", flush=True)

q = queue.Queue()
for j in ext_jobs:
    q.put(j)
done, failed, lock = [], [], threading.Lock()


def worker(gpu):
    while True:
        try:
            j = q.get_nowait()
        except queue.Empty:
            return
        cmd = ["python", os.path.join(ELDIR, "extract_activations.py"),
               f"--model_size={j['m']}", f"--ds={j['ds']}", f"--seed={j['seed']}",
               f"--split={j['split']}", f"--contrast={j['ds']}", f"--layers={LAYERS}",
               f"--n_docs={N_TRAIN}", f"--n_test_docs={N_TEST}",
               f"--batch_size={MBS[j['m']]}", f"--out={ACTS}", "--device=cuda"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE="disabled",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        tag = f"{j['ds']}:{j['m']}:s{j['seed']}:{j['split']}"
        t0 = time.time()
        r = subprocess.run(cmd, cwd=OUT, env=env, capture_output=True, text=True)
        with lock:
            if r.returncode == 0:
                done.append(tag)
                print(f"[gpu{gpu}] OK {tag} ({time.time()-t0:.0f}s)  {len(done)}/{len(ext_jobs)}", flush=True)
            else:
                failed.append(tag)
                print(f"[gpu{gpu}] FAIL {tag}\n{r.stderr[-1200:]}", flush=True)


ts = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(f"\nEXTRACTION DONE: {len(done)} ok, {len(failed)} failed", flush=True)
if failed:
    print("FAILED:", failed, flush=True)

# Phase 2 — CPU analysis per config
print("\nrunning run_elicitation (CPU)...", flush=True)
n_ok = 0
for m in models:
    for ds in DSETS:
        for seed in SEEDS:
            cmd = ["python", os.path.join(ELDIR, "run_elicitation.py"),
                   f"--ds={ds}", f"--model_size={m}", f"--seed={seed}",
                   f"--acts={ACTS}", f"--out={RUNS}"]
            r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
            if r.returncode == 0:
                n_ok += 1
                print(r.stdout.strip(), flush=True)
            else:
                print(f"run_elicitation FAIL {ds}:{m}:s{seed}\n{r.stderr[-600:]}", flush=True)
print(f"\nALL DONE: extraction {len(done)}/{len(ext_jobs)}, analysis {n_ok} configs", flush=True)
with open(os.path.join(OUT, "elicitation_status.txt"), "w") as f:
    f.write(f"extraction done={len(done)} failed={len(failed)}\nanalysis_ok={n_ok}\n"
            f"FAILED:\n" + "\n".join(failed) + "\n")
