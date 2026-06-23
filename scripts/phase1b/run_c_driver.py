#!/usr/bin/env python3
"""Phase 1b Component C GPU-pool driver (run ON the 8-GPU box). SciQ, xent.

C-mixing (naive): 10 weak<=strong pairs x {0.10,0.25,0.50,1.0} x 3 seeds = 120
C-random (control): 4 strong x {0.10,0.25} x 3 seeds = 24, weak source = gpt2.
Total 144 runs. Cleans model weights + pkl per run. Writes to OUT/results/data.

Usage: python run_c_driver.py [OUT]
"""
import glob, os, queue, subprocess, sys, threading, time

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NGPU = 8
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
RANK = {m: i for i, m in enumerate(ORDER)}
MBS = {"gpt2": 16, "gpt2-medium": 8, "gpt2-large": 4, "gpt2-xl": 2}
LR = {"gpt2": "5e-05", "gpt2-medium": "5e-05", "gpt2-large": "1e-05", "gpt2-xl": "1e-05"}
FDIR = {0.1: "010", 0.25: "025", 0.5: "050", 1.0: "100"}
SEEDS = [0, 1, 2]


def wlp(weak, seed):
    d = (f"bs=32-dn=sciq-e=2-ee=1000000-lp=0-l=xent-l={LR[weak]}-ls=cosi_anne-mc=1024-"
         f"ms={weak}-nd=20000-ntd=10000-o=adam-s={seed}-twd=0")
    return f"{OUT}/results/data/baseline/seed{seed}/{d}/weak_labels"


jobs = []
# C-mixing (naive): full weak<=strong triangle (10 pairs)
pairs = [(s, w) for s in ORDER for w in ORDER if RANK[w] <= RANK[s]]
for strong, weak in pairs:
    for frac in [0.1, 0.25, 0.5, 1.0]:
        for seed in SEEDS:
            jobs.append(dict(strat="naive", strong=strong, weak=weak, frac=frac, seed=seed,
                             sub=f"sciq_mixing/{FDIR[frac]}/seed{seed}"))
# C-random control: vary strong only, weak source = gpt2
for strong in ORDER:
    for frac in [0.1, 0.25]:
        for seed in SEEDS:
            jobs.append(dict(strat="random_labels", strong=strong, weak="gpt2", frac=frac, seed=seed,
                             sub=f"sciq_random/{FDIR[frac]}/seed{seed}"))

print(f"OUT={OUT} total jobs={len(jobs)} (mixing=120, random=24)", flush=True)
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
        cmd = ["python", "train_simple.py", f"--model_size={j['strong']}", "--ds_name=sciq",
               "--loss=xent", f"--seed={j['seed']}", f"--gt_seed={j['seed']}",
               f"--gt_fraction={j['frac']}", f"--mixing_strategy={j['strat']}",
               f"--minibatch_size_per_device={MBS[j['strong']]}",
               f"--results_folder={OUT}/results/data", f"--sweep_subfolder={j['sub']}",
               f"--weak_labels_path={wlp(j['weak'], j['seed'])}"]
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


ts = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(f"\nALL DONE: {len(done)} ok, {len(failed)} failed", flush=True)
if failed:
    print("FAILED:", failed, flush=True)
with open(f"{OUT}/phase1c_status.txt", "w") as f:
    f.write(f"done={len(done)} failed={len(failed)}\nFAILED:\n" + "\n".join(failed) + "\n")
