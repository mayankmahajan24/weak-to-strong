#!/usr/bin/env python3
"""Mechanism experiment driver — KEEPS results.pkl (per-example test predictions).

Tests imitation vs correction: on test rows where the weak teacher is wrong, does the strong
student move toward truth (correction / generalization) or re-imitate the teacher's error, as the
GT fraction rises? Widest gap = gpt2 -> gpt2-xl.

Two phases on an 8-GPU box:
  A. gpt2 GT (teacher) on boolq+sciq, seeds 0,1 -> test preds (results.pkl) + weak_labels.
  B. gpt2-xl mixing at frac {0,0.1,0.25,0.5,1.0} on the gpt2 weak_labels, boolq+sciq, seeds 0,1
     -> student test preds (results.pkl) at each budget.
Writes OUT/results/data/mechanism/seed{N}/...  Cleans ONLY model weights (keeps results.pkl +
weak_labels). f=1.0 mixing == train on GT (the corrected reference); f=0 == pure weak transfer.

Usage: python run_mechanism_driver.py [OUT] [--seeds=0,1]
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
SEEDS = [int(x) for x in opts.get("seeds", "0,1").split(",")]
DATASETS = ["boolq", "sciq"]
FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
MBS = {"gpt2": 16, "gpt2-xl": 2}
LR = {"gpt2": "5e-05", "gpt2-xl": "1e-05"}
WEAK, STRONG = "gpt2", "gpt2-xl"


def wlp(ds, seed):
    d = (f"bs=32-dn={ds}-e=2-ee=1000000-lp=0-l=xent-l={LR[WEAK]}-ls=cosi_anne-mc=1024-"
         f"ms={WEAK}-nd=20000-ntd=10000-o=adam-s={seed}-twd=0")
    return f"{OUT}/results/data/mechanism/seed{seed}/{d}/weak_labels"


def clean_weights(seed):  # keep results.pkl + weak_labels; drop only big model weights
    for pat in ("pytorch_model*.bin", "*.safetensors"):
        for f in glob.glob(os.path.join(OUT, "results/data", f"mechanism/seed{seed}", "**", pat), recursive=True):
            try: os.remove(f)
            except OSError: pass


def base_cmd(model, ds, seed, sub, extra):
    return ["python", "train_simple.py", f"--model_size={model}", f"--ds_name={ds}",
            f"--seed={seed}", f"--minibatch_size_per_device={MBS[model]}",
            f"--results_folder={OUT}/results/data", f"--sweep_subfolder={sub}", *extra]


def run_pool(jobs, label):
    print(f"\n=== {label}: {len(jobs)} jobs ===", flush=True)
    q = queue.Queue()
    for j in jobs: q.put(j)
    done, failed, lock = [], [], threading.Lock()

    def worker(gpu):
        while True:
            try: j = q.get_nowait()
            except queue.Empty: return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WANDB_MODE="disabled",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            t0 = time.time()
            r = subprocess.run(j["cmd"], cwd=OUT, env=env, capture_output=True, text=True)
            clean_weights(j["seed"])
            with lock:
                if r.returncode == 0:
                    done.append(j["tag"]); print(f"[gpu{gpu}] OK {j['tag']} ({time.time()-t0:.0f}s)  {len(done)}/{len(jobs)}", flush=True)
                else:
                    failed.append(j["tag"]); print(f"[gpu{gpu}] FAIL {j['tag']}\n{r.stderr[-800:]}", flush=True)

    ts = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
    for t in ts: t.start()
    for t in ts: t.join()
    print(f"=== {label} done: {len(done)} ok, {len(failed)} failed ===", flush=True)
    return failed


# Phase A: teacher GT (gpt2) -> test preds + weak_labels
gt_jobs = [dict(seed=s, tag=f"TEACHER gpt2:{ds}:s{s}", cmd=base_cmd("gpt2", ds, s, f"mechanism/seed{s}", []))
           for ds in DATASETS for s in SEEDS]
# Phase B: student gpt2-xl mixing at each fraction
def fdir(f): return f"f{int(round(f*100)):03d}"
st_jobs = [dict(seed=s, tag=f"STUDENT xl<-gpt2:{ds}:f{f}:s{s}",
                cmd=base_cmd(STRONG, ds, s, f"mechanism/seed{s}/{fdir(f)}",
                             [f"--gt_fraction={f}", f"--gt_seed={s}", f"--weak_labels_path={wlp(ds, s)}"]))
           for ds in DATASETS for f in FRACS for s in SEEDS]

print(f"OUT={OUT} seeds={SEEDS}  teacher={len(gt_jobs)}  student={len(st_jobs)}", flush=True)
fa = run_pool(gt_jobs, "PHASE A (teacher gpt2 GT)")
fb = run_pool(st_jobs, "PHASE B (student gpt2-xl mixing)")
print(f"\nALL DONE. teacher_failed={len(fa)} student_failed={len(fb)}", flush=True)
open(f"{OUT}/mechanism_status.txt", "w").write(f"teacher_failed={fa}\nstudent_failed={fb}\n")
