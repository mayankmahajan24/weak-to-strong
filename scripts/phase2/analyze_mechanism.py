#!/usr/bin/env python3
"""Mechanism analysis — imitation vs correction (runs on the box where results.pkl live).

For gpt2 (teacher) -> gpt2-xl (student) at GT fractions {0,0.1,0.25,0.5,1.0}, joins teacher & student
per-example TEST predictions (by 'txt') and asks: on rows where the teacher is WRONG, does the
student recover the truth as GT budget rises (correction/generalization), or does it re-imitate the
teacher's error (imitation)? Writes results/mechanism_summary.csv + results/plots/mechanism.png.

Diffuse-regularizer signature: recovery on teacher-wrong rows is ~LINEAR in fraction (you only fix
what you directly supervise). Generalization signature: recovery is CONCAVE (a little GT teaches the
student to override the teacher broadly). Run: python scripts/phase2/analyze_mechanism.py
"""
import glob, pickle, statistics as st, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/data/mechanism"
DSETS = ["boolq", "sciq"]
SEEDS = [0, 1]
FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
def fdir(f): return f"f{int(round(f*100)):03d}"

def load_preds(pkl):
    d = pickle.load(open(pkl, "rb"))["test_results"]
    return {t: (int(g), int(p)) for t, g, p in zip(d["txt"], d["gt_label"], d["hard_label"])}

def teacher_pkl(seed, ds):
    for p in glob.glob(str(DATA / f"seed{seed}/bs=*ms=gpt2-nd*/results.pkl")):
        if f"dn={ds}" in p: return p
def student_pkl(seed, ds, f):
    for p in glob.glob(str(DATA / f"seed{seed}/{fdir(f)}/bs=*ms=gpt2-xl*/results.pkl")):
        if f"dn={ds}" in p: return p

rows = []
for ds in DSETS:
    for f in FRACS:
        accs_tw, accs_tc, accs_all, imit_tw = [], [], [], []
        for s in SEEDS:
            tp, sp = teacher_pkl(s, ds), student_pkl(s, ds, f)
            if not tp or not sp: continue
            T, S = load_preds(tp), load_preds(sp)
            common = [t for t in T if t in S]
            tw = [t for t in common if T[t][1] != T[t][0]]      # teacher wrong
            tc = [t for t in common if T[t][1] == T[t][0]]      # teacher correct
            if tw:
                accs_tw.append(st.mean(S[t][1] == T[t][0] for t in tw))   # student correct on teacher-wrong
                imit_tw.append(st.mean(S[t][1] == T[t][1] for t in tw))   # student == teacher (imitates the error)
            if tc:
                accs_tc.append(st.mean(S[t][1] == T[t][0] for t in tc))
            accs_all.append(st.mean(S[t][1] == T[t][0] for t in common))
        if accs_all:
            rows.append(dict(ds=ds, frac=f,
                             acc_all=round(st.mean(accs_all), 4),
                             acc_teacher_wrong=round(st.mean(accs_tw), 4) if accs_tw else None,
                             acc_teacher_correct=round(st.mean(accs_tc), 4) if accs_tc else None,
                             imitate_rate_on_wrong=round(st.mean(imit_tw), 4) if imit_tw else None,
                             n_seeds=len(accs_all)))

# write csv + print table
out_csv = ROOT / "results/mechanism_summary.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {out_csv}\n")
for ds in DSETS:
    print(f"=== {ds} : gpt2 -> gpt2-xl ===")
    print(f"  {'frac':>5} {'acc_all':>8} {'acc|Twrong':>11} {'acc|Tright':>11} {'imitate|Twrong':>14}")
    for r in [r for r in rows if r["ds"] == ds]:
        print(f"  {r['frac']:>5} {r['acc_all']:>8} {str(r['acc_teacher_wrong']):>11} "
              f"{str(r['acc_teacher_correct']):>11} {str(r['imitate_rate_on_wrong']):>14}")
    # recovery linearity diagnostic on teacher-wrong rows
    tw = {r["frac"]: r["acc_teacher_wrong"] for r in rows if r["ds"] == ds and r["acc_teacher_wrong"] is not None}
    if 0.0 in tw and 1.0 in tw and 0.5 in tw and tw[1.0] > tw[0.0]:
        frac_recovered_at_half = (tw[0.5] - tw[0.0]) / (tw[1.0] - tw[0.0])
        print(f"  -> teacher-wrong recovery at f=0.5: {frac_recovered_at_half:.0%} of the f=0->1 gap "
              f"({'concave/generalizing' if frac_recovered_at_half > 0.6 else 'linear-or-slower/diffuse'})\n")

# plot
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DSETS), figsize=(11, 4.5), sharey=True)
    for ax, ds in zip(axes, DSETS):
        rr = [r for r in rows if r["ds"] == ds]
        fr = [r["frac"] for r in rr]
        ax.plot(fr, [r["acc_teacher_wrong"] for r in rr], "o-", color="tab:red", label="student acc | teacher WRONG")
        ax.plot(fr, [r["acc_teacher_correct"] for r in rr], "s-", color="tab:green", label="student acc | teacher right")
        ax.plot(fr, [r["imitate_rate_on_wrong"] for r in rr], "^--", color="gray", label="imitates teacher's error")
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_title(f"{ds}: gpt2 → gpt2-xl"); ax.set_xlabel("GT fraction"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("rate"); axes[0].legend(fontsize=8, loc="best")
    out = ROOT / "results/plots/mechanism.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Imitation vs correction: does GT recover the teacher's errors?")
    fig.tight_layout(); fig.savefig(out, dpi=140); print(f"wrote {out}")
except Exception as e:
    print("plot skipped:", e)
