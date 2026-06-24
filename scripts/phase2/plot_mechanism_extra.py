#!/usr/bin/env python3
"""Mechanism extra plot — recovery-normalized view that makes 'diffuse/linear' explicit.

From mechanism_summary.csv: fraction of the f=0→1 teacher-wrong-row gap recovered, vs GT budget,
against the y=x reference. On/below the diagonal = linear-or-slower recovery = diffuse/volume-bound
(GT fixes only what it directly supervises). Above = concave = generalizing (a little GT → broad
override). Output: results/plots/mechanism_recovery.png
"""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(ROOT / "results/mechanism_summary.csv")))
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="linear (diffuse / volume-bound)")
for ds, col in [("boolq", "tab:blue"), ("sciq", "tab:green")]:
    rr = sorted([r for r in rows if r["ds"] == ds], key=lambda r: float(r["frac"]))
    tw = {float(r["frac"]): float(r["acc_teacher_wrong"]) for r in rr}
    lo, hi = tw[0.0], tw[1.0]
    fr = [float(r["frac"]) for r in rr]
    rec = [(tw[f] - lo) / (hi - lo) for f in fr]
    ax.plot(fr, rec, "o-", color=col, label=f"{ds} (recovered @0.5 = {(tw[0.5]-lo)/(hi-lo):.0%})")
ax.set_xlabel("GT fraction spent"); ax.set_ylabel("fraction of teacher-wrong errors recovered\n(of the f=0→1 gap)")
ax.set_title("Mechanism — error recovery is ~LINEAR in budget (diffuse), not concave\n→ targeting/combination can't help: only volume does")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="upper left")
ax.annotate("concave region\n(generalizing)\n— we are NOT here", (0.55, 0.85), fontsize=8, color="gray", ha="center")
fig.tight_layout(); fig.savefig(ROOT / "results/plots/mechanism_recovery.png", dpi=140)
print("wrote mechanism_recovery.png")
