#!/usr/bin/env python3
"""Phase 2 M5 — unit test for the checkpoint-selection helper `_select_best_step`.

`train.py` has a heavy import chain (torch/transformers/torch_optimizer), so we extract just
the pure function from source via AST and exec it in an empty namespace — this tests the ACTUAL
code with no GPU/ML deps. (The loop integration that *produces* the per-step dict is verified by
the on-box M5 smoke run, since it only executes inside a real training run.)
"""
import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "weak_to_strong" / "train.py").read_text()
_fn = next(
    ast.get_source_segment(SRC, n)
    for n in ast.parse(SRC).body
    if isinstance(n, ast.FunctionDef) and n.name == "_select_best_step"
)
_ns = {}
exec(_fn, _ns)
select = _ns["_select_best_step"]

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


check("picks the max-accuracy step", select({10: 0.60, 20: 0.80, 30: 0.70}) == 20)
check("tie ⇒ earliest step (least training)", select({10: 0.80, 20: 0.80, 30: 0.50}) == 10)
check("monotone increasing ⇒ last step", select({5: 0.5, 10: 0.6, 15: 0.7}) == 15)
check("monotone decreasing ⇒ first step", select({5: 0.7, 10: 0.6, 15: 0.5}) == 5)
check("single eval ⇒ that step", select({42: 0.66}) == 42)
check("unsorted insertion order handled", select({30: 0.7, 10: 0.9, 20: 0.8}) == 10)

print(f"\nALL {PASS} CHECKS PASSED")
