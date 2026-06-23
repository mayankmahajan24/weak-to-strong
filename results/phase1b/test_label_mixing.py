#!/usr/bin/env python3
"""Phase 1b — unit tests for the new label-mixing allocation strategies.

Tests `select_gt_indices` (the pure allocation core) on:
  (1) synthetic gt/hard arrays with a known error pattern (exhaustive logic), and
  (2) a real preserved weak_labels arrow (oracle must target the actual weak errors).

Real-data part needs pyarrow; run with the scratchpad venv:
  <venv>/bin/python results/phase1b/test_label_mixing.py [path/to/weak_labels.arrow]
The synthetic part runs on plain Python (no deps).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from weak_to_strong.label_mixing import select_gt_indices  # noqa: E402

PASS = 0


def check(name, cond):
    global PASS
    assert cond, f"FAIL: {name}"
    PASS += 1
    print(f"  ok: {name}")


def synthetic_tests():
    print("[synthetic]")
    # 10 rows; weak wrong on indices {1,3,5,7} (4 errors)
    n = 10
    gt = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    hard = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]  # wrong where gt!=hard -> {1,3,7}
    wrong = {i for i in range(n) if gt[i] != hard[i]}
    assert wrong == {1, 3, 7}, wrong

    # naive: size, determinism, subset
    s1 = select_gt_indices(n, gt, hard, 0.3, gt_seed=0, strategy="naive")
    s2 = select_gt_indices(n, gt, hard, 0.3, gt_seed=0, strategy="naive")
    check("naive size == round(0.3*10)=3", len(s1) == 3)
    check("naive deterministic in seed", s1 == s2)
    check("naive differs across seed", s1 != select_gt_indices(n, gt, hard, 0.3, 1, "naive"))
    check("naive subset of range(n)", s1 <= set(range(n)))

    # random_labels selects the SAME rows as naive (only non-GT labels differ downstream)
    sr = select_gt_indices(n, gt, hard, 0.3, gt_seed=0, strategy="random_labels")
    check("random_labels selection == naive selection", sr == s1)

    # oracle, budget <= #wrong: every selected row is a weak error
    o_small = select_gt_indices(n, gt, hard, 0.2, gt_seed=0, strategy="oracle")  # k=2, wrong=3
    check("oracle k<=#wrong: size 2", len(o_small) == 2)
    check("oracle k<=#wrong: all selected are weak-wrong", o_small <= wrong)

    # oracle, budget == #wrong: exactly the error set
    o_eq = select_gt_indices(n, gt, hard, 0.3, gt_seed=0, strategy="oracle")  # k=3 == 3 wrong
    check("oracle k==#wrong: exactly the error set", o_eq == wrong)

    # oracle, budget > #wrong: all errors included + fill from correct
    o_big = select_gt_indices(n, gt, hard, 0.6, gt_seed=0, strategy="oracle")  # k=6 > 3 wrong
    check("oracle k>#wrong: size 6", len(o_big) == 6)
    check("oracle k>#wrong: all errors included", wrong <= o_big)
    check("oracle k>#wrong: fill from correct rows", (o_big - wrong) <= (set(range(n)) - wrong))
    check("oracle deterministic", o_big == select_gt_indices(n, gt, hard, 0.6, 0, "oracle"))

    # edge: fraction 0 -> empty; fraction 1 -> all
    check("frac 0 -> empty", select_gt_indices(n, gt, hard, 0.0, 0, "oracle") == set())
    check("frac 1 -> all", select_gt_indices(n, gt, hard, 1.0, 0, "oracle") == set(range(n)))


def real_data_test(arrow_path):
    print(f"[real weak_labels] {arrow_path}")
    import pyarrow as pa
    try:
        tbl = pa.ipc.open_stream(pa.memory_map(arrow_path, "r")).read_all()
    except Exception:
        tbl = pa.ipc.open_file(pa.memory_map(arrow_path, "r")).read_all()
    gt = tbl.column("gt_label").to_pylist()
    hard = tbl.column("hard_label").to_pylist()
    n = len(gt)
    wrong = {i for i in range(n) if gt[i] != hard[i]}
    err_rate = len(wrong) / n
    print(f"  rows={n}  weak error rate={err_rate:.3f}  (#wrong={len(wrong)})")

    # at a 10% budget, oracle should be entirely weak-errors (err rate >> 0.10 expected)
    k = round(0.10 * n)
    o = select_gt_indices(n, gt, hard, 0.10, gt_seed=1, strategy="oracle")
    check("real oracle@0.10 size == k", len(o) == k)
    if len(wrong) >= k:
        check("real oracle@0.10 targets only weak errors", o <= wrong)
    naive = select_gt_indices(n, gt, hard, 0.10, gt_seed=1, strategy="naive")
    # oracle's error-hit rate is 100% (if budget<=errors); naive's ~= base error rate
    naive_hit = len(naive & wrong) / len(naive)
    oracle_hit = len(o & wrong) / len(o)
    print(f"  error-coverage: oracle={oracle_hit:.2f} vs naive≈{naive_hit:.2f} (base rate {err_rate:.2f})")
    check("real oracle hits more errors than naive", oracle_hit > naive_hit)


if __name__ == "__main__":
    synthetic_tests()
    default = ("results/data/baseline/seed0/bs=32-dn=boolq-e=2-ee=1000000-lp=0-l=xent-"
               "l=5e-05-ls=cosi_anne-mc=1024-ms=gpt2-nd=20000-ntd=10000-o=adam-s=0-twd=0/"
               "weak_labels/data-00000-of-00001.arrow")
    arrow = sys.argv[1] if len(sys.argv) > 1 else default
    if Path(arrow).exists():
        try:
            real_data_test(arrow)
        except ImportError:
            print("[real weak_labels] skipped (pyarrow not available)")
    else:
        print(f"[real weak_labels] skipped (not found: {arrow})")
    print(f"\nALL {PASS} CHECKS PASSED")
