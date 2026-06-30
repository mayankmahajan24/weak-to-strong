#!/usr/bin/env bash
# Run the full unit suite. The Phase-0/1/2 tests need the project deps (torch, datasets); the
# elicitation tests are numpy-only and run anywhere. Tests whose deps are missing SKIP cleanly
# (they do not count as failures). Install the full stack with `pip install .`.
# Override the interpreter with PYTHON=/path/to/python.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
pass=0 skip=0 fail=0

run_one() {
  echo "### $1"
  "$PY" "$1"; rc=$?
  if [ "$rc" -eq 0 ]; then pass=$((pass+1))
  elif [ "$rc" -eq 77 ]; then skip=$((skip+1))   # 77 = dependency missing, skipped
  else fail=$((fail+1)); fi
  echo
}

# torch/datasets-dependent (run on the box or a venv with the full stack)
for t in tests/test_losses.py tests/test_soft_gt.py tests/test_reliability.py \
         tests/test_select_step.py tests/test_label_mixing.py; do
  run_one "$t"
done
# elicitation (numpy-only, CPU)
for t in tests/test_extract_activations.py tests/test_probe.py tests/test_ccs.py \
         tests/test_run_phase3.py; do
  run_one "$t"
done

echo "$pass passed, $skip skipped, $fail failed"
[ "$fail" -eq 0 ] || exit 1
