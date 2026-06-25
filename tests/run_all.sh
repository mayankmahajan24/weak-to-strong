#!/usr/bin/env bash
# Run the full unit suite. The Phase-0/1/2 tests need the project deps (torch, datasets); the
# elicitation tests are numpy-only and run anywhere. Override with PYTHON=/path/to/python.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
fail=0
# torch/datasets-dependent (run on the box or a venv with the full stack)
for t in tests/test_losses.py tests/test_soft_gt.py tests/test_reliability.py \
         tests/test_select_step.py tests/test_label_mixing.py; do
  echo "### $t"
  "$PY" "$t" || fail=$((fail+1))
  echo
done
# elicitation (numpy-only, CPU)
for t in tests/test_extract_activations.py tests/test_probe.py tests/test_ccs.py \
         tests/test_run_elicitation.py; do
  echo "### $t"
  "$PY" "$t" || fail=$((fail+1))
  echo
done
[ "$fail" -eq 0 ] && echo "ALL TEST FILES PASSED" || { echo "$fail TEST FILE(S) FAILED"; exit 1; }
