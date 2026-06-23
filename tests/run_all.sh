#!/usr/bin/env bash
# Run the full unit suite. Requires the project deps (torch, datasets) — the same env
# train_simple.py runs in (the GPU box, or a local venv with torch+datasets+transformers).
# Override the interpreter with PYTHON=/path/to/python tests/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
fail=0
for t in tests/test_losses.py tests/test_soft_gt.py tests/test_reliability.py \
         tests/test_select_step.py tests/test_label_mixing.py; do
  echo "### $t"
  "$PY" "$t" || fail=$((fail+1))
  echo
done
[ "$fail" -eq 0 ] && echo "ALL TEST FILES PASSED" || { echo "$fail TEST FILE(S) FAILED"; exit 1; }
