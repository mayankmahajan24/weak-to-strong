#!/bin/bash
set -e

RESULTS=/root/weak-to-strong/results/data
cd /root/weak-to-strong

START_TIME=$(date +%s)

echo "=== BoolQ Baseline Sweep (A100 80GB x8) ==="
echo "Start time: $(date)"

# ============================================
# PHASE 1: Ground truth (4 runs on GPUs 0-3)
# ============================================
echo ""
echo "=== PHASE 1: Ground Truth (4 runs) ==="
echo "Started at: $(date)"

CUDA_VISIBLE_DEVICES=0 python3 train_simple.py --model_size=gpt2 --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=32 > /tmp/gt_gpt2.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 train_simple.py --model_size=gpt2-medium --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=16 > /tmp/gt_medium.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python3 train_simple.py --model_size=gpt2-large --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=4 > /tmp/gt_large.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python3 train_simple.py --model_size=gpt2-xl --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/gt_xl.log 2>&1 &
P3=$!

echo "Waiting for ground truth runs..."
FAIL=0
for pid in $P0 $P1 $P2 $P3; do
    wait $pid || FAIL=1
done
if [ $FAIL -ne 0 ]; then
    echo "ERROR: Ground truth phase failed! Check /tmp/gt_*.log"
    for f in /tmp/gt_gpt2.log /tmp/gt_medium.log /tmp/gt_large.log /tmp/gt_xl.log; do echo "=== $f ==="; tail -5 "$f"; done
    exit 1
fi
echo "Phase 1 complete at: $(date)"
echo "Results so far: $(find $RESULTS/baseline -name results_summary.json 2>/dev/null | wc -l)"

echo "Cleaning up Phase 1 model files..."
find $RESULTS/baseline -name "pytorch_model.bin" -delete 2>/dev/null || true
find $RESULTS/baseline -name "model.safetensors" -delete 2>/dev/null || true

# ============================================
# PHASE 2: xent transfers (10 runs in 2 batches)
# ============================================
echo ""
echo "=== PHASE 2: xent transfers (10 runs) ==="
echo "Started at: $(date)"

# Batch 2a: 8 runs on GPUs 0-7
CUDA_VISIBLE_DEVICES=0 python3 train_simple.py --model_size=gpt2 --weak_model_size=gpt2 --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=32 > /tmp/xent_t1.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 train_simple.py --model_size=gpt2-medium --weak_model_size=gpt2 --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=16 > /tmp/xent_t2.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python3 train_simple.py --model_size=gpt2-large --weak_model_size=gpt2 --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=4 > /tmp/xent_t3.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2 --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/xent_t4.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=4 python3 train_simple.py --model_size=gpt2-medium --weak_model_size=gpt2-medium --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=16 > /tmp/xent_t5.log 2>&1 &
P4=$!
CUDA_VISIBLE_DEVICES=5 python3 train_simple.py --model_size=gpt2-large --weak_model_size=gpt2-medium --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=4 > /tmp/xent_t6.log 2>&1 &
P5=$!
CUDA_VISIBLE_DEVICES=6 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2-medium --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/xent_t7.log 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=7 python3 train_simple.py --model_size=gpt2-large --weak_model_size=gpt2-large --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=4 > /tmp/xent_t8.log 2>&1 &
P7=$!

echo "Waiting for xent batch 1 (8 runs)..."
FAIL=0
for pid in $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7; do
    wait $pid || FAIL=1
done
if [ $FAIL -ne 0 ]; then
    echo "ERROR: xent batch 1 failed!"
    for f in /tmp/xent_t1.log /tmp/xent_t2.log /tmp/xent_t3.log /tmp/xent_t4.log /tmp/xent_t5.log /tmp/xent_t6.log /tmp/xent_t7.log /tmp/xent_t8.log; do echo "=== $f ==="; tail -5 "$f"; done
    exit 1
fi
echo "Batch 2a complete at: $(date)"

# Batch 2b: 2 runs on GPUs 0-1
CUDA_VISIBLE_DEVICES=0 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2-large --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/xent_t9.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2-xl --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/xent_t10.log 2>&1 &
P1=$!

echo "Waiting for xent batch 2 (2 runs)..."
FAIL=0
for pid in $P0 $P1; do wait $pid || FAIL=1; done
if [ $FAIL -ne 0 ]; then
    echo "ERROR: xent batch 2 failed!"
    for f in /tmp/xent_t9.log /tmp/xent_t10.log; do echo "=== $f ==="; tail -5 "$f"; done
    exit 1
fi
echo "Batch 2b complete at: $(date)"

echo "Results so far: $(find $RESULTS/baseline -name results_summary.json 2>/dev/null | wc -l)"

echo "Cleaning up Phase 2 model files..."
find $RESULTS/baseline -name "pytorch_model.bin" -delete 2>/dev/null || true
find $RESULTS/baseline -name "model.safetensors" -delete 2>/dev/null || true

# ============================================
# PHASE 3: logconf transfers (10 runs in 2 batches)
# ============================================
echo ""
echo "=== PHASE 3: logconf transfers (10 runs) ==="
echo "Started at: $(date)"

# Batch 3a: 8 runs on GPUs 0-7
CUDA_VISIBLE_DEVICES=0 python3 train_simple.py --model_size=gpt2 --weak_model_size=gpt2 --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=32 > /tmp/logconf_t1.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 train_simple.py --model_size=gpt2-medium --weak_model_size=gpt2 --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=16 > /tmp/logconf_t2.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python3 train_simple.py --model_size=gpt2-large --weak_model_size=gpt2 --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=4 > /tmp/logconf_t3.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=3 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2 --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/logconf_t4.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=4 python3 train_simple.py --model_size=gpt2-medium --weak_model_size=gpt2-medium --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=16 > /tmp/logconf_t5.log 2>&1 &
P4=$!
CUDA_VISIBLE_DEVICES=5 python3 train_simple.py --model_size=gpt2-large --weak_model_size=gpt2-medium --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=4 > /tmp/logconf_t6.log 2>&1 &
P5=$!
CUDA_VISIBLE_DEVICES=6 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2-medium --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/logconf_t7.log 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=7 python3 train_simple.py --model_size=gpt2-large --weak_model_size=gpt2-large --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=4 > /tmp/logconf_t8.log 2>&1 &
P7=$!

echo "Waiting for logconf batch 1 (8 runs)..."
FAIL=0
for pid in $P0 $P1 $P2 $P3 $P4 $P5 $P6 $P7; do
    wait $pid || FAIL=1
done
if [ $FAIL -ne 0 ]; then
    echo "ERROR: logconf batch 1 failed!"
    for f in /tmp/logconf_t1.log /tmp/logconf_t2.log /tmp/logconf_t3.log /tmp/logconf_t4.log /tmp/logconf_t5.log /tmp/logconf_t6.log /tmp/logconf_t7.log /tmp/logconf_t8.log; do echo "=== $f ==="; tail -5 "$f"; done
    exit 1
fi
echo "Batch 3a complete at: $(date)"

# Batch 3b: 2 runs on GPUs 0-1
CUDA_VISIBLE_DEVICES=0 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2-large --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/logconf_t9.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python3 train_simple.py --model_size=gpt2-xl --weak_model_size=gpt2-xl --loss=logconf --ds_name=boolq --results_folder=$RESULTS --sweep_subfolder=baseline --minibatch_size_per_device=2 > /tmp/logconf_t10.log 2>&1 &
P1=$!

echo "Waiting for logconf batch 2 (2 runs)..."
FAIL=0
for pid in $P0 $P1; do wait $pid || FAIL=1; done
if [ $FAIL -ne 0 ]; then
    echo "ERROR: logconf batch 2 failed!"
    for f in /tmp/logconf_t9.log /tmp/logconf_t10.log; do echo "=== $f ==="; tail -5 "$f"; done
    exit 1
fi
echo "Batch 3b complete at: $(date)"

echo "Results so far: $(find $RESULTS/baseline -name results_summary.json 2>/dev/null | wc -l)"

echo "Cleaning up Phase 3 model files..."
find $RESULTS/baseline -name "pytorch_model.bin" -delete 2>/dev/null || true
find $RESULTS/baseline -name "model.safetensors" -delete 2>/dev/null || true

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
echo ""
echo "=== SWEEP COMPLETE ==="
echo "End time: $(date)"
echo "Total elapsed: ${ELAPSED} minutes"
echo "Total results: $(find $RESULTS/baseline -name results_summary.json 2>/dev/null | wc -l) / 24"
