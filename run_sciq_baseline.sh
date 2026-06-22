#!/bin/bash
RESULTS=/root/weak-to-strong/results/data
cd /root/weak-to-strong

run_model() {
    local gpu=$1 model=$2 mbs=$3 loss=$4 weak=$5
    local loss_arg="" weak_arg=""
    [ "$loss" = "logconf" ] && loss_arg="--loss=logconf"
    [ -n "$weak" ] && weak_arg="--weak_model_size=$weak"
    CUDA_VISIBLE_DEVICES=$gpu python3 train_simple.py \
        --model_size=$model --ds_name=sciq \
        --results_folder=$RESULTS --sweep_subfolder=baseline \
        --minibatch_size_per_device=$mbs \
        $loss_arg $weak_arg 2>&1
    find $RESULTS -name "pytorch_model*" -delete 2>/dev/null
}

echo "=== Phase 1: sciq GT ($(date -u)) ==="
run_model 0 gpt2        16 xent "" > /tmp/sciq_gt_gpt2.log 2>&1 &
run_model 1 gpt2-medium  8 xent "" > /tmp/sciq_gt_medium.log 2>&1 &
run_model 2 gpt2-large   4 xent "" > /tmp/sciq_gt_large.log 2>&1 &
run_model 3 gpt2-xl      1 xent "" > /tmp/sciq_gt_xl.log 2>&1 &
wait
echo "GT done at $(date -u)"

echo "=== Phase 2: sciq xent transfers ($(date -u)) ==="
run_model 0 gpt2        16 xent gpt2        > /tmp/sciq_xt1.log 2>&1 &
run_model 1 gpt2-medium  8 xent gpt2        > /tmp/sciq_xt2.log 2>&1 &
run_model 2 gpt2-large   4 xent gpt2        > /tmp/sciq_xt3.log 2>&1 &
run_model 3 gpt2-xl      1 xent gpt2        > /tmp/sciq_xt4.log 2>&1 &
run_model 4 gpt2-medium  8 xent gpt2-medium > /tmp/sciq_xt5.log 2>&1 &
run_model 5 gpt2-large   4 xent gpt2-medium > /tmp/sciq_xt6.log 2>&1 &
run_model 6 gpt2-xl      1 xent gpt2-medium > /tmp/sciq_xt7.log 2>&1 &
run_model 7 gpt2-large   4 xent gpt2-large  > /tmp/sciq_xt8.log 2>&1 &
wait
echo "xent b1 done $(date -u)"
run_model 0 gpt2-xl 1 xent gpt2-large > /tmp/sciq_xt9.log 2>&1 &
run_model 1 gpt2-xl 1 xent gpt2-xl    > /tmp/sciq_xt10.log 2>&1 &
wait
echo "xent b2 done $(date -u)"

echo "=== Phase 3: sciq logconf transfers ($(date -u)) ==="
run_model 0 gpt2        16 logconf gpt2        > /tmp/sciq_lc1.log 2>&1 &
run_model 1 gpt2-medium  8 logconf gpt2        > /tmp/sciq_lc2.log 2>&1 &
run_model 2 gpt2-large   4 logconf gpt2        > /tmp/sciq_lc3.log 2>&1 &
run_model 3 gpt2-xl      1 logconf gpt2        > /tmp/sciq_lc4.log 2>&1 &
run_model 4 gpt2-medium  8 logconf gpt2-medium > /tmp/sciq_lc5.log 2>&1 &
run_model 5 gpt2-large   4 logconf gpt2-medium > /tmp/sciq_lc6.log 2>&1 &
run_model 6 gpt2-xl      1 logconf gpt2-medium > /tmp/sciq_lc7.log 2>&1 &
run_model 7 gpt2-large   4 logconf gpt2-large  > /tmp/sciq_lc8.log 2>&1 &
wait
echo "logconf b1 done $(date -u)"
run_model 0 gpt2-xl 1 logconf gpt2-large > /tmp/sciq_lc9.log 2>&1 &
run_model 1 gpt2-xl 1 logconf gpt2-xl    > /tmp/sciq_lc10.log 2>&1 &
wait
echo "logconf b2 done $(date -u)"

total=$(find $RESULTS/baseline -name "results_summary.json" -path "*sciq*" 2>/dev/null | wc -l)
echo "=== SCIQ DONE: $total/24 at $(date -u) ==="
