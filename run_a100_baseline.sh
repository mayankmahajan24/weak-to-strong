#!/bin/bash
RESULTS=/root/weak-to-strong/results/data
cd /root/weak-to-strong

run_model() {
    local gpu=$1 model=$2 mbs=$3 loss=$4 weak=$5 extra=$6
    local loss_arg="" weak_arg=""
    [ "$loss" = "logconf" ] && loss_arg="--loss=logconf"
    [ -n "$weak" ] && weak_arg="--weak_model_size=$weak"
    CUDA_VISIBLE_DEVICES=$gpu python3 train_simple.py \
        --model_size=$model --ds_name=boolq \
        --results_folder=$RESULTS --sweep_subfolder=baseline \
        --minibatch_size_per_device=$mbs $extra \
        $loss_arg $weak_arg 2>&1
    find $RESULTS -name "pytorch_model.bin" -delete 2>/dev/null
}

echo "=== GT: gpt2-xl with max_ctx=512 ($(date -u)) ==="
run_model 0 gpt2-xl 1 xent "" "--max_ctx=512" > /tmp/gt_xl_final.log 2>&1
if [ $? -ne 0 ]; then echo "xl GT FAILED"; tail -3 /tmp/gt_xl_final.log; exit 1; fi
echo "xl GT done at $(date -u)"

echo "=== xent transfers batch 1 ($(date -u)) ==="
run_model 0 gpt2        16 xent gpt2        "" > /tmp/xent_t1.log 2>&1 &
run_model 1 gpt2-medium  8 xent gpt2        "" > /tmp/xent_t2.log 2>&1 &
run_model 2 gpt2-large   4 xent gpt2        "" > /tmp/xent_t3.log 2>&1 &
run_model 3 gpt2-xl      1 xent gpt2        "--max_ctx=512" > /tmp/xent_t4.log 2>&1 &
run_model 4 gpt2-medium  8 xent gpt2-medium "" > /tmp/xent_t5.log 2>&1 &
run_model 5 gpt2-large   4 xent gpt2-medium "" > /tmp/xent_t6.log 2>&1 &
run_model 6 gpt2-xl      1 xent gpt2-medium "--max_ctx=512" > /tmp/xent_t7.log 2>&1 &
run_model 7 gpt2-large   4 xent gpt2-large  "" > /tmp/xent_t8.log 2>&1 &
wait
echo "xent b1 done $(date -u)"
run_model 0 gpt2-xl 1 xent gpt2-large "--max_ctx=512" > /tmp/xent_t9.log 2>&1 &
run_model 1 gpt2-xl 1 xent gpt2-xl    "--max_ctx=512" > /tmp/xent_t10.log 2>&1 &
wait
echo "xent b2 done $(date -u)"

echo "=== logconf transfers batch 1 ($(date -u)) ==="
run_model 0 gpt2        16 logconf gpt2        "" > /tmp/lc_t1.log 2>&1 &
run_model 1 gpt2-medium  8 logconf gpt2        "" > /tmp/lc_t2.log 2>&1 &
run_model 2 gpt2-large   4 logconf gpt2        "" > /tmp/lc_t3.log 2>&1 &
run_model 3 gpt2-xl      1 logconf gpt2        "--max_ctx=512" > /tmp/lc_t4.log 2>&1 &
run_model 4 gpt2-medium  8 logconf gpt2-medium "" > /tmp/lc_t5.log 2>&1 &
run_model 5 gpt2-large   4 logconf gpt2-medium "" > /tmp/lc_t6.log 2>&1 &
run_model 6 gpt2-xl      1 logconf gpt2-medium "--max_ctx=512" > /tmp/lc_t7.log 2>&1 &
run_model 7 gpt2-large   4 logconf gpt2-large  "" > /tmp/lc_t8.log 2>&1 &
wait
echo "lc b1 done $(date -u)"
run_model 0 gpt2-xl 1 logconf gpt2-large "--max_ctx=512" > /tmp/lc_t9.log 2>&1 &
run_model 1 gpt2-xl 1 logconf gpt2-xl    "--max_ctx=512" > /tmp/lc_t10.log 2>&1 &
wait
echo "lc b2 done $(date -u)"

total=$(find $RESULTS/baseline -name "results_summary.json" 2>/dev/null | wc -l)
echo "=== ALL DONE: $total/24 at $(date -u) ==="
