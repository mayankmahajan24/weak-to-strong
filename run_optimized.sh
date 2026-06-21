#!/bin/bash
set -e
RESULTS=/home/ubuntu/weak-to-strong/results/data
cd /home/ubuntu/weak-to-strong

run_model() {
    local gpu=$1 ds=$2 model=$3 mbs=$4 weak_model=$5
    local weak_arg=""
    if [ -n "$weak_model" ]; then
        weak_arg="--weak_model_size=$weak_model"
    fi
    CUDA_VISIBLE_DEVICES=$gpu python train_simple.py \
        --model_size=$model --ds_name=$ds --results_folder=$RESULTS \
        --minibatch_size_per_device=$mbs $weak_arg 2>&1
}

# Minibatch sizes per model, tuned for long-sequence datasets on A100 80GB
# Short sequences (sciq, cosmos_qa): can use larger batches
# Long sequences (boolq, amazon_polarity, anthropic_hh): need smaller batches
get_mbs() {
    local ds=$1 model=$2
    case $model in
        gpt2)
            case $ds in
                anthropic_hh) echo 16 ;;
                *)            echo 32 ;;
            esac ;;
        gpt2-medium)
            case $ds in
                anthropic_hh|boolq) echo 4 ;;
                *)                  echo 16 ;;
            esac ;;
        gpt2-large)
            case $ds in
                anthropic_hh) echo 1 ;;
                *)            echo 2 ;;
            esac ;;
        gpt2-xl)
            echo 1 ;;
    esac
}

run_dataset() {
    local ds=$1
    local mbs_gpt2=$(get_mbs $ds gpt2)
    local mbs_medium=$(get_mbs $ds gpt2-medium)
    local mbs_large=$(get_mbs $ds gpt2-large)
    local mbs_xl=$(get_mbs $ds gpt2-xl)

    echo "========================================="
    echo "DATASET: $ds — Phase 1: Ground Truth ($(date -u))"
    echo "  minibatch sizes: gpt2=$mbs_gpt2, medium=$mbs_medium, large=$mbs_large, xl=$mbs_xl"
    echo "========================================="

    run_model 0 "$ds" gpt2        $mbs_gpt2   "" > /tmp/gt_gpt2.log 2>&1 &
    p1=$!
    run_model 1 "$ds" gpt2-medium $mbs_medium "" > /tmp/gt_medium.log 2>&1 &
    p2=$!
    run_model 2 "$ds" gpt2-large  $mbs_large  "" > /tmp/gt_large.log 2>&1 &
    p3=$!
    run_model 3 "$ds" gpt2-xl     $mbs_xl     "" > /tmp/gt_xl.log 2>&1 &
    p4=$!
    failed=0
    for p in $p1 $p2 $p3 $p4; do wait $p || { failed=1; echo "Process $p failed"; }; done
    if [ $failed -eq 1 ]; then
        echo "ERROR: Ground truth failed for $ds"
        for f in /tmp/gt_*.log; do echo "=== $f ==="; tail -3 "$f"; done
        return 1
    fi
    echo "Ground truth done at $(date -u)"

    echo "Phase 2: Transfers ($(date -u))"
    # Phase 2a: 8 transfers in parallel across 8 GPUs
    run_model 0 "$ds" gpt2        $mbs_gpt2   gpt2        > /tmp/t1.log 2>&1 &
    run_model 1 "$ds" gpt2-medium $mbs_medium gpt2        > /tmp/t2.log 2>&1 &
    run_model 2 "$ds" gpt2-large  $mbs_large  gpt2        > /tmp/t3.log 2>&1 &
    run_model 3 "$ds" gpt2-xl     $mbs_xl     gpt2        > /tmp/t4.log 2>&1 &
    run_model 4 "$ds" gpt2-medium $mbs_medium gpt2-medium > /tmp/t5.log 2>&1 &
    run_model 5 "$ds" gpt2-large  $mbs_large  gpt2-medium > /tmp/t6.log 2>&1 &
    run_model 6 "$ds" gpt2-xl     $mbs_xl     gpt2-medium > /tmp/t7.log 2>&1 &
    run_model 7 "$ds" gpt2-large  $mbs_large  gpt2-large  > /tmp/t8.log 2>&1 &
    wait
    echo "Transfer batch 1 done at $(date -u)"

    # Phase 2b: remaining 2 transfers
    run_model 0 "$ds" gpt2-xl $mbs_xl gpt2-large > /tmp/t9.log 2>&1 &
    run_model 1 "$ds" gpt2-xl $mbs_xl gpt2-xl    > /tmp/t10.log 2>&1 &
    wait
    echo "Transfer batch 2 done at $(date -u)"

    echo "DATASET $ds COMPLETE at $(date -u)"
    echo "========================================="
}

# Run datasets in priority order
for ds in boolq cosmos_qa amazon_polarity anthropic_hh; do
    run_dataset "$ds" || echo "WARNING: $ds had failures, continuing to next dataset"
done

echo "ALL DATASETS COMPLETE at $(date -u)"
