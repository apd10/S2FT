#!/bin/bash

datasets=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)

master_port=$((RANDOM % 5000 + 20000))
MODEL=./outs/s2_random_mlp_seed_42
for dataset in "${datasets[@]}"; do
    OUTPUT=./results/s2_random_mlp_seed_42
    ZERO_STAGE=1
    mkdir -p $OUTPUT
    BATCH_SIZE=8
    accelerate launch --main_process_port $master_port evaluation/run_commonsense_parallel.py \
        --data_path ./LLM-Adapters/dataset/$dataset/test.json \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed 1234 \
        --dtype bf16 \
        --dataset $dataset \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
done


