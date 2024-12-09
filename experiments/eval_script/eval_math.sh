#!/bin/bash

MODEL=$1
OUTPUT_DIR=$2

if [ "$OUTPUT_DIR" == "" ]; then
    OUTPUT_DIR=./results/$MODEL/math
fi

datasets=(MultiArith gsm8k AddSub AQuA SingleEq SVAMP mawps)

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=16
    accelerate launch --main_process_port $master_port eval/run_math_parallel.py \
        --data_path ~/LLM-Adapters/dataset/$dataset/test.json \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed 1234 \
        --dtype bf16 \
        --dataset $dataset \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
done


