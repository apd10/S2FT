#!/bin/bash

MODEL=meta-llama/Llama-3.1-8B
OUTPUT_DIR=./results/math/dext/

datasets=(MultiArith gsm8k AddSub AQuA SingleEq SVAMP mawps)

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=16
    accelerate launch --main_process_port $master_port eval/run_math_parallel.py \
        --data_path ~/aditya-india/LLM-Adapters/dataset/$dataset/test.json \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed 1234 \
        --dtype bf16 \
        --dataset $dataset \
	--ft_dir ./outs/math/dext/ \
	--dext \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
done


