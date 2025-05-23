#!/bin/bash

dim=$1
MODEL=meta-llama/Meta-Llama-3-8B
OUTPUT_DIR=./artifacts/results/math/dora.$dim/


datasets=$2
if [ "$datasets" == "" ]
then
	datasets=(AQuA mawps SVAMP gsm8k)
fi

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
    OUTPUT=$OUTPUT_DIR/$dataset
    mkdir -p $OUTPUT
    BATCH_SIZE=8
    echo "accelerate launch --num_processes 1 --main_process_port $master_port eval/run_math_parallel.py \
        --data_path ~/LLM-Adapters/dataset/$dataset/test.json \
        --model_name_or_path $MODEL \
        --per_device_eval_batch_size $BATCH_SIZE \
        --seed 1234 \
        --dtype bf16 \
        --dataset $dataset \
	      --ft_dir  artifacts/models/math/dora.$dim/ \
        --dora \
        --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log "
done


