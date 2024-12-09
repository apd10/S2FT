#!/bin/bash


MODEL=$1
OUTPUT=$2
ZERO_STAGE=$3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./outs/math/s4_random_all_llama2
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=1                 
fi
mkdir -p $OUTPUT

master_port=$((RANDOM % 5000 + 20000))
# add offload add master_port if socket error
deepspeed --include=localhost:0,1,2,3 \
    --master_port $master_port ./train/finetune.py \
    --offload \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_seq_len 2048 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --dtype bf16 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type linear \
    --num_warmup_steps 100 \
    --seed 42 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --gradient_checkpointing \
    --save_interval 5000 \
    --instruction_type single \
    --load_last_model \
    --s2 \
    --v_ratio 0.0 \
    --o_ratio 0.052 \
    --u_ratio 0.0 \
    --d_ratio 0.02 \
    --data_path  ~/LLM-Adapters/ft-training_set/math_10k.json  \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \
