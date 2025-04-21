#!/bin/bash

dataset=commonsense
lr=5e-5
dim=$1
OUTPUT=./artifacts/models/$dataset/dora.$dim.LR$lr
ZERO_STAGE=1
mkdir -p $OUTPUT

master_port=$((RANDOM % 5000 + 20000))
# add offload add master_port if socket error
deepspeed --include=localhost:0 \
    --master_port $master_port ./train/finetune.py \
    --offload \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --max_seq_len 2048 \
    --learning_rate $lr \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --dtype bf16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type linear \
    --num_warmup_steps 100 \
    --seed 0 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --gradient_checkpointing \
    --save_interval 5000 \
    --instruction_type single \
    --val_set_size 120 \
    --eval_step 80 \
    --dora \
    --dora_config_file ./dora/lora_$dim.json \
    --data_path  ~/LLM-Adapters/ft-training_set/commonsense_170k.json  \
    --eval_delay 0 \
    --early_stop_window 0.2 \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \

