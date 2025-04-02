#!/bin/bash


OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./outs/commonsense/lora_llama3
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=1                    # lets use zero_stage 1 for now
fi
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
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --num_train_epochs 1 \
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
    --eval_step 50 \
    --lora \
    --lora_dim 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_module_name up_proj down_proj \
    --data_path  ~/LLM-Adapters/ft-training_set/commonsense_15k.json  \
    --eval_delay 0 \
    --output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \
