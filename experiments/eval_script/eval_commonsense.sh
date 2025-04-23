#!/bin/bash

MODEL=meta-llama/Llama-3.1-8B
dext_models=$(ls -d /home/ubuntu/PEFT_ARTIFACTS/models/commonsense/dext.*)
dora_models=$(ls -d /home/ubuntu/PEFT_ARTIFACTS/models/commonsense/dora.[0-9] /home/ubuntu/PEFT_ARTIFACTS/models/commonsense/dora.[0-9][0-9])

datasets=(boolq piqa social_i_qa ARC-Challenge ARC-Easy openbookqa hellaswag winogrande)

master_port=$((RANDOM % 5000 + 20000))
for dataset in "${datasets[@]}"; do
	for model in $dext_models; do 
		OUTPUT_DIR=/home/ubuntu/NEW_ARTIFACTS/results/$(basename $model)/commonsense/
    		OUTPUT=$OUTPUT_DIR/$dataset
    		mkdir -p $OUTPUT
    		BATCH_SIZE=32
    		echo "accelerate launch --main_process_port $master_port eval/run_commonsense_parallel.py \
        		--data_path ~/LLM-Adapters/dataset/$dataset/test.json \
        		--model_name_or_path $MODEL \
        		--per_device_eval_batch_size $BATCH_SIZE \
        		--seed 1234 \
        		--dtype bf16 \
        		--dataset $dataset \
			--ft_dir $model \
			--dext \
        		--output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
			"
	done

	for model in $dora_models; do 
		OUTPUT_DIR=/home/ubuntu/NEW_ARTIFACTS/results/$(basename $model)/commonsense/
    		OUTPUT=$OUTPUT_DIR/$dataset
    		mkdir -p $OUTPUT
    		BATCH_SIZE=32
    		echo "accelerate launch --main_process_port $master_port eval/run_commonsense_parallel.py \
        		--data_path ~/LLM-Adapters/dataset/$dataset/test.json \
        		--model_name_or_path $MODEL \
        		--per_device_eval_batch_size $BATCH_SIZE \
        		--seed 1234 \
        		--dtype bf16 \
        		--dataset $dataset \
			--ft_dir $model \
			--dora \
        		--output_dir $OUTPUT 2> >(tee $OUTPUT/err.log >&2) | tee $OUTPUT/training.log \ 
			"
	done

done


