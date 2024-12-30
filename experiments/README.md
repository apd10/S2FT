# Benchmark Results

This directory provides the code for reproducing the results in the original paper.

## Installation

```
conda create --name s2ft --python=3.10
conda activate s2ft
pip install -r ../requirements.txt
conda activate s2ft
export CUDA_HOME="/usr/local/cuda"
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

## Prepare Dataset

```bash
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
```

## Fine-tuning and Evaluation on the Commonsense Reasoning Dataset

```bash
./train_script/finetune_commonsense_llama3.sh 
./eval_script/eval_commensense.sh
```

## Fine-tuning and Evaluation on the Arithmetic Reasoning Dataset

```bash
./train_script/finetune_math_llama3.sh 
./eval_script/eval_math.sh
```

Please replace data_path with your own path for both fine-tuning and evaluation.