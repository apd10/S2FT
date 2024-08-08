# S2FT_Rebuttal

## Environment Setup

```
conda env create --name s2ft --file=environment.yml
conda activate s2ft
export CUDA_HOME="/usr/local/cuda"
export LIBRARY_PATH="/usr/local/cuda/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

## Prepare Dataset

```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
```

## Fine-tuning Script

```
./train_script/finetune.sh (replace data_path with your own path to commonsense reasoning data) 
```

## Eval Script

```
./eval_script/eval_commensense.sh (replace data_path with your own path to commonsense reasoning data) 
```