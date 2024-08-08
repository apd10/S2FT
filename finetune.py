import copy, re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import json
import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()                # add this if enconter AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer
from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param, safe_get_full_optimizer_state, safe_set_full_optimizer_state
import random
import warnings 
import os

from utils.utils import (
    print_rank_0, 
    to_device, 
    save_hf_format, 
    set_random_seed, 
    get_all_reduce_mean, 
    get_optimizer_grouped_parameters, 
    save_zero_three_model, 
    load_hf_tokenizer, 
    load_json_data, 
    get_output_or_chosen, 
    get_instruction_or_prompt,
    filter_prompt,
    gather_log_probs,
    safe_set_grad,
    moving_average,
    select_heads_based_on_grads,
    load_state_dict_into_model,
    get_cosine_schedule_with_multiple_warmups,
    random_pruning_,
)
from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.data.data_utils import create_prompt_dataset

from utils.module.s2 import convert_qkv_layer_to_s2, convert_mlp_layer_to_s2, convert_s2_to_linear_layer, only_optimize_s2_parameters, make_model_gradient_checkpointing_compatible

from utils.model.model_utils import create_hf_model
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import math
from utils.perf import print_throughput
import time
import argparse


IGNORE_INDEX = -100

BASE_PROMPT = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}
                
### Response:
''' 


BASE_PROMPT_WITH_INPUT = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}
                
### Response:
''' 

def get_alpaca_prompt(example):
    if 'input' in example and example['input'] != '':
        return BASE_PROMPT_WITH_INPUT.format_map({'instruction': example['instruction'], 'input': example['input']})
    else:
        return BASE_PROMPT.format_map({'instruction': example['instruction']})

def get_output_or_chosen(example):
    if 'output' in example:
        return example['output']
    elif 'chosen' in example:
        return example['chosen']
    elif 'answer' in example:
        return example['answer'].split('####')[0].strip()
    elif 'Rationale' in example:
        return example['Rationale']
    elif 'rationale' in example:
        return example['rationale']
    elif 'solution' in example:
        return example['solution']
    else:
        raise ValueError('double check your data format')

def get_instruction_or_prompt(example):
    if 'input' in example and example['input'] != '':
        return example['input']
    elif 'instruction' in example:
        return example['instruction']
    elif 'prompt' in example:
        return example['prompt']
    elif 'question' in example:
        return example['question']
    elif 'Problem' in example:
        return example['Problem']
    elif 'problem' in example:
        return example['problem']
    else:
        raise ValueError('double check your data format')

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    ids_list = tokenizer(
        strings,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=False
    )['input_ids']

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    print_rank_0('-----------------')
    print_rank_0(examples[0])
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, instruction_type: str, args):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_json_data(data_path)       # try both formats
        logging.warning("Formatting inputs...")
        
        # We might want to clean this up, it's a bit messy
        if instruction_type == 'single':
            print_rank_0('single-round conversation', args.global_rank)
            if 'chat' not in args.model_name_or_path:
                print_rank_0('base model', args.global_rank)
                if 'alpaca' in data_path:
                    sources = [get_alpaca_prompt(example) for example in list_data_dict]
                else:
                    sources = [
                        BASE_PROMPT.format_map({'instruction': get_instruction_or_prompt(example)}) 
                        for example in list_data_dict
                    ]
            else:
                print_rank_0('chat model', args.global_rank)
                sources = []
                for example in list_data_dict:
                    chat = [
                        {'role': 'system', 'content': "You are a helpful assistant. You will be given a user's question, and you need to answer it."},
                        {'role': 'user', 'content': get_instruction_or_prompt(example)}
                    ]
                    source = tokenizer.apply_chat_template(chat, tokenize = False)
                    source += ' '
                    sources.append(source)
        
        # add EOS token (in case some sentence already has </s>, we remove it first), chat template add one space after </s>
        if 'chat' in args.model_name_or_path:
            targets = [f"{get_output_or_chosen(example).replace('</s>', '')} {tokenizer.eos_token}" for example in list_data_dict]    
        else:
            targets = [f"{get_output_or_chosen(example).replace('</s>', '')}{tokenizer.eos_token}" for example in list_data_dict]         

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def int_or_float(value):
    try:
        float_val = float(value)
        if float_val.is_integer():
            return int(float_val)
        else:
            return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be an int or a float")

def random_integers_sum_to_target(n, target):
    random_numbers = [random.randint(0, target) for _ in range(n-1)]
    random_numbers.append(0)
    random_numbers.append(target)
    random_numbers.sort()
    result = [random_numbers[i+1] - random_numbers[i] for i in range(n)]
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Forgetting")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['my_data/train_year1.json'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    # if you only do SFT or if you find it's fine/helpful to use overlapping data in different steps (which is possible).
    parser.add_argument('--data_split',
                        type=str,
                        default='10,0,0',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60%% of data for phase 1, 20%% for phase 2'
                        'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/my_data',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=0,
        help="Size of the validation set. If 0, no validation set is used.",
    )
    parser.add_argument('--load_last_model',
                        action='store_true',
                        help='only save the last model')
    parser.add_argument(
        "--eval_step",
        type=int,
        default=80,
        help="size of eval_step",
    )
    parser.add_argument(
        "--eval_delay",
        type=int_or_float,
        default=0,
        help="eval after certain steps if it is an integer, or eval after certain ratio of steps if it is a float",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.") 
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Training data type')
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='dropout rate of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')

    # SAU
    parser.add_argument('--s2',
                        action='store_true',
                        help='use S2FT for efficient training.')
    parser.add_argument('--kv',
                        action='store_true',
                        help='only select qv')
    parser.add_argument('--k',
                        action='store_true',
                        help='only select k')
    parser.add_argument('--qo',
                        action='store_true',
                        help='select qkv')
    parser.add_argument('--o',
                        action='store_true',
                        help='select o')
    parser.add_argument('--q_ratio',
                        type=float,
                        default=0.0,)
    parser.add_argument('--v_ratio',
                        type=float,
                        default=0.0)
    parser.add_argument('--ug',
                        action='store_true',
                        help='only select ud')
    parser.add_argument('--g',
                        action='store_true',
                        help='only select g')
    parser.add_argument('--u_ratio',
                        type=float,
                        default=0.0)
    parser.add_argument('--d_ratio',
                        type=float,
                        default=0.0)
    parser.add_argument('--only_optimize_s2',
                        action='store_true',
                        help='Only optimize the S2FT parameters.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    parser.add_argument('--instruction_type',
                        type=str,
                        choices=['single', 'multi'],
                        default="single")
    parser.add_argument("--regularization_strength",
                        type=float,
                        default=1e-6,
                        help="regularization")
    parser.add_argument("--save_interval",
                        type=int,
                        default=500,
                        help="save deepspeed engine checkpoint for recover the training")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args



def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="Forgetting",
                                    profiler_path=args.output_dir)
    
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)         # add bos_token = False
    tokenizer.model_max_length = args.max_seq_len
    print_rank_0(f"Tokenizer: {tokenizer.model_max_length}", args.global_rank)

    print_rank_0(f"Loading model from {args.model_name_or_path}", args.global_rank)
    model = create_hf_model(AutoModelForCausalLM,
                        args.model_name_or_path,
                        tokenizer,
                        ds_config,
                        dropout=args.dropout)
    if args.s2:
        print_rank_0('------use S2FT------', args.global_rank)
        
        parameters_q = {}
        parameters_v = {}
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        mha_indices = [i for i in range(model.config.num_attention_heads * model.config.num_hidden_layers)]
        print_rank_0(f'head_dim: {head_dim}', args.global_rank)
        print_rank_0(f'hidden_size: {model.config.hidden_size}', args.global_rank)
        selected_parameters_mha = None
        for i in range(model.config.num_hidden_layers):
            parameters_q[i] = []
            parameters_v[i] = []
        num_q = int(model.config.num_attention_heads * model.config.num_hidden_layers * args.q_ratio)
        num_v = int(model.config.num_attention_heads * model.config.num_hidden_layers * args.v_ratio)
        select_q = sorted(random.sample(mha_indices, num_q))
        for q in select_q:
            parameters_q[q // model.config.num_attention_heads].append(q % model.config.num_attention_heads)
        select_v = sorted(random.sample(mha_indices, num_v))
        for v in select_v:
            parameters_v[v // model.config.num_attention_heads].append(v % model.config.num_attention_heads)
        selected_parameters_mha = {'q_proj': parameters_q, 'v_proj': parameters_v}
        model = convert_qkv_layer_to_s2(model, selected_parameters = selected_parameters_mha, kv=args.kv, qo=args.qo, k=args.k, o=args.o)

        parameters_u = {}
        parameters_d = {}
        intermediate_dim = model.config.intermediate_size
        mlp_indices = [i for i in range(intermediate_dim * model.config.num_hidden_layers)]
        print_rank_0(f'intermediate__dim: {intermediate_dim}', args.global_rank)
        print_rank_0(f'hidden_size: {model.config.hidden_size}', args.global_rank)
        selected_parameters_mlp = None
        for i in range(model.config.num_hidden_layers):
            parameters_u[i] = []
            parameters_d[i] = []
        num_u = int(intermediate_dim * model.config.num_hidden_layers * args.u_ratio)
        num_d = int(intermediate_dim * model.config.num_hidden_layers * args.d_ratio)
        select_u = sorted(random.sample(mlp_indices, num_u))
        for u in select_u:
            parameters_u[u // intermediate_dim].append(u % intermediate_dim)
        select_d = sorted(random.sample(mlp_indices, num_d))
        for d in select_d:
            parameters_d[d // intermediate_dim].append(d % intermediate_dim)
        selected_parameters_mlp = {'up_proj': parameters_u, 'down_proj': parameters_d}
        model = convert_mlp_layer_to_s2(model, selected_parameters = selected_parameters_mlp, ug=args.ug, g=args.g)

        if args.only_optimize_s2:
            model = only_optimize_s2_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

        print_rank_0(f'learning rate: {args.learning_rate}', args.global_rank)

    if len(args.data_path) == 1 and '.json' in args.data_path[0]:                  
        print_rank_0(f"------json Data: {args.data_path[0]}", args.global_rank)
        if 'pairwise' in args.data_path[0] and args.instruction_type != 'single':
            raise ValueError('pairwise data has to be single-round conversation')
        train_dataset = SupervisedDataset(
            data_path = args.data_path[0],
            tokenizer = tokenizer,
            instruction_type = args.instruction_type,
            args = args,
        )
        if args.val_set_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [len(train_dataset) - args.val_set_size, args.val_set_size])
            print_rank_0(f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}", args.global_rank)
        
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        raise ValueError('Only json format is supported for now. Please check your data format.')
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        if args.val_set_size > 0:
            val_sampler = SequentialSampler(val_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        if args.val_set_size > 0:
            val_sampler = DistributedSampler(val_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    if args.val_set_size > 0:
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.learning_rate)

    # deepspeed
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()
    
    # Training
    print_rank_0("***** Running training *****", args.global_rank)
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    print_rank_0(formatted_args, args.global_rank)

    # print trainable parameters
    with deepspeed.zero.GatheredParameters(model.module.parameters(), modifier_rank=0, enabled=args.zero_stage == 3):
        num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print_rank_0(f"Number of trainable parameters: {num}", args.global_rank)
    
    total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    current_step_count = 0

    lr_plot = []
    best_val_loss = float('inf')
    final_saved_model_index = 0
    best_model = None

    args.eval_step = args.eval_step * args.gradient_accumulation_steps

    args.eval_delay = args.eval_delay if isinstance(args.eval_delay, int) else int(args.eval_delay * total_training_steps)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        mean_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            lr_plot.append(lr_scheduler.get_last_lr()[1] if len(lr_scheduler.get_last_lr()) > 1 else lr_scheduler.get_last_lr()[0])
            start = time.time()
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            total_loss = loss
 
            model.backward(total_loss)
            model.step()
            current_step_count += 1

            end = time.time()
            # print throughput every 100 steps 
            if torch.distributed.get_rank() == 0 and step % 100 == 0:
                print_throughput(model.model, args, end - start,
                                 args.global_rank)
            mean_loss += loss.item()

            #print_rank_0(f"{current_step_count}, {args.eval_step}, {args.val_set_size}, {current_step_count >= args.eval_delay}, {args.load_last_model}", args.global_rank)

            if current_step_count % args.eval_step == 0 and args.val_set_size > 0 and not args.load_last_model and current_step_count >= args.eval_delay:
                ppl, val_loss = evaluation(model, val_dataloader)
                print_rank_0(f"Validation perplexity: {ppl}, Validation loss: {val_loss}", args.global_rank)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.zero_stage != 3:
                        if args.global_rank == 0:
                            best_model = copy.deepcopy(model.module).to('cpu')
                    else:
                        output_state_dict = {}
                        for k, v in model.module.named_parameters():
                            if hasattr(v, 'ds_id'):
                                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                                            ]),
                                                                    enabled=args.zero_stage == 3):
                                    v_p = v.data.cpu()
                            else:
                                v_p = v.cpu()
                            if args.global_rank == 0:
                                output_state_dict[k] = v_p
                        best_model = output_state_dict
                    final_saved_model_index = current_step_count // args.eval_step

        print_rank_0(
                f"Epoch {epoch+1}/{args.num_train_epochs} Train loss: {mean_loss/len(train_dataloader)}", args.global_rank)
        
        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        # evaluate last model
        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluation(model, val_dataloader)
            print_rank_0(f"Validation perplexity: {ppl}, Validation loss: {val_loss}", args.global_rank)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.zero_stage != 3:
                    if args.global_rank == 0:
                        best_model = copy.deepcopy(model.module).to('cpu')
                else:
                    output_state_dict = {}
                    for k, v in model.module.named_parameters():
                        if hasattr(v, 'ds_id'):
                            with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                                        ]),
                                                                enabled=args.zero_stage == 3):
                                v_p = v.data.cpu()
                        else:
                            v_p = v.cpu()
                        if args.global_rank == 0:
                            output_state_dict[k] = v_p
                    best_model = output_state_dict
                final_saved_model_index = 'last'
            print_rank_0(f"Best validation loss: {best_val_loss}", args.global_rank)
            print_rank_0(f"Savings the best model at step {final_saved_model_index}", args.global_rank)

        if args.load_last_model:
            print_rank_0('only load the last model ...', args.global_rank)
        
        # if best model is dict, we load the state dict
        # which means zero3
        if isinstance(best_model, dict):
            load_state_dict_into_model(model.module, best_model, "", zero_stage=args.zero_stage)
        else:
            model = best_model.to(device) if best_model else model

        if args.s2:
            print_rank_0('converting s2 to linear layer ...', args.global_rank)
            model = convert_s2_to_linear_layer(model)
            
        if args.global_rank == 0:
            print_rank_0('saving the model ...', args.global_rank)
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)

        torch.save(lr_plot, os.path.join(args.output_dir, 'lr_plot.pt'))



if __name__ == "__main__":
    main()
            