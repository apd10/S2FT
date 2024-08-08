# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator
import torch.nn as nn
import tqdm
from transformers import StoppingCriteria
from deepspeed import comm as dist
from typing import Dict
import logging
import torch.nn.functional as F
import warnings
import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param, safe_get_full_optimizer_state, safe_set_full_optimizer_state

def print_trainable_parameters(model, rank=None, zero_stage=0):
    with deepspeed.zero.GatheredParameters(model.module.parameters(), modifier_rank=0, enabled=zero_stage == 3):
        trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.module.parameters()) 
    msg =  f"model params: {all_params:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_params}"
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)

def set_grad(param, value):
    if param._hp_mapping is not None:
        hp_mapping = param._hp_mapping
        if hp_mapping.use_offload:
            gradient_dict = hp_mapping.offload_gradient_dict
        else:
            gradient_dict = hp_mapping.gradient_dict
        if hp_mapping.param_group_index not in gradient_dict or gradient_dict[hp_mapping.param_group_index] is None:
            raise ValueError("Gradients are only available immediately after backward and before engine step")
        lp_grad_fragment = gradient_dict[hp_mapping.param_group_index][param._index_in_param_group]
        hp_grad_fragment = lp_grad_fragment.to(torch.float32).flatten() # 32008 * 4096
        lp_frag_address = param._hp_mapping.lp_fragment_address
        value_fragment = torch.narrow(value.flatten(), 0, lp_frag_address.start, lp_frag_address.numel)
        hp_grad_fragment.copy_(value_fragment.data)

def safe_set_grad(param, value):
    if param.grad is not None:
        param.grad.data = value
    else:
        # let assume only zero 1 and 2 and bf16
        if hasattr(param, '_hp_mapping'):
            set_grad(param, value)
        # zero 3
        if hasattr(param, 'ds_id'):   
            warnings.warn("Might not WORK!")
            set_grad(param._z3_optimizer, value)


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def filter_prompt(example):
    human_parts = example.split('Human:')
    pairs = []
    for i in range(1, len(human_parts)):
        human_dialogue = human_parts[i]
        assistant_parts = human_dialogue.split('Assistant:')
        human_text = assistant_parts[0].strip()
        assistant_text = ''
        if len(assistant_parts) > 1:
            assistant_text = assistant_parts[1].strip()
        pairs.append((human_text, assistant_text))
    return pairs

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return [json.loads(line) for line in file]
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return None  

def get_output_or_chosen(example):
    if 'output' in example:
        return example['output']
    elif 'chosen' in example:
        return example['chosen']
    else:
        raise ValueError('double check your data format')

def get_instruction_or_prompt(example):
    if 'instruction' in example:
        return example['instruction']
    elif 'prompt' in example:
        return example['prompt']
    else:
        raise ValueError('double check your data format')

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequences_should_be_stopped.append(True)
                    break
            sequences_should_be_stopped.append(False)
        return all(sequences_should_be_stopped)
    
@torch.no_grad()
def generate_completions(model, device, tokenizer, prompts, batch_size=1, stop_id_sequences=None, disable_tqdm=False, verbose=False, **generation_kwargs):
    generations = []
    if hasattr(model, "module"):
        print_rank_0(f'-----{model.module.generation_config}-----')
    else:
        print_rank_0(f'-----{model.generation_config}-----')

    if generation_kwargs:
        print_rank_0(f'-----{generation_kwargs}-----')
    
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding = 'longest', return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        batch_input_ids = batch_input_ids.to(device)
        attention_mask = attention_mask.to(device)

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
            batch_outputs = batch_outputs.detach().cpu()

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # in case piece id out of range
            #batch_outputs[batch_outputs >= tokenizer.vocab_size] = tokenizer.unk_token_id
            #batch_outputs[batch_outputs == -1] = tokenizer.unk_token_id
            
            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        if verbose:
            print("--------")
            print(batch_generations[0])
            
        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(batch_input_ids, attention_mask).logits[(torch.arange(batch_input_ids.size(0)), attention_mask.sum(dim=-1)-1)]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs
    

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


class ExponentialMovingAverage:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.ema = None

    def update(self, num):
        prev_ema = num if self.ema is None else self.ema
        self.ema = self.alpha * prev_ema + (1.0 - self.alpha) * num
        return self.ema

    def get(self):
        return self.ema if self.ema is not None else 0.


def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path.lower():
        #from transformers.models.llama import LlamaTokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)       # not adding start token 
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            #print("Adding pad token")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)      # not adding start token 
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})
    return tokenizer

def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        # remove lora, sau, dora weights
        if "lora" in key or "s2" in key or 'magnitude' in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
def load_state_dict_into_model(model_to_load=None,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight", "sau_weights", 'magnitude', 's4'],
    loraplus_lr_ratio=1.0,
):  
    # they suppose to be the same when lr_ratio is 1.0, however, slight discrepancy is observed, to be consistent with previous results
    # we keep the original implementation when lr_ratio is 1.0
    if loraplus_lr_ratio == 1.0:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list)
                        and p.requires_grad and not any(nd in n.lower()
                                                        for nd in lora_name_list))
                ],
                "weight_decay":
                weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list)
                        and p.requires_grad and any(nd in n.lower()
                                                    for nd in lora_name_list))
                ],
                "weight_decay":
                weight_decay,
                "lr":
                lora_lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n.lower()
                            for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay":
                0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list)
                        and p.requires_grad and not any(nd in n.lower()
                                                        for nd in lora_name_list))
                ],
                "weight_decay":
                weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list)
                        and p.requires_grad and any(nd in n.lower()
                                                    for nd in lora_name_list) and 'lora_left_weight' not in n.lower())
                ],
                "weight_decay":
                weight_decay,
                "lr":
                lora_lr
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list)
                        and p.requires_grad and any(nd in n.lower()
                                                    for nd in lora_name_list) and 'lora_left_weight' in n.lower())
                ],
                "weight_decay":
                weight_decay,
                "lr":
                lora_lr * loraplus_lr_ratio
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n.lower()
                            for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay":
                0.0,
            },
        ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k and 'sau' not in k and 'magnitude' not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def select_heads_based_on_grads(grads, n=108):
    # grads (module_name, layer_number, head_number): value
    # select the top n heads based on the absolute value of the gradients
    grads_list = [(key, value) for key, value in grads.items()]
    grads_list.sort(key=lambda x: torch.abs(x[1].mean()).item(), reverse=True)

    top_n_grads = grads_list[:n]
    top_n_heads = [item[0] for item in top_n_grads]

    return top_n_heads



def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps <= num_training_steps, "warmup + adjust_step is more than full training steps"
    assert adjust_step + first_warmup_steps <= restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        return float(current_step) / float(max(1, first_warmup_steps))

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps and current_step >= restart_every:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every + restart_warmup_steps - first_warmup_steps) /
            float(max(1, num_training_steps - first_warmup_steps))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - first_warmup_steps) / float(max(1, num_training_steps - first_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio=0.1,
    adjust_step=0,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def random_pruning_(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(random_pruning_mask)


def relora_optimizer_reset(
    param,
    prune_ratio = 0.99,
):
  pruning_fn = partial(random_pruning_, prune_ratio=prune_ratio)
  for optim_state_key in ['exp_avg', 'exp_avg_sq']:   # adamW
    state = safe_get_full_optimizer_state(param, optim_state_key)
    pruned_state = pruning_fn(state)
    safe_set_full_optimizer_state(param, optim_state_key, pruned_state)
   





