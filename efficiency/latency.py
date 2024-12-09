import argparse
import torch
import copy
import random
import numpy as np

from transformers import AutoModelForCausalLM, SchedulerType, set_seed

import time
from typing import List

from s2ft import convert_mha_layer_to_s2, convert_ffn_layer_to_s2, AdamW


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

def stable_mean(arr: List[float]) -> float:
    if len(arr) < 4:
        return np.mean(arr)
    size = int(len(arr) * 0.25)
    return np.mean(sorted(arr)[size:-size])


def generate_data(data_num, context_length):
    input_ids = torch.randint(low=5000, high=20000, size=(data_num, context_length))
    input_position_ids = torch.arange(0, context_length, dtype=torch.long)
    input_position_ids = input_position_ids.unsqueeze(0).repeat(data_num, 1)
    labels = copy.deepcopy(input_ids)
    return input_ids, input_position_ids, labels
    

def get_trainable_parameters(model, args):
    for param in model.parameters():
        param.requires_grad = False

    parameters_v = {}
    parameters_o = {}
    mha_indices = [i for i in range(model.config.num_attention_heads * model.config.num_hidden_layers)]
    for i in range(model.config.num_hidden_layers):
        parameters_v[i] = []
        parameters_o[i] = []
    num_v = int(model.config.num_attention_heads * model.config.num_hidden_layers * args.v_ratio)
    num_o = int(model.config.num_attention_heads * model.config.num_hidden_layers * args.o_ratio)
    select_v = sorted(random.sample(mha_indices, num_v))
    for v in select_v:
        parameters_v[v // model.config.num_attention_heads].append(v % model.config.num_attention_heads)
    select_o = sorted(random.sample(mha_indices, num_o))
    for o in select_o:
        parameters_o[o // model.config.num_attention_heads].append(o % model.config.num_attention_heads)
    selected_parameters_mha = {'v_proj': parameters_v, 'o_proj': parameters_o}

    convert_mha_layer_to_s2(model, selected_parameters_mha)

    parameters_u = {}
    parameters_d = {}
    intermediate_dim = model.config.intermediate_size
    ffn_indices = [i for i in range(intermediate_dim * model.config.num_hidden_layers)]
    for i in range(model.config.num_hidden_layers):
        parameters_u[i] = []
        parameters_d[i] = []
    num_u = int(intermediate_dim * model.config.num_hidden_layers * args.u_ratio)
    num_d = int(intermediate_dim * model.config.num_hidden_layers * args.d_ratio)
    select_u = sorted(random.sample(ffn_indices, num_u))
    for u in select_u:
        parameters_u[u // intermediate_dim].append(u % intermediate_dim)
    select_d = sorted(random.sample(ffn_indices, num_d))
    for d in select_d:
        parameters_d[d // intermediate_dim].append(d % intermediate_dim)
    selected_parameters_ffn = {'up_proj': parameters_u, 'down_proj': parameters_d}

    convert_ffn_layer_to_s2(model, selected_parameters_ffn)

    num_parameters = sum(p.numel() for p in model.parameters())
    num_trainable_parameters = (num_v + num_o) * model.config.hidden_size // model.config.num_attention_heads * model.config.hidden_size + (num_u + num_d) * model.config.hidden_size
    print(
        f"Number of parameters: {num_parameters}, Number of trainable parameters: {num_trainable_parameters}, Ratio: {num_trainable_parameters/num_parameters:.4f}"
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--data_num", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=100,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument('--v_ratio',
                        type=float,
                        default=0.0,
                        help="The ratio of trainable parameters in each Value Projection.")
    parser.add_argument('--o_ratio',
                        type=float,
                        default=0.0,
                        help="The ratio of trainable parameters in each Output Projection.")
    parser.add_argument('--u_ratio',
                        type=float,
                        default=0.0,
                        help="The ratio of trainable parameters in each Up Projection.")
    parser.add_argument('--d_ratio',
                        type=float,
                        default=0.0,
                        help="The ratio of trainable parameters in each Down Projection.")

    args = parser.parse_args()
    set_random_seed(0)

    input_ids, input_position_ids, labels = generate_data(
        args.data_num, args.context_length
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).cuda()
    model.train()

    get_trainable_parameters(model, args)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    times = []
    for round in range(50):
        for _ in range(200):
            optimizer.zero_grad()
            start_time = cuda_time()
            input_ids = input_ids.cuda()
            input_position_ids = input_position_ids.cuda()
            labels = labels.cuda()
            loss = model(
                input_ids=input_ids, position_ids=input_position_ids, labels=labels
            )[0]
            loss.backward()
            optimizer.step()
            end_time = cuda_time()
        if round >= 10:
            times.append((end_time - start_time))
        if times:
            print(round, stable_mean(times) * 1000)


if __name__ == "__main__":
    main()
