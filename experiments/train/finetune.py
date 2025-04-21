import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import torch
import json
import random
import math
import time
import argparse
import deepspeed

# add this if enconter AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, SchedulerType, get_scheduler
import composable_ai.extension_layers as DEXT
import module.lora as LORA
import module.dora as DORA

from utils.s2_utils import (
    convert_ffn_layer_to_s2,
    convert_mha_layer_to_s2,
    convert_s2_to_linear_layer,
    only_optimize_s2_parameters,
)

from utils.utils import (
    print_rank_0,
    to_device,
    set_random_seed,
    get_all_reduce_mean,
    int_or_float,
)

from utils.ds_utils import get_train_ds_config

from utils.model_utils import (
    load_hf_tokenizer,
    create_hf_model,
    save_hf_format,
    get_optimizer_grouped_parameters,
    make_model_gradient_checkpointing_compatible,
    print_throughput
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="S2FT Training")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
    parser.add_argument(
        "--load_last_model", action="store_true", help="only save the last model"
    )
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
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
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
        default="cosine",
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
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Training data type",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout rate of the model."
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    # S2FT
    parser.add_argument(
        "--s2", action="store_true", help="use S2FT for efficient training."
    )
    parser.add_argument("--v_ratio", type=float, default=0.0)
    parser.add_argument("--o_ratio", type=float, default=0.0)
    parser.add_argument("--u_ratio", type=float, default=0.0)
    parser.add_argument("--d_ratio", type=float, default=0.0)

    # DEXT
    parser.add_argument(
        "--dext", action="store_true", help="use LoRA for efficient training."
    )
    
    parser.add_argument("--dext_config_file",
                        type=str,
                        default=None,
                        help="Extension layer config")

    # LoRA
    parser.add_argument(
        "--lora", action="store_true", help="use LoRA for efficient training."
    )
    parser.add_argument(
        "--dora", action="store_true", help="use DoRA for efficient training."
    )

    parser.add_argument("--lora_config_file",
                        type=str,
                        default=None,
                        help="Extension layer config")


    parser.add_argument("--dora_config_file",
                        type=str,
                        default=None,
                        help="Extension layer config")

    # parser.add_argument("--lora_dim",
    #                     type=int,
    #                     default=0,
    #                     help="If > 0, use LoRA for efficient training.")
    # parser.add_argument("--lora_dropout",
    #                     type=float,
    #                     default=0.0,
    #                     help="LoRA dropout rate.")
    # parser.add_argument("--lora_alpha",
    #                     type=float,
    #                     default=16.0,
    #                     help="lora scaling factor.")
    # parser.add_argument("--lora_module_name",
    #                     type=str,
    #                     nargs='+',
    #                     default=['.layers'],
    #                     help="The scope of LoRA.")

    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="tensorboard")
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="save deepspeed engine checkpoint for recover the training",
    )
    parser.add_argument(
            "--early_stop_window",
            type=float,
            default=-1,
            help="early_stop_window"
            )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def save_model(args, model):
    if args.output_dir is not None:
        if args.s2:
            print_rank_0("converting s2 to linear layer ...", args.global_rank)
            model = convert_s2_to_linear_layer(model)
            # TODO
        if args.global_rank == 0:
            print_rank_0("saving the model ...", args.global_rank)
            if args.dext:
                os.makedirs(args.output_dir, exist_ok=True)
                DEXT.save_adapter_model(model, args.output_dir)
            elif args.lora:
                os.makedirs(args.output_dir, exist_ok=True)
                LORA.save_adapter_model(model, args.output_dir)
            elif args.dora:
                os.makedirs(args.output_dir, exist_ok=True)
                DORA.save_adapter_model(model, args.output_dir)

            else:    
                save_hf_format(model, tokenizer, args)

def main():
    args = parse_args()

    ## init distributed training with deepspeed
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="S2FT Training",
        profiler_path=args.output_dir,
    )

    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    set_random_seed(args.seed)
    torch.distributed.barrier()

    ## Load model and tokenizer
    tokenizer = load_hf_tokenizer(
        args.model_name_or_path, fast_tokenizer=True
    )  
    tokenizer.model_max_length = args.max_seq_len
    print_rank_0(f"Tokenizer: {tokenizer.model_max_length}", args.global_rank)

    print_rank_0(f"Loading model from {args.model_name_or_path}", args.global_rank)

    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        dropout=args.dropout,
    )

    ## Enable S2FT for Fine-tuning
    if args.dext:
        model.requires_grad_(False) # set core model parameters to frozen state.
        model = DEXT.create_new_model_from_config_file(model, args.dext_config_file)
        print_rank_0("--- use DEXT -----")
        model = make_model_gradient_checkpointing_compatible(model)
    ## Enable LoRA for Fine-tuning
    elif args.lora:
        model.requires_grad_(False)
        model = LORA.create_new_model_from_config_file(model, args.lora_config_file)
        print_rank_0("------use LoRA------", args.global_rank)
        model = make_model_gradient_checkpointing_compatible(model)
    ## Enable DoRA for Fine-tuning
    elif args.dora:
        model.requires_grad_(False)
        model = DORA.create_new_model_from_config_file(model, args.dora_config_file)
        print_rank_0("------use DoRA------", args.global_rank)
        model = make_model_gradient_checkpointing_compatible(model)
    elif args.s2:
        print_rank_0("------use S2FT------", args.global_rank)
        if args.v_ratio > 0 or args.o_ratio > 0:
            parameters_v = {}
            parameters_o = {}
            mha_indices = [
                i
                for i in range(
                    model.config.num_attention_heads * model.config.num_hidden_layers
                )
            ]
            for i in range(model.config.num_hidden_layers):
                parameters_v[i] = []
                parameters_o[i] = []
            num_v = int(
                model.config.num_attention_heads
                * model.config.num_hidden_layers
                * args.v_ratio
            )
            num_o = int(
                model.config.num_attention_heads
                * model.config.num_hidden_layers
                * args.o_ratio
            )
            select_v = sorted(random.sample(mha_indices, num_v))
            for v in select_v:
                parameters_v[v // model.config.num_attention_heads].append(
                    v % model.config.num_attention_heads
                )
            select_o = sorted(random.sample(mha_indices, num_o))
            for o in select_o:
                parameters_o[o // model.config.num_attention_heads].append(
                    o % model.config.num_attention_heads
                )
            selected_parameters_mha = {"v_proj": parameters_v, "o_proj": parameters_o}

            convert_mha_layer_to_s2(model, selected_parameters_mha)

        if args.u_ratio > 0 or args.d_ratio > 0:
            parameters_u = {}
            parameters_d = {}
            intermediate_dim = model.config.intermediate_size
            ffn_indices = [
                i for i in range(intermediate_dim * model.config.num_hidden_layers)
            ]
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
            selected_parameters_ffn = {"up_proj": parameters_u, "down_proj": parameters_d}

            convert_ffn_layer_to_s2(model, selected_parameters_ffn)

        model = only_optimize_s2_parameters(model)
        model = make_model_gradient_checkpointing_compatible(model)

        print_rank_0(f"learning rate: {args.learning_rate}", args.global_rank)




    print(model)
    ## Load Data
    if len(args.data_path) == 1 and ".json" in args.data_path[0]:
        print_rank_0(f"------json Data: {args.data_path[0]}", args.global_rank)
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        if args.val_set_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
            print_rank_0(
                f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}",
                args.global_rank,
            )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        raise ValueError(
            "Only json format is supported for now. Please check your data format."
        )

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
        model, args.weight_decay, args.learning_rate
    )

    ## Init deepspeed optimizer
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
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
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ## Training
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

    print_rank_0("***** Running training *****", args.global_rank)
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    print_rank_0(formatted_args, args.global_rank)

    # print trainable parameters
    num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print_rank_0(f"Number of trainable parameters: {num}", args.global_rank)

    total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    current_step_count = 0

    lr_plot = []
    best_val_loss = float("inf")
    final_saved_model_index = 0
    best_model = None

    args.eval_step = args.eval_step * args.gradient_accumulation_steps

    args.eval_delay = (
        args.eval_delay
        if isinstance(args.eval_delay, int)
        else int(args.eval_delay * total_training_steps)
    )

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()
        mean_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            lr_plot.append(
                lr_scheduler.get_last_lr()[1]
                if len(lr_scheduler.get_last_lr()) > 1
                else lr_scheduler.get_last_lr()[0]
            )
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
                print_throughput(model.model, args, end - start, args.global_rank)
            mean_loss += loss.item()
            if current_step_count % 50 == 0:
                print_rank_0(
                    f"Mean Loss: {mean_loss/(step+1)}",
                    args.global_rank,
                )

            if (
                current_step_count % args.eval_step == 0
                and args.val_set_size > 0
                and not args.load_last_model
                and current_step_count >= args.eval_delay
            ):
                ppl, val_loss = evaluation(model, val_dataloader)
                print_rank_0(
                    f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                    args.global_rank,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.global_rank == 0:
                        best_model = copy.deepcopy(model.module).to("cpu")
                        print("Saving model at step", current_step_count)
                        save_model(args, best_model)
                    final_saved_model_index = current_step_count

            if args.early_stop_window > 0 and (current_step_count - final_saved_model_index) > args.early_stop_window * total_training_steps:
                print_rank_0("No updates since 1/4th Epoch -- breaking")
                break

        if args.early_stop_window > 0 and (current_step_count - final_saved_model_index) > args.early_stop_window * total_training_steps:
            break
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} Train loss: {mean_loss/len(train_dataloader)}",
            args.global_rank,
        )

        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        # evaluate last model
        if args.val_set_size > 0 and not args.load_last_model:
            ppl, val_loss = evaluation(model, val_dataloader)
            print_rank_0(
                f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                args.global_rank,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.global_rank == 0:
                    best_model = copy.deepcopy(model.module).to("cpu")
                final_saved_model_index = "last"
            print_rank_0(f"Best validation loss: {best_val_loss}", args.global_rank)
            print_rank_0(
                f"Savings the best model at step {final_saved_model_index}",
                args.global_rank,
            )

        if args.load_last_model:
            print_rank_0("only load the last model ...", args.global_rank)

        model = best_model.to(device) if best_model else model
        save_model(args, model)
        torch.save(lr_plot, os.path.join(args.output_dir, "lr_plot.pt"))


if __name__ == "__main__":
    main()
