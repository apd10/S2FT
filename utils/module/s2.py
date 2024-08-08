import torch
import math
from torch import nn

from torch.nn.modules import Module
from torch import Tensor
from torch.nn import init

from torch.nn.parameter import Parameter

from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed

class S2Linear(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 start=None, end=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs),requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.start = start
        self.end = end

        self.s2 = nn.Parameter(torch.zeros(end-start, in_features), requires_grad=True)
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        self.weight.data[self.start:self.end, :] += self.s2
        del self.s2
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[self.start:self.end, :] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:
        base_output = torch.nn.functional.linear(input, self.weight, self.bias) 
        if self.fused:
            return base_output
        else:
            s2_output = torch.nn.functional.linear(input, self.s2, None)
            base_output[:, :, self.start:self.end] += s2_output
            return base_output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    

class S2Linear2(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 start=None, end=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs),requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs),requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.start = start
        self.end = end

        self.s2 = nn.Parameter(torch.zeros(out_features, end-start), requires_grad=True)
        self.weight.requires_grad = False
        self.fused = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def fuse_s2_weight(self):
        if self.fused == True:
            return
        self.weight.data[:, self.start:self.end] += self.s2
        del self.s2
        self.fused = True

    def unfuse_s2_weight(self):
        if self.fused == False:
            return
        self.weight[:, self.start:self.end] -= self.s2
        self.fused = False

    def forward(self, input: Tensor) -> Tensor:
        base_output = torch.nn.functional.linear(input, self.weight, self.bias) 
        if self.fused:
            return base_output
        else:
            s2_output = torch.nn.functional.linear(input[:, :, self.start:self.end], self.s2, None)
            base_output += s2_output
            return base_output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    

# convert the linear layer to S2
def convert_qkv_layer_to_s2(model,   
                               selected_parameters,
                               kv=False,
                               qo=False,
                               k=False,
                               o=False,):       # pass selected parameters here since each layer might have different head selections
    import copy
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for i in range(model.config.num_hidden_layers):
        ### Assume num_attention_heads = num_key_value_heads
        layer = model.model.layers[i]
        only_q = [j for j in selected_parameters['q_proj'][i] if j not in selected_parameters['v_proj'][i]]
        only_v = [j for j in selected_parameters['v_proj'][i] if j not in selected_parameters['q_proj'][i]]
        qv = [j for j in selected_parameters['q_proj'][i] if j in selected_parameters['v_proj'][i]]
        order = only_q + qv + only_v
        for j in range(model.config.num_attention_heads):
            if j not in order:
                order.append(j)
        if len(only_q) + len(qv) > 0:
            if not o:
                module = layer.self_attn.q_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.q_proj = S2Linear(in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias,
                            start=0,
                            end=(len(only_q) + len(qv))*head_dim,
                            device=next(module.parameters()).device,
                            dtype=next(module.parameters()).dtype)
                layer.self_attn.q_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
            if qo or o:
                module = layer.self_attn.o_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.o_proj = S2Linear2(in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias,
                            start=0,
                            end=(len(only_q) + len(qv))*head_dim,
                            device=next(module.parameters()).device,
                            dtype=next(module.parameters()).dtype)
                layer.self_attn.o_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
        if len(qv) + len(only_v) > 0:
            if not k:
                module = layer.self_attn.v_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.v_proj = S2Linear(in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias,
                            start=len(only_q)*head_dim,
                            end=(len(only_q) + len(qv) + len(only_v))*head_dim,
                            device=next(module.parameters()).device,
                            dtype=next(module.parameters()).dtype)
                layer.self_attn.v_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
            if kv or k:
                module = layer.self_attn.k_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.self_attn.k_proj = S2Linear(in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias,
                            start=len(only_q)*head_dim,
                            end=(len(only_q) + len(qv) + len(only_v))*head_dim,
                            device=next(module.parameters()).device,
                            dtype=next(module.parameters()).dtype)
                layer.self_attn.k_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
        q_weight = layer.self_attn.q_proj.weight.data
        q_weight = q_weight.reshape(model.config.num_attention_heads, -1, q_weight.shape[-1])
        layer.self_attn.q_proj.weight.data = q_weight[order, :, :].reshape(-1, q_weight.shape[-1])
        k_weight = layer.self_attn.k_proj.weight.data
        k_weight = k_weight.reshape(model.config.num_key_value_heads, -1, k_weight.shape[-1])
        layer.self_attn.k_proj.weight.data = k_weight[order, :, :].reshape(-1, k_weight.shape[-1])
        v_weight = layer.self_attn.v_proj.weight.data
        v_weight = v_weight.reshape(model.config.num_key_value_heads, -1, v_weight.shape[-1])
        layer.self_attn.v_proj.weight.data = v_weight[order, :, :].reshape(-1, v_weight.shape[-1])
        o_weight = layer.self_attn.o_proj.weight.data
        o_weight = o_weight.reshape(o_weight.shape[0], model.config.num_attention_heads, -1)
        layer.self_attn.o_proj.weight.data = o_weight[:, order, :].reshape(o_weight.shape[0], -1)

        del q_weight, k_weight, v_weight, o_weight
    return model

# convert the linear layer to S2
def convert_mlp_layer_to_s2(model,   
                            selected_parameters,
                            ug=False,
                            g=False):       # pass selected parameters here since each layer might have different head selections
    import copy
    for i in range(model.config.num_hidden_layers):
        ### Assume num_attention_heads = num_key_value_heads
        layer = model.model.layers[i]
        only_u = [j for j in selected_parameters['up_proj'][i] if j not in selected_parameters['down_proj'][i]]
        only_d = [j for j in selected_parameters['down_proj'][i] if j not in selected_parameters['up_proj'][i]]
        ud = [j for j in selected_parameters['up_proj'][i] if j in selected_parameters['down_proj'][i]]
        order = only_u + ud + only_d
        for j in range(model.config.intermediate_size):
            if j not in order:
                order.append(j)
        if len(only_u) + len(ud) > 0:
            if not g:
                module = layer.mlp.up_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.mlp.up_proj = S2Linear(in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias,
                            start=0,
                            end=(len(only_u) + len(ud)),
                            device=next(module.parameters()).device,
                            dtype=next(module.parameters()).dtype)
                layer.mlp.up_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
            if ug or g:
                module = layer.mlp.gate_proj
                checkpoint = copy.deepcopy(module.state_dict())
                layer.mlp.gate_proj = S2Linear(in_features=module.in_features,
                            out_features=module.out_features,
                            bias=module.bias,
                            start=0,
                            end=(len(only_u) + len(ud)),
                            device=next(module.parameters()).device,
                            dtype=next(module.parameters()).dtype)
                layer.mlp.gate_proj.load_state_dict(checkpoint, strict=False)
                del module
                del checkpoint
        if len(ud) + len(only_d) > 0:
            module = layer.mlp.down_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.down_proj = S2Linear2(in_features=module.in_features,
                           out_features=module.out_features,
                           bias=module.bias,
                           start=len(only_u),
                           end=(len(only_u) + len(ud) + len(only_d)),
                           device=next(module.parameters()).device,
                           dtype=next(module.parameters()).dtype)
            layer.mlp.down_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
        u_weight = layer.mlp.up_proj.weight.data
        layer.mlp.up_proj.weight.data = u_weight[order, :]
        g_weight = layer.mlp.gate_proj.weight.data
        layer.mlp.gate_proj.weight.data = g_weight[order, :]
        d_weight = layer.mlp.down_proj.weight.data
        layer.mlp.down_proj.weight.data = d_weight[:, order]

    return model


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]

# convert the SAU layer to linear layer
def convert_s2_to_linear_layer(model):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, S2Linear) or isinstance(module, S2Linear2):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        # fetch weight and bias once
        all_params_to_fetch = _z3_params_to_fetch([
            module.weight, 
            module.bias
        ])
        # fetch sau weights
        params_to_fetch = _z3_params_to_fetch([
            module.s2
        ])
        all_params_to_fetch.extend(params_to_fetch)
        with deepspeed.zero.GatheredParameters(all_params_to_fetch, modifier_rank=0, enabled=zero_stage_3):
            module.fuse_s2_weight()
    return model

def only_optimize_s2_parameters(model):
    # Turn off the gradient of all the parameters except the S2FT parameters
    for name, param in model.named_parameters():
        if "s2" in name:  
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model