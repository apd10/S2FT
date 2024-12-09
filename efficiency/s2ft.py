import torch
import math
import warnings
import copy

from typing import Callable, Iterable, Tuple
from torch import nn
from torch.optim import Optimizer
from torch.autograd.function import Function
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch import Tensor
from torch.nn import init


class S2ColumnLinearFunction(Function):
    @staticmethod
    def forward(input, weight, bias, start, end):
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, start, end = inputs
        input_save = input[:, :, start:end].clone()
        del input
        ctx.save_for_backward(input_save, weight, bias)
        ctx.start = start
        ctx.end = end

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        start = ctx.start
        end = ctx.end
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.transpose(-1, -2).matmul(input)

        ## Storing the partial gradient to grad2
        weight.grad2 = grad_weight.sum(0)
        weight.start = start
        weight.end = end

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, grad_bias, None, None
    

def s2columnlinear(input, weight, bias=None, start=None, end=None):
    return S2ColumnLinearFunction.apply(input, weight, bias, start, end)

class S2ColumnLinear(Module):

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

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return s2columnlinear(input, self.weight, self.bias, self.start, self.end)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class S2RowLinearFunction(Function):
    @staticmethod
    def forward(input, weight, bias, start, end):
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, start, end = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.start = start
        ctx.end = end

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        start = ctx.start
        end = ctx.end
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        grad_weight = grad_output[:, :, start:end].transpose(-1, -2).matmul(input)

        ## Storing the partial gradient to grad2
        weight.grad2 = grad_weight.sum(0)
        weight.start = start
        weight.end = end

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, None, grad_bias, None, None
    

def s2rowlinear(input, weight, bias=None, start=None, end=None):
    return S2RowLinearFunction.apply(input, weight, bias, start, end)

class S2RowLinear(Module):

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

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return s2rowlinear(input, self.weight, self.bias, self.start, self.end)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    

# convert the linear layers in the MHA module to S2
def convert_mha_layer_to_s2(model, selected_parameters):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        only_v = list(
            set(selected_parameters["v_proj"][i])
            - set(selected_parameters["o_proj"][i])
        )
        only_o = list(
            set(selected_parameters["o_proj"][i])
            - set(selected_parameters["v_proj"][i])
        )
        vo = list(
            set(selected_parameters["o_proj"][i])
            & set(selected_parameters["v_proj"][i])
        )
        order = only_v + vo + only_o
        for j in range(model.config.num_attention_heads):
            if j not in order:
                order.append(j)
        if len(only_v) + len(vo) > 0:
            module = layer.self_attn.v_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.v_proj = S2ColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_v) + len(vo)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.v_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
        if len(only_o) + len(vo) > 0:
            module = layer.self_attn.o_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.self_attn.o_proj = S2RowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=(len(only_v)) * head_dim,
                end=(len(only_v) + len(vo) + len(only_o)) * head_dim,
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.self_attn.o_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint

        q_weight = layer.self_attn.q_proj.weight.data
        q_weight = q_weight.reshape(
            model.config.num_key_value_heads, -1, q_weight.shape[-1]
        )
        layer.self_attn.q_proj.weight.data = q_weight[order, :, :].reshape(
            -1, q_weight.shape[-1]
        )
        k_weight = layer.self_attn.k_proj.weight.data
        k_weight = k_weight.reshape(
            model.config.num_key_value_heads, -1, k_weight.shape[-1]
        )
        layer.self_attn.k_proj.weight.data = k_weight[order, :, :].reshape(
            -1, k_weight.shape[-1]
        )
        v_weight = layer.self_attn.v_proj.weight.data
        v_weight = v_weight.reshape(
            model.config.num_key_value_heads, -1, v_weight.shape[-1]
        )
        layer.self_attn.v_proj.weight.data = v_weight[order, :, :].reshape(
            -1, v_weight.shape[-1]
        )
        o_weight = layer.self_attn.o_proj.weight.data
        o_weight = o_weight.reshape(
            o_weight.shape[0], model.config.num_attention_heads, -1
        )
        layer.self_attn.o_proj.weight.data = o_weight[:, order, :].reshape(
            o_weight.shape[0], -1
        )

        del v_weight, o_weight
    return model
    
# convert the linear layers in the FFN module to S2
def convert_ffn_layer_to_s2(model, selected_parameters):
    for i in range(model.config.num_hidden_layers):
        layer = model.model.layers[i]
        only_u = [
            j
            for j in selected_parameters["up_proj"][i]
            if j not in selected_parameters["down_proj"][i]
        ]
        only_d = [
            j
            for j in selected_parameters["down_proj"][i]
            if j not in selected_parameters["up_proj"][i]
        ]
        ud = [
            j
            for j in selected_parameters["up_proj"][i]
            if j in selected_parameters["down_proj"][i]
        ]
        order = only_u + ud + only_d
        for j in range(model.config.intermediate_size):
            if j not in order:
                order.append(j)
        if len(only_u) + len(ud) > 0:
            module = layer.mlp.up_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.up_proj = S2ColumnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=0,
                end=(len(only_u) + len(ud)),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.mlp.up_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint

        if len(ud) + len(only_d) > 0:
            module = layer.mlp.down_proj
            checkpoint = copy.deepcopy(module.state_dict())
            layer.mlp.down_proj = S2RowLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                start=len(only_u),
                end=(len(only_u) + len(ud) + len(only_d)),
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            layer.mlp.down_proj.load_state_dict(checkpoint, strict=False)
            del module
            del checkpoint
        u_weight = layer.mlp.up_proj.weight.data
        layer.mlp.up_proj.weight.data = u_weight[order, :]
        g_weight = layer.mlp.gate_proj.weight.data
        layer.mlp.gate_proj.weight.data = g_weight[order, :]
        d_weight = layer.mlp.down_proj.weight.data
        layer.mlp.down_proj.weight.data = d_weight[:, order]

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "grad2") and p.grad2 is not None:
                    grad = p.grad2
                else:
                    if p.grad is None:
                        continue
                    grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom

                if hasattr(p, "start") and p.start is not None:
                    if p.end - p.start == norm_grad.shape[0]:
                        p[p.start:p.end].add_(norm_grad, alpha=-step_size)
                    else:
                        p[:, p.start:p.end].add_(norm_grad, alpha=-step_size)
                else:
                    p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss