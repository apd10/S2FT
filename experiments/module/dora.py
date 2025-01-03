import math
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr

class DoRALinear(nn.Module):
    # an simple implementation of DoRA that supports Linear Layer
    def __init__(self, weight, lora_dim=0, lora_scaling=1, lora_dropout=0, bias=None):
        super(DoRALinear, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(columns, lora_dim))
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))

        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        lora_weight = torch.matmul(
            self.lora_left_weight.t(), self.lora_right_weight.t())
        weight_norm = self._get_weight_norm(weight, lora_weight, self.lora_scaling)
        # magnitude column-wise across output dimension
        self.magnitude = nn.Parameter(weight_norm, requires_grad=True)

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False
    
    def _get_weight_norm(self, weight, lora_weight, scaling):
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))            # lora_A
        nn.init.zeros_(self.lora_left_weight)                                       # lora_B   

    def fuse_lora_weight(self):
        if self.fuse_lora == True:
            return
        lora_weight = torch.matmul(
            self.lora_left_weight.t(), self.lora_right_weight.t()) 
        weight_norm = self._get_weight_norm(self.weight, lora_weight, self.lora_scaling).detach()
        mag_norm_scale = (self.magnitude / weight_norm).view(-1, 1)
        self.weight.data = mag_norm_scale * (self.weight.data + lora_weight * self.lora_scaling)
        self.fuse_lora = True
    
    def _apply_dora(self, x, lora_right_weight, lora_left_weight, scaling):
        lora_weight = torch.matmul(lora_left_weight.t(), lora_right_weight.t())
        weight_norm = self._get_weight_norm(self.weight, lora_weight, scaling).detach()
        mag_norm_scale = (self.magnitude / weight_norm).view(1, -1)
        res = (mag_norm_scale - 1) * F.linear(x, self.weight, self.bias) + mag_norm_scale * (x @ lora_right_weight @ lora_left_weight) * scaling
        return res

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            base_result = F.linear(input, self.weight, self.bias)
            input = self.lora_dropout(input)
            return base_result + self._apply_dora(input, self.lora_right_weight, self.lora_left_weight, self.lora_scaling)

def only_optimize_dora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

# convert the linear layer to LoRA layer
def convert_linear_layer_to_dora(
    model, part_module_name, lora_dim=0, lora_scaling=1, lora_dropout=0
):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(part in name for part in part_module_name):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        layer = DoRALinear(module.weight, lora_dim, lora_scaling, lora_dropout, module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, layer)
    return model

# convert the LoRA layer to linear layer
def convert_dora_to_linear_layer(model):
    for module in model.modules():
        if isinstance(module, DoRALinear):
            module.fuse_lora_weight()
    return model
