import math
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr

class LoRALinear(nn.Module):
    # an simple implementation of LoRA that supports Linear Layer
    def __init__(self, weight, lora_dim=0, lora_scaling=1, lora_dropout=0, bias=None):
        super(LoRALinear, self).__init__()
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

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

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
        self.weight.data += self.lora_scaling * torch.matmul(self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora == False:
            return
        self.weight.data -= self.lora_scaling * torch.matmul(self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling

def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

# convert the linear layer to LoRA layer
def convert_linear_layer_to_lora(
    model, part_module_name, lora_dim=0, lora_scaling=1, lora_dropout=0
):
    replace_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(part in name for part in part_module_name):
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        layer = LoRALinear(module.weight, lora_dim, lora_scaling, lora_dropout, module.bias).to(module.weight.device).to(module.weight.dtype)
        recursive_setattr(model, name, layer)
    return model

# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.fuse_lora_weight()
    return model
