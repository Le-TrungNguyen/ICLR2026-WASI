import torch.nn as nn
import torch
from custom_op.linear.linear_lora import LoRALinear
from custom_op.linear.linear_WASI import Linear_WASI
from custom_op.linear.linear_WSI import Linear_WSI

def get_all_layer_with_name(model):
    layers = {}
    visited = set()

    for name, mod in model.named_modules():

        if model.model_type == "transformer":
            if any(name.startswith(v + ".") for v in visited):
                continue
            
            if not model.count_attention:
                if isinstance(mod, LoRALinear) and 'mlp' in name:
                    layers[name] = mod
                    visited.add(name)
                elif (isinstance(mod, nn.modules.linear.Linear) or isinstance(mod, Linear_WASI) or isinstance(mod, Linear_WSI)) and 'mlp' in name:
                    layers[name] = mod
            elif model.count_attention:
                if isinstance(mod, LoRALinear) and ('mlp' in name or 'self_attention' in name):
                    layers[name] = mod
                    visited.add(name)
                elif (isinstance(mod, nn.modules.linear.Linear) or isinstance(mod, Linear_WASI) or isinstance(mod, Linear_WSI)) and ('mlp' in name or 'self_attention' in name):
                    layers[name] = mod

        elif model.model_type == "cnn":
            if isinstance(mod, nn.modules.conv.Conv2d): 
                layers[name] = mod

    return layers


def get_all_attn_with_name(model):
    attn_layers = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            attn_layers[name] = mod
    return attn_layers



def get_active_layer_with_name(model):
    total_layer = get_all_layer_with_name(model)

    if model.num_of_finetune == "all" or model.num_of_finetune > len(total_layer):
        return total_layer
    elif model.num_of_finetune == None or model.num_of_finetune == 0:
        return -1
    else:
        active_layers = dict(list(total_layer.items())[-model.num_of_finetune:])
        return active_layers
        

class Hook:
    def __init__(self, module):
        self.module = module
        self.input_size = None
        self.output_size = None
        self.inputs = []
        self.outputs = []

        self.weight_size = None
        self.weight = None

        self.is_lora = False
        self.original_weight = None
        self.lora_A_weight = None
        self.lora_B_weight = None
        self.lora_rank = None
        self.lora_alpha = None
        self.lora_scaling = None
        self.effective_weight = None 

        # ---- WASI info ----
        self.is_wasi = False
        self.L_weight = None
        self.R_weight = None
        self.L_size = None
        self.R_size = None
        self.reconstructed_weight = None
        self.weight_rank = None
        
        self.active = True
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if not self.active:
            return

        # --- handle input ---
        if isinstance(input, tuple):
            if len(input) == 1 and torch.is_tensor(input[0]):
                Input = input[0].clone().detach()
            elif len(input) > 0 and torch.is_tensor(input[0]):
                # attention: query, key, value
                Input = input[0].clone().detach()  # lấy query làm đại diện
            elif len(input) > 0 and isinstance(input[0], tuple) and torch.is_tensor(input[0][0]):
                # input = ((query, key, value),)
                Input = input[0][0].clone().detach()
            else:
                Input = None
        elif torch.is_tensor(input):
            Input = input.clone().detach()
        else:
            Input = None
            
        # --- handle output ---
        if isinstance(output, tuple):
            if len(output) > 0 and torch.is_tensor(output[0]):
                Output = output[0].clone().detach()  
            else:
                Output = None
        elif torch.is_tensor(output):
            Output = output.clone().detach()
        else:
            Output = None

        if Input is not None:
            self.input_size = torch.tensor(Input.shape)
            self.inputs.append(Input)
        if Output is not None:
            self.output_size = torch.tensor(Output.shape)
            self.outputs.append(Output)

        # =========================================================
        # 1) LoRA branch
        # =========================================================
        if hasattr(module, 'original_layer') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            self.is_lora = True
            
            if hasattr(module.original_layer, 'weight') and module.original_layer.weight is not None:
                self.original_weight = module.original_layer.weight.clone().detach()
            
            if hasattr(module.lora_A, 'weight') and module.lora_A.weight is not None:
                self.lora_A_weight = module.lora_A.weight.clone().detach()
                
            if hasattr(module.lora_B, 'weight') and module.lora_B.weight is not None:
                self.lora_B_weight = module.lora_B.weight.clone().detach()
            
            self.lora_rank = module.rank if hasattr(module, 'rank') else None
            self.lora_scaling = module.scaling if hasattr(module, 'scaling') else None
            self.lora_alpha = module.alpha if hasattr(module, 'alpha') else None
            
            if self.original_weight is not None and self.lora_A_weight is not None and self.lora_B_weight is not None:
                if len(self.lora_A_weight.shape) == 2 and len(self.lora_B_weight.shape) == 2:
                    lora_weight = self.lora_B_weight @ self.lora_A_weight
                    self.effective_weight = self.original_weight + lora_weight * self.lora_scaling
                    
            self.weight = self.original_weight
            self.weight_size = self.original_weight.shape if self.original_weight is not None else None

        # =========================================================
        # 2) WASI branch
        # =========================================================
        elif hasattr(module, 'L') and hasattr(module, 'R'):
            self.is_wasi = True

            self.L_weight = None if module.L is None else module.L.detach().clone()
            self.R_weight = None if module.R is None else module.R.detach().clone()

            self.L_size = None if self.L_weight is None else self.L_weight.shape
            self.R_size = None if self.R_weight is None else self.R_weight.shape

            self.weight_rank = getattr(module, 'weight_rank', None)

            if self.L_weight is not None and self.R_weight is not None:
                self.reconstructed_weight = self.L_weight @ self.R_weight
                self.weight = self.reconstructed_weight
                self.weight_size = self.reconstructed_weight.shape
            else:
                self.reconstructed_weight = None
                self.weight = None
                self.weight_size = None

        # =========================================================
        # 3) Generic module branch
        # =========================================================
        else:
            if hasattr(module, 'weight') and module.weight is not None:
                self.weight_size = module.weight.shape
                self.weight = module.weight.clone().detach()

    def get_lora_details(self):
        if not self.is_lora:
            return "Not LoRA"
        
        details = {
            "original_weight_shape": self.original_weight.shape if self.original_weight is not None else None,
            "lora_A_weight_shape": self.lora_A_weight.shape if self.lora_A_weight is not None else None,
            "lora_B_weight_shape": self.lora_B_weight.shape if self.lora_B_weight is not None else None,
            "lora_rank": self.lora_rank,
            "lora_scaling": self.lora_scaling,
            "lora_alpha": self.lora_alpha
        }

        return details
    
    def get_wasi_details(self):
        if not self.is_wasi:
            return "Not WASI"

        return {
            "L_shape": None if self.L_weight is None else tuple(self.L_weight.shape),
            "R_shape": None if self.R_weight is None else tuple(self.R_weight.shape),
            "reconstructed_weight_shape": None if self.reconstructed_weight is None else tuple(self.reconstructed_weight.shape),
            "weight_rank": self.weight_rank,
        }

    def activate(self, active):
        self.active = active

    def remove(self):
        self.input_size = None
        self.output_size = None
        self.inputs.clear()
        self.outputs.clear()
        self.weight_size = None
        self.weight =  None

        self.original_weight = None
        self.lora_A_weight = None
        self.lora_B_weight = None
        self.lora_rank = None
        self.lora_scaling = None
        self.effective_weight = None

        self.is_wasi = False
        self.L_weight = None
        self.R_weight = None
        self.L_size = None
        self.R_size = None
        self.reconstructed_weight = None
        self.weight_rank = None
    
        self.active = False
        self.hook.remove()

def calculate_flops_subspace_iteration(size_1, size_2, rank):
    if isinstance(size_1, torch.Tensor):
        m = torch.max(size_1, rank)
        n = torch.min(size_1, rank)
    else:
        m = max(size_1, rank)
        n = min(size_1, rank)
    return size_1 * rank * (2*size_2 - 1) + size_2 * rank * (2*size_1 - 1) + 2*m*n**2

def calculate_flops_SVD(size_1, size_2):
    if isinstance(size_1, torch.Tensor):
        m = torch.max(size_1, size_2)
        n = torch.min(size_1, size_2)
    else:
        m = max(size_1, size_2)
        n = min(size_1, size_2)
    return 4*m*n**2 + 8*n**3
