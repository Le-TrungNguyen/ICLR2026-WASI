import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=16, use_bias=False):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.original_layer = linear_layer
        
        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)  # A does not have bias
        self.lora_B = nn.Linear(rank, self.out_features, bias=use_bias)  # B may have bias
        
        nn.init.normal_(self.lora_A.weight, std=1/rank)
        nn.init.zeros_(self.lora_B.weight)
        if use_bias and self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)
        
        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha / rank
        self.use_bias = use_bias
        
    def forward(self, x):
        # Original forward
        original_output = self.original_layer(x)
        
        # LoRA forward
        lora_output = self.lora_B(self.lora_A(x))
        
        # Scale and add
        return original_output + lora_output * self.scaling
    
def wrap_linearLora(linear, lora_alpha, rank):
    new_linear = LoRALinear(linear, rank=rank, alpha=lora_alpha, use_bias=(linear.bias is not None))
    return new_linear