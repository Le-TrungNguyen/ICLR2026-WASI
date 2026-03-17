import torch.nn as nn
import time
import torch
from torch.autograd import Function

class LoRA_op(Function):
    @staticmethod
    def forward(ctx, *args):
        x, original_layer, lora_A, lora_B, scaling, backward_time, forward_time = args
        
        start_f = time.time()
        
        # Original forward
        original_output = x@original_layer.t()
        
        # LoRA forward
        lora_a_output = x@lora_A.t()
        lora_output = lora_a_output@lora_B.t()
        
        # Scale and add
        output = original_output + lora_output * scaling

        end_f = time.time()
        forward_time.append(end_f - start_f)
        
        # Lưu context cho backward pass
        ctx.save_for_backward(x, original_layer, lora_A, lora_B, lora_a_output)
        ctx.scaling = scaling
        ctx.backward_time = backward_time
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, original_layer, lora_A, lora_B, lora_a_output = ctx.saved_tensors
        scaling = ctx.scaling
        backward_time = ctx.backward_time
        
        start_b = time.time()
        
        grad_lora_output = grad_output * scaling

        grad_lora_a_output = grad_lora_output @ lora_B # = grad_lora_b_input
        grad_lora_a_input = grad_lora_a_output @ lora_A


        grad_lora_B = torch.einsum('bi,bo->oi', lora_a_output, grad_lora_output) if lora_a_output.dim() == 2 else \
                     torch.einsum('bli,blo->oi', lora_a_output, grad_lora_output) if lora_a_output.dim() == 3 else \
                     torch.einsum('bhwc,bhwd->dc', lora_a_output, grad_lora_output)

        grad_lora_A = torch.einsum('bi,bo->oi', x, grad_lora_a_output) if x.dim() == 2 else \
                     torch.einsum('bli,blo->oi', x, grad_lora_a_output) if x.dim() == 3 else \
                     torch.einsum('bhwc,bhwd->dc', x, grad_lora_a_output)
        
        
        
        # grad_lora_B_bias = None
        # if lora_B.bias is not None:
        #     grad_lora_B_bias = grad_lora_output.sum(0)
        
        end_b = time.time()
        backward_time.append(end_b - start_b)
        
        # Return None cho các tham số không cần gradient
        return grad_lora_a_input, None, grad_lora_A, grad_lora_B, None, None, None

class LoRA_inference_op(Function):
    @staticmethod
    def forward(ctx, *args):
        x, original_layer, lora_A, lora_B, scaling, inference_time = args
        
        start_f = time.time()
        
        # Original forward
        original_output = x@original_layer.t()
        
        # LoRA forward
        lora_a_output = x@lora_A.t()
        lora_output = lora_a_output@lora_B.t()
        
        # Scale and add
        output = original_output + lora_output * scaling

        end_f = time.time()
        inference_time.append(end_f - start_f)
        
        
        return output
    @staticmethod
    def backward(ctx, grad_output):
        pass
    
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=16, use_bias=False, backward_time=None, forward_time=None, inference_time=None):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.original_layer = linear_layer
        self.backward_time = backward_time if backward_time is not None else []
        self.forward_time = forward_time if forward_time is not None else []
        self.inference_time = inference_time if inference_time is not None else []
        
        # Đảm bảo không huấn luyện tham số của lớp gốc
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA A và B ma trận
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)  # A không có bias
        self.lora_B = nn.Linear(rank, self.out_features, bias=use_bias)  # B có thể có bias
        
        # Khởi tạo
        nn.init.normal_(self.lora_A.weight, std=1/rank)
        nn.init.zeros_(self.lora_B.weight)
        if use_bias and self.lora_B.bias is not None:
            nn.init.zeros_(self.lora_B.bias)  # Khởi tạo bias bằng 0
            
        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha / rank
        self.use_bias = use_bias
        
    def forward(self, x):
        if torch.is_grad_enabled():  # Training mode
            # Sử dụng custom op để đo thời gian
            return LoRA_op.apply(x, self.original_layer.weight, self.lora_A.weight, self.lora_B.weight, self.scaling, self.backward_time, self.forward_time)
        else:  # Inference mode
            return LoRA_inference_op.apply(x, self.original_layer.weight, self.lora_A.weight, self.lora_B.weight, self.scaling, self.inference_time)


def wrap_linearLora(linear, lora_alpha, rank, backward_time, forward_time, inference_time):
    new_linear = LoRALinear(linear, rank=rank, alpha=lora_alpha, use_bias=(linear.bias is not None),
                        backward_time = backward_time,
                        forward_time = forward_time,
                        inference_time = inference_time)
    return new_linear