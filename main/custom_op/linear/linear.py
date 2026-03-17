import torch
import torch.nn as nn
from torch.autograd import Function

#######################################################################

class Linear_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias = args

        print("hehe")
        

        # Infer output
        output = input@weight.t()
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        ctx.save_for_backward(input, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        input, weight, bias = ctx.saved_tensors
    
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output@weight
        if ctx.needs_input_grad[1]:
            if input.dim() == 4:
                grad_weight = torch.einsum('bhwc,bhwd->dc', input, grad_output)
            elif input.dim() == 3:
                grad_weight = torch.einsum('bli,blo->oi', input, grad_output)
            else:
                print("Chưa triển khai cho input có dim khác 3 hoặc 4")

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
 
    
class Linear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None):
        super(Linear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(self, input):
        if torch.is_grad_enabled(): # Training mode
            output = Linear_op.apply(input, self.weight, self.bias)
        else: # Validation mode
            output = super().forward(input)
        return output
    

def wrap_linear(linear):
    has_bias = (linear.bias is not None)
    new_linear = Linear(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias
                        )
    

    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear