import torch
import torch.nn as nn
from torch.autograd import Function
import time

#######################################################################

class Linear4_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, backward_time, forward_time, energy_logger = args

        if energy_logger is not None:
            energy_logger.resume()
        
        start_f = time.time()

        # Infer output
        output = input@weight.t()
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        ctx.save_for_backward(input, weight, bias)
        end_f = time.time()
        forward_time.append(end_f-start_f)

        if energy_logger is not None:
            energy_logger.pause()
        
        ctx.energy_logger = energy_logger
        ctx.backward_time = backward_time

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        input, weight, bias = ctx.saved_tensors

    
        grad_input = grad_weight = grad_bias = None

        backward_time = ctx.backward_time

        energy_logger = ctx.energy_logger
        if energy_logger is not None:
            energy_logger.resume()

        start = time.time()

        if ctx.needs_input_grad[0]:
            grad_input = grad_output@weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('bhwc,bhwd->dc', input, grad_output)
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        end = time.time()
        backward_time.append(end - start)

        ctx.energy_logger = energy_logger
        if energy_logger is not None:
            energy_logger.pause()

        return grad_input, grad_weight, grad_bias, None, None, None
    
class Linear3_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, backward_time, forward_time, energy_logger = args

        if energy_logger is not None:
            energy_logger.resume()

        start_f = time.time()

        # Infer output
        output = input@weight.t()
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        ctx.save_for_backward(input, weight, bias)

        end_f = time.time()
        forward_time.append(end_f-start_f)

        ctx.energy_logger = energy_logger
        if energy_logger is not None:
            energy_logger.pause()

        ctx.backward_time = backward_time
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None

        backward_time = ctx.backward_time
        energy_logger = ctx.energy_logger
        if energy_logger is not None:
            energy_logger.resume()

        start = time.time()

        if ctx.needs_input_grad[0]:
            grad_input = grad_output@weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('bli,blo->oi', input, grad_output)
                            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        
        end = time.time()
        backward_time.append(end - start)
        if energy_logger is not None:
            energy_logger.pause()
        return grad_input, grad_weight, grad_bias, None, None, None
    
class Linear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            backward_time=None,
            forward_time=None,
            inference_time=None,
            energy_logger=None):
        super(Linear, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.backward_time = backward_time
        self.forward_time = forward_time
        self.inference_time = inference_time

        self.energy_logger = energy_logger


    def forward(self, input):
        if torch.is_grad_enabled(): # Training mode
            if input.dim() == 4:
                output = Linear4_op.apply(input, self.weight, self.bias, self.backward_time, self.forward_time, self.energy_logger)
            elif input.dim() == 3:
                output = Linear3_op.apply(input, self.weight, self.bias, self.backward_time, self.forward_time, self.energy_logger)
            else:
                raise ValueError("Chưa triển khai cho input có dim khác 3 hoặc 4")

        else: # Validation mode
            start = time.time()
            output = super().forward(input)
            end = time.time()
            self.inference_time.append(end-start)
        return output
    

def wrap_linear(linear, backward_time, forward_time, inference_time, energy_logger):
    has_bias = (linear.bias is not None)
    new_linear = Linear(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        backward_time = backward_time,
                        forward_time = forward_time,
                        inference_time = inference_time,
                        energy_logger = energy_logger
                        )
    

    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear