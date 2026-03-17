import torch
import torch.nn as nn
from torch.autograd import Function

from ..compression.rank.hosvd_power_iteration import hosvd_power_iteration, restore_hosvd_power_iteration
from ..compression.explain_var.hosvd_4_mode_var import hosvd_4_mode_var

class Linear_ASI4_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, S, U_list = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(S, U_list[0], U_list[1], U_list[2], U_list[3], weight, bias)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        S, U1, U2, U3, U4, weight, bias = ctx.saved_tensors
    
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            Z1 = torch.einsum("Ba,BHWD->aHWD", U1, grad_output) # Shape: (B, K1) and (B, H, W, D) -> (K1, H, W, D)
            Z2 = torch.einsum("Hb,abcd->aHcd", U2, S) # Shape: (H, K2) and (K1, K2, K3, K4) -> (K1, H, K3, K4)
            Z3 = torch.einsum("Wc,aHWD->aHcD", U3, Z1) # Shape: (W, K3) and (K1, H, W, D) -> (K1, H, K3, D)
            Z4 = torch.einsum("Cd,aHcd->aHCc", U4, Z2) # Shape: (C, K4) and (K1, H, K3, K4) -> (K1, H, C, K3)
            grad_weight = torch.einsum("aHcD,aHCc->DC", Z3, Z4) # Shape: (K1, H, K3, D) and (K1, H, C, K3) -> (D, C)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None
    
class Linear_ASI3_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, S, U_list = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(S, U_list[0], U_list[1], U_list[2], weight, bias)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        S, U1, U2, U3, weight, bias = ctx.saved_tensors
    
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            Z1 = torch.einsum('blo,bk->lok', grad_output, U1) # Shape: B, L, O and B, K1 -> L, O, K1
            Z2 = torch.einsum('abc,lb->acl', S, U2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
            Z3 = torch.einsum('acl,ic->ail', Z2, U3) # Shape: K1, K3, L and I, K3 -> K1, I, L
            grad_weight = torch.einsum('lok,kil->oi', Z1, Z3) # Shape: L, O, K1 and K1, I, L -> O, I

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None

class Linear_ASI(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            rank=1,
            truncation_threshold=None):
        super(Linear_ASI, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.rank = rank
        self.reuse_U = False
        self.u_list = None
        self.truncation_threshold=truncation_threshold

    def forward(self, input):
        if self.activate and torch.is_grad_enabled(): # Training mode
            if self.truncation_threshold is not None and self.u_list is None:
                S, self.u_list = hosvd_4_mode_var(input, self.truncation_threshold)
            else:
                S, self.u_list = hosvd_power_iteration(input, previous_Ulist=self.u_list, reuse_U=self.reuse_U, rank=self.rank)
            if self.reuse_U == False:
                self.reuse_U = True
            if input.dim() == 4:
                output = Linear_ASI4_op.apply(input, self.weight, self.bias, S, self.u_list)
            elif input.dim() == 3:
                output = Linear_ASI3_op.apply(input, self.weight, self.bias, S, self.u_list)
            else:
                raise ValueError("Not implemented for input with dim != 3 or 4")
            
        else: # activate is False or Validation mode
            output = super().forward(input)
        return output
    

def wrap_linearASI(linear, active, rank, truncation_threshold=None):
    has_bias = (linear.bias is not None)
    new_linear = Linear_ASI(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        rank=rank,
                        truncation_threshold=truncation_threshold
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear