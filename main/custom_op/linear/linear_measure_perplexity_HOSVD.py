import torch
import torch.nn as nn
from torch.autograd import Function

from ..compression.explain_var.hosvd_4_mode_var import hosvd_4_mode_var

class Linear_measure_perplexity_HOSVD_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, explain_variance_threshold, perplexity, measured_rank_hosvd, layer_mem, layer_idx = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        S, u_list, rank_list = hosvd_4_mode_var(input, var=explain_variance_threshold, return_rank=True)

        layer_mem[layer_idx] = ((S.numel() + sum(u.numel() for u in u_list)) * 4 / (1024 * 1024))  # MB


        measured_rank_hosvd[layer_idx] = rank_list

        ctx.save_for_backward(input, S, weight, bias)
        ctx.u_list = u_list
        ctx.perplexity = perplexity
        ctx.layer_idx = layer_idx

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        input, S, weight, bias = ctx.saved_tensors

        if input.dim() == 4:
            U1, U2, U3, U4 = ctx.u_list
        elif input.dim() == 3:
            U1, U2, U3 = ctx.u_list

        perplexity = ctx.perplexity
        layer_idx = ctx.layer_idx
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if input.dim() == 4:
                grad_weight = torch.einsum('bhwc,bhwd->dc', input, grad_output)
                Z1 = torch.einsum("Ba,BHWD->aHWD", U1, grad_output) # Shape: (B, K1) and (B, H, W, D) -> (K1, H, W, D)
                Z2 = torch.einsum("Hb,abcd->aHcd", U2, S) # Shape: (H, K2) and (K1, K2, K3, K4) -> (K1, H, K3, K4)
                Z3 = torch.einsum("Wc,aHWD->aHcD", U3, Z1) # Shape: (W, K3) and (K1, H, W, D) -> (K1, H, K3, D)
                Z4 = torch.einsum("Cd,aHcd->aHCc", U4, Z2) # Shape: (C, K4) and (K1, H, K3, K4) -> (K1, H, C, K3)
                grad_weight_low_rank = torch.einsum("aHcD,aHCc->DC", Z3, Z4) # Shape: (K1, H, K3, D) and (K1, H, C, K3) -> (D, C)

            elif input.dim() == 3:
                grad_weight = torch.einsum('bli,blo->oi', input, grad_output)
                Z1 = torch.einsum('blo,bk->lok', grad_output, U1) # Shape: B, L, O and B, K1 -> L, O, K1
                Z2 = torch.einsum('abc,lb->acl', S, U2) # Shape: K1, K2, K3 and L, K2 -> K1, K3, L
                Z3 = torch.einsum('acl,ic->ail', Z2, U3) # Shape: K1, K3, L and I, K3 -> K1, I, L
                grad_weight_low_rank = torch.einsum('lok,kil->oi', Z1, Z3) # Shape: L, O, K1 and K1, I, L -> O, I

            perplexity[layer_idx] = torch.norm(grad_weight_low_rank - grad_weight)


        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, None, None, None, None, None, None, None

class Linear_measure_perplexity_HOSVD(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            explain_variance_threshold=None,
            perplexity=None,
            measured_rank_svd=None,
            layer_mem=None,
            layer_idx=None):
        super(Linear_measure_perplexity_HOSVD, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.explain_variance_threshold = explain_variance_threshold
        self.perplexity = perplexity
        self.measured_rank_svd = measured_rank_svd
        self.layer_mem = layer_mem
        self.layer_idx=layer_idx

    def forward(self, input):
        if self.activate and torch.is_grad_enabled(): # Training mode
            output = Linear_measure_perplexity_HOSVD_op.apply(input, self.weight, self.bias, \
                                                  self.explain_variance_threshold, self.perplexity, self.measured_rank_svd, self.layer_mem, self.layer_idx)
        else: # activate is False or Validation mode
            output = super().forward(input)
        return output
    

def wrap_linear_measure_perplexity_HOSVD(linear, active, explain_variance_threshold, perplexity, measured_rank_svd, layer_mem, layer_idx):
    has_bias = (linear.bias is not None)
    new_linear = Linear_measure_perplexity_HOSVD(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        explain_variance_threshold = explain_variance_threshold,
                         perplexity = perplexity,
                         measured_rank_svd=measured_rank_svd,
                         layer_mem = layer_mem,
                         layer_idx=layer_idx
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear