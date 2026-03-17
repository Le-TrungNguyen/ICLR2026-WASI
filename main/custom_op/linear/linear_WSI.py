########################################################################
# This is pseudo implementation
# Because Torch Auto grad doesn't allow returning a gradient with a shape different from the input in the forward function, the weights must be retained and passed to the custom function (Linear_WSI_op) as dummy input to store the gradient. further engineering effort is needed to improve this class in the future.
########################################################################

import torch
import torch.nn as nn
from torch.autograd import Function

from ..compression.rank.power_iteration import decompose_tensor_keep_projection
from torch.nn import functional as F

def SVD_var(weight, var, use_k=False):
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    if use_k:
        return U[:, :var], torch.diag_embed(S[:var]), Vt[:var, :], U, torch.diag_embed(S), Vt, var

    else:
        total_variance = torch.sum(S**2)
        explained_variance = torch.cumsum(S**2, dim=0) / total_variance
        k = torch.searchsorted(explained_variance, var).item() + 1
        Vt_k = Vt[:k, :]
        return U[:, :k], torch.diag_embed(S[:k]), Vt_k, U, torch.diag_embed(S), Vt, k, S
    

#######################################################################

class Linear_WSI_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, US_k, Vt_k = args
        
        # Low rank forward pass
        output = input@Vt_k.t()@US_k.t()
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        ctx.save_for_backward(input, bias, US_k, Vt_k)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forward pass
        input, bias, US_k, Vt_k = ctx.saved_tensors

    
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output@US_k@Vt_k
        if ctx.needs_input_grad[1]:
            if input.dim() == 4:
                grad_weight = torch.einsum('bhwc,bhwd->dc', input, grad_output)
            elif input.dim() == 3:
                grad_weight = torch.einsum('bli,blo->oi', input, grad_output)
            else:
                raise ValueError("Chưa triển khai cho input có dim khác 3 hoặc 4")

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None

class Linear_WSI(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            rank=1,
            size = None,
            layer_idx=None,
            WSI_with_sub_iter=True):
        
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.rank = rank
        self.reuse_map = False

        self.WSI_with_sub_iter = WSI_with_sub_iter


        self.previous_weight = None
        self.previous_k = None

        self.size = size
        self.layer_idx = layer_idx

        # for weight
        # Because Torch Auto grad doesn't allow returning a gradient with a shape different from the input in the forward function, the weights must be retained and passed to the forward function as dummy input to store the gradient. further engineering effort is needed to improve this class in the future.
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.register_parameter("L", None)
        self.register_parameter("R", None)

    def forward(self, input):
        if self.WSI_with_sub_iter:
            ### Keep projection
            if torch.is_grad_enabled(): 
                if not self.reuse_map:
                    with torch.no_grad():
                        # Decompose weight
                        Uk_torch, Sk_torch, Vtk_torch, U, S, Vt, k, _ = SVD_var(self.weight.clone().detach(), self.rank)
                        self.weight_rank = k
                        R = (Sk_torch@Vtk_torch)


                    self.L = nn.Parameter(Uk_torch.contiguous()).requires_grad_(False)
                    self.R = nn.Parameter(R.contiguous()).requires_grad_(False)

                    self.reuse_map = True
                else:
                    with torch.no_grad():
                        L, Rt = decompose_tensor_keep_projection(self.weight, previous_p=self.L.data, reuse_p=self.reuse_map, rank=self.weight_rank, device='cuda')
                        self.L.copy_(L)
                        self.R.copy_(Rt.t())

                if torch.is_grad_enabled() and self.size is not None:
                    self.size[0].append(self.L.shape[0])
                    self.size[1].append(self.L.shape[1])
                    self.size[2].append(self.R.shape[1])
                    self.size[3].append(input.shape)
                output = Linear_WSI_op.apply(input, self.weight, self.bias, self.L, self.R)
            else: # Validation
                output = F.linear(F.linear(input, self.R, None), self.L, self.bias)
        else: # SVD all
            with torch.no_grad():
                Uk_torch, Sk_torch, Vtk_torch, U, S, Vt, k, eigen_values = SVD_var(self.weight.clone().detach(), self.rank)
                self.R = (Sk_torch@Vtk_torch)

                if not self.reuse_map:
                    self.L = nn.Parameter(Uk_torch.contiguous()).requires_grad_(False)
                    self.R = nn.Parameter(R.contiguous()).requires_grad_(False)
                    self.reuse_map = True
                else:
                    self.L.copy_(L)
                    self.R.copy_(Rt.t())


            if torch.is_grad_enabled() and self.size is not None:
                self.size[0].append(self.L.shape[0])
                self.size[1].append(self.L.shape[1])
                self.size[2].append(self.R.shape[1])
                self.size[3].append(input.shape)
                self.size[4][self.layer_idx].append(eigen_values)

            output = F.linear(F.linear(input, self.R, None), self.L, self.bias)

        return output
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination.pop(prefix + "weight", None)
    

def wrap_linearWSI(linear, rank, size, layer_idx, WSI_with_sub_iter=True):
    has_bias = (linear.bias is not None)
    new_linear = Linear_WSI(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        rank=rank,
                        WSI_with_sub_iter = WSI_with_sub_iter,
                        size=size,
                        layer_idx=layer_idx
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear