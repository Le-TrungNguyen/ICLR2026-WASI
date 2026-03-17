########################################################################
# This is pseudo implementation
# Because Torch Auto grad doesn't allow returning a gradient with a shape different from the input in the forward function, the weights must be retained and passed to the custom function (Linear_WASI_op) as dummy input to store the gradient. further engineering effort is needed to improve this class in the future.
########################################################################

import torch
import torch.nn as nn
from torch.autograd import Function

from ..compression.rank.power_iteration import decompose_tensor_keep_projection
from ..compression.rank.hosvd_power_iteration import hosvd_power_iteration
from torch.nn import functional as F

def SVD_var(weight, var, use_k=False, device='cuda'):
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    if use_k:
        k = int(var)
        return U[:, :k], torch.diag(S[:k]), Vt[:k, :], k
    else:
        total_variance = torch.sum(S ** 2)
        explained_variance = torch.cumsum(S ** 2, dim=0) / total_variance
        k = torch.searchsorted(
            explained_variance,
            torch.tensor(var, device=S.device, dtype=S.dtype)
        ).item() + 1
        return U[:, :k], torch.diag(S[:k]), Vt[:k, :], k

#######################################################################

class Linear_WASI_op(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, L, R, S_activation, U_list_activation):
        out = F.linear(F.linear(input, R, None), L, bias)
        ctx.save_for_backward(S_activation, L, R, bias)
        ctx.U_list_activation = U_list_activation
        return out

    @staticmethod
    def backward(ctx, grad_output):
        S, L, R, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None


        if ctx.needs_input_grad[0]:
            grad_input = F.linear(F.linear(grad_output, L.t(), None), R.t(), None)

        if ctx.needs_input_grad[1]:

            if len(ctx.U_list_activation) == 4:
                U1, U2, U3, U4 = ctx.U_list_activation
                Z1 = torch.einsum("Ba,BHWD->aHWD", U1, grad_output)   # (K1,H,W,D)
                Z2 = torch.einsum("Hb,abcd->aHcd", U2, S)            # (K1,H,K3,K4)
                Z3 = torch.einsum("Wc,aHWD->aHcD", U3, Z1)           # (K1,H,K3,D)
                Z4 = torch.einsum("Cd,aHcd->aHCc", U4, Z2)           # (K1,H,C,K3)
                grad_weight = torch.einsum("aHcD,aHCc->DC", Z3, Z4)       # (out, in)

            elif len(ctx.U_list_activation) == 3:
                U1, U2, U3 = ctx.U_list_activation
                Z1 = torch.einsum("blo,bk->lok", grad_output, U1)   # (L,O,K1)
                Z2 = torch.einsum("abc,lb->acl", S, U2)             # (K1,K3,L)
                Z3 = torch.einsum("acl,ic->ail", Z2, U3)            # (K1,I,L)
                grad_weight = torch.einsum("lok,kil->oi", Z1, Z3)        # (out, in)



        if bias is not None and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output.dim() - 1))
            grad_bias = grad_output.sum(dim=reduce_dims)

        return grad_input, grad_weight, grad_bias, None, None, None, None


    
class Linear_WASI(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activation_ranks=1,
            explained_variance_threshold=1.0):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.device = device


        self.activation_ranks = activation_ranks
        self.explained_variance_threshold = explained_variance_threshold

        self.reuse_map = False

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


        self.weight_rank = None
        
        # for activation
        self.u_list_activation = None

    def forward(self, input):
        ### Keep projection
        if torch.is_grad_enabled(): # Training mode
            # Decompose activation
            S_activation, self.u_list_activation = hosvd_power_iteration(input, previous_Ulist=self.u_list_activation, reuse_U=self.reuse_map, rank=self.activation_ranks)

            if not self.reuse_map:
                with torch.no_grad():
                    # Decompose weight
                    Uk_torch, Sk_torch, Vtk_torch, k = SVD_var(self.weight.clone().detach(), self.explained_variance_threshold)
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


            if input.dim() in (3, 4):
                output = Linear_WASI_op.apply(input, self.weight, self.bias, self.L, self.R, S_activation, self.u_list_activation)
            else:
                raise ValueError("Not implemented for input dim = {}".format(input.dim()))

        else: # Validation mode
            output = F.linear(F.linear(input, self.R, None), self.L, self.bias)

        return output
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination.pop(prefix + "weight", None)
    

def wrap_linearWASI(linear, activation_ranks, explained_variance_threshold):
    has_bias = (linear.bias is not None)
    new_linear = Linear_WASI(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activation_ranks=activation_ranks,
                        explained_variance_threshold=explained_variance_threshold
                        )
    

    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear