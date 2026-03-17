############################
## Better version can be found in main/
############################

import torch
import torch.nn as nn
from torch.autograd import Function
import time
from torch.nn import functional as F

from ..compression.power_iteration import decompose_tensor, decompose_tensor_keep_projection
from ..compression.hosvd_power_iteration import hosvd_power_iteration, restore_hosvd_power_iteration

def SVD_var(weight, var, use_k=False):
    U, S, Vt = torch.linalg.svd(weight, full_matrices=False)

    if use_k:
        return U[:, :var], torch.diag_embed(S[:var]), Vt[:var, :], U, torch.diag_embed(S), Vt, var

    else:
        total_variance = torch.sum(S**2)
        explained_variance = torch.cumsum(S**2, dim=0) / total_variance
        k = torch.searchsorted(explained_variance, var).item() + 1
        # US_k = U[:, :k]@torch.diag_embed(S[:k])
        Vt_k = Vt[:k, :]
        return U[:, :k], torch.diag_embed(S[:k]), Vt_k, U, torch.diag_embed(S), Vt, k

#######################################################################

class Linear_WASI_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, L, R, S_activation, U_list_activation, backward_time, forward_time, energy_logger, output_calculation_time = args
        
        if energy_logger is not None:
            energy_logger.resume()

        start_f = time.time()

        # Infer output
        output = F.linear(F.linear(input, R, None), L, bias)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        ctx.save_for_backward(S_activation, L, R, bias)
        ctx.U_list_activation = U_list_activation
        end_f = time.time()
        forward_time[-1] += (end_f-start_f)
        if output_calculation_time is not None: output_calculation_time.append(end_f-start_f)

        if energy_logger is not None:
            energy_logger.pause()
        ctx.energy_logger = energy_logger

        ctx.backward_time = backward_time

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        S, L, R, bias = ctx.saved_tensors
    
        grad_input = grad_weight = grad_bias = None

        backward_time = ctx.backward_time

        energy_logger = ctx.energy_logger
        if energy_logger is not None:
            energy_logger.resume()

        start = time.time()

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
            grad_bias = grad_output.sum(0).squeeze(0)

        end = time.time()
        backward_time.append(end - start)

        if energy_logger is not None:
            energy_logger.pause()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None
    
class Linear_WASI(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activation_ranks=1,
            explained_variance_threshold=1.0,
            backward_time=None,
            forward_time=None,
            inference_time=None,
            energy_logger=None,
            random_seed=233,
            output_calculation_time = None,
            orthogonalization_time = None,
            matmuls_time = None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.backward_time = backward_time
        self.forward_time = forward_time
        self.inference_time = inference_time

        self.activation_ranks = activation_ranks
        self.explained_variance_threshold = explained_variance_threshold

        self.reuse_map = False

        # for weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.register_parameter("L", None)
        self.register_parameter("R", None)

        # for activation
        self.u_list_activation = None

        self.energy_logger = energy_logger
        self.random_seed=random_seed

        #### Micro-benchmark #####
        self.output_calculation_time = output_calculation_time
        self.orthogonalization_time = orthogonalization_time
        self.matmuls_time = matmuls_time

    def forward(self, input):
        ### Keep projection
        if torch.is_grad_enabled(): # Training mode
            if self.orthogonalization_time is not None: self.orthogonalization_time.append(0.0)
            if self.matmuls_time is not None: self.matmuls_time.append(0.0)


            # Decompose activation
            S_activation, self.u_list_activation = hosvd_power_iteration(input, previous_Ulist=self.u_list_activation, reuse_U=self.reuse_map, rank=self.activation_ranks, device=input.device, random_seed=self.random_seed, orthogonalization_time=self.orthogonalization_time, matmuls_time=self.matmuls_time)

            if self.energy_logger is not None:
                self.energy_logger.resume()

            if not self.reuse_map:
                if self.energy_logger is not None: # Don't measure the initialization since it can be done online
                    self.energy_logger.pause()
                
                with torch.no_grad():
                # Decompose weight
                    Uk_torch, Sk_torch, Vtk_torch, U, S, Vt, k = SVD_var(self.weight.clone().detach(), self.explained_variance_threshold)
                    self.weight_rank = k
                    R = (Sk_torch@Vtk_torch)
                    self.reuse_map = True

                self.L = nn.Parameter(Uk_torch.contiguous()).requires_grad_(False)
                self.R = nn.Parameter(R.contiguous()).requires_grad_(False)
                
                if self.energy_logger is not None:
                    self.energy_logger.resume()

                ##############################
                with torch.no_grad():
                    L, Rt = decompose_tensor_keep_projection(self.weight, previous_p=self.L.data, reuse_p=self.reuse_map, rank=self.weight_rank, device=input.device, orthogonalization_time=self.orthogonalization_time, matmuls_time=self.matmuls_time)
                    self.L.copy_(L)
                    self.R.copy_(Rt.t())

            else:
                with torch.no_grad():
                    L, Rt = decompose_tensor_keep_projection(self.weight, previous_p=self.previous_p, reuse_p=self.reuse_map, rank=self.weight_rank, device=input.device, orthogonalization_time=self.orthogonalization_time, matmuls_time=self.matmuls_time)
                    self.L.copy_(L)
                    self.R.copy_(Rt.t())

            if self.energy_logger is not None:
                self.energy_logger.pause()

            self.forward_time.append(self.orthogonalization_time[-1] + self.matmuls_time[-1])


            if input.dim() in (3, 4):
                output = Linear_WASI_op.apply(input, self.weight, self.bias, self.L, self.R, S_activation, self.u_list_activation, self.backward_time, self.forward_time, self.energy_logger, self.output_calculation_time)
            else:
                raise ValueError("Not implemented for input dim = {}".format(input.dim()))

        else: # inference
            start_infer = time.time()
            output = F.linear(F.linear(input, self.R, None), self.L, self.bias)
            end_infer = time.time()
            self.inference_time.append(end_infer-start_infer)

        return output
    

def wrap_linearWASI(linear, activation_ranks, explained_variance_threshold, backward_time, forward_time, inference_time, energy_logger, random_seed=233,
                    output_calculation_time = None,
                    orthogonalization_time = None,
                    matmuls_time = None):
    has_bias = (linear.bias is not None)
    new_linear = Linear_WASI(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activation_ranks=activation_ranks,
                        explained_variance_threshold=explained_variance_threshold,
                        backward_time = backward_time,
                        forward_time = forward_time,
                        inference_time = inference_time,
                        energy_logger = energy_logger,
                        random_seed=random_seed,
                        output_calculation_time = output_calculation_time,
                        orthogonalization_time = orthogonalization_time,
                        matmuls_time = matmuls_time
                        )

    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear