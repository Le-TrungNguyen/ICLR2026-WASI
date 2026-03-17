import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d, pad
import torch.nn as nn
from ..compression.rank.hosvd_power_iteration import hosvd_power_iteration, restore_hosvd_power_iteration
from ..compression.explain_var.hosvd_4_mode_var import hosvd_4_mode_var

def SVD_var(weight, var, use_k=False):
    U, S, Vt = th.linalg.svd(weight, full_matrices=False)

    if use_k:
        return U[:, :var], th.diag_embed(S[:var]), Vt[:var, :], U, th.diag_embed(S), Vt, var

    else:
        total_variance = th.sum(S**2)
        explained_variance = th.cumsum(S**2, dim=0) / total_variance
        k = th.searchsorted(explained_variance, var).item() + 1
        Vt_k = Vt[:k, :]
        return U[:, :k], th.diag_embed(S[:k]), Vt_k, U, th.diag_embed(S), Vt, k
    
def restore_tensor(qt, p, shape):
    tensor = p@qt
    tensor = tensor.reshape(shape[0], shape[1], shape[2], shape[3])
    return tensor

class Conv2d_WSI_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, S, u_list, shape = args

        # Performing convolution with weight restored from low rank components - just for testing
        weight_ = restore_hosvd_power_iteration(S, u_list)
        output = conv2d(input, weight_, bias, stride, padding, dilation=dilation, groups=groups)

        ctx.save_for_backward(input, weight_, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        input, weight, bias  = ctx.saved_tensors

        B, C, H, W = input.shape

        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs
        
        # Compute gradient with respect to the input
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input((B,C,H,W), weight, grad_output, stride, padding, dilation, groups)

        # Compute gradient with respect to the weights
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None
    
class Conv2d_WSI_inference_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, bias, stride, dilation, padding, groups, S, u_list, shape = args

        weight_ = restore_hosvd_power_iteration(S, u_list)
        output = conv2d(input, weight_, bias, stride, padding, dilation=dilation, groups=groups)

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass

class Conv2d_WSI(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            rank=1
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        super(Conv2d_WSI, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.rank = rank
        self.previous_rank = None
        self.reuse_U = False
        self.u_list_weight = None


    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.activate:
            if th.is_grad_enabled(): # Training mode

                if not self.reuse_U:
                    S, self.u_list_weight, self.previous_rank = hosvd_4_mode_var(self.weight, self.rank, True) # HOSVD 4 mode

                    self.reuse_U = True
                else:
                    S, self.u_list_weight = hosvd_power_iteration(self.weight, previous_Ulist=self.u_list_weight, reuse_U=self.reuse_U, rank=self.previous_rank) # HOSVD 4 mode


                y = Conv2d_WSI_op.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, S, self.u_list_weight, [self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]])
            else: # Inference mode
                S, u_list_weight = hosvd_power_iteration(self.weight, previous_Ulist=self.u_list_weight, reuse_U=self.reuse_U, rank=self.previous_rank) # HOSVD 4 mode

                y = Conv2d_WSI_inference_op.apply(x, self.bias, self.stride, self.dilation, self.padding, self.groups, S, u_list_weight, [self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]])

        else: # activate is False or Inference mode
            y = super().forward(x)
        return y

def wrap_convWSI(conv, active, rank, layer_idx=None):

    new_conv = Conv2d_WSI(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         rank=rank
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv