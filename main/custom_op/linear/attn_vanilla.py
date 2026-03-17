import torch
import torch.nn as nn
import torch.nn.functional as F
# from .linear import Linear 


def wrap_attn_vanilla(attn):
    """
    Wraps a PyTorch MultiheadAttention module into a custom MultiheadAttention_vanilla module,
    replacing the out_proj Linear layer with a custom Linear layer while preserving all weights.
    
    Args:
        attn: torch.nn.MultiheadAttention instance
    Returns:
        MultiheadAttention_vanilla: Custom attention module with copied weights
    """
    has_bias = (attn.out_proj.bias is not None)
    new_attn = MultiheadAttention_vanilla(
        embed_dim=attn.embed_dim,
        num_heads=attn.num_heads,
        dropout=attn.dropout.p if isinstance(attn.dropout, nn.Dropout) else attn.dropout,
        bias=has_bias,
        batch_first=attn.batch_first
    )

    # Extract Q, K, V weights from in_proj_weight
    in_proj_weight = attn.in_proj_weight
    embed_dim = attn.embed_dim
    assert in_proj_weight.shape == (3 * embed_dim, embed_dim), "Unexpected in_proj_weight shape"
    
    # Split in_proj_weight into Q, K, V weights
    q_weight, k_weight, v_weight = torch.split(in_proj_weight, embed_dim, dim=0)
    new_attn.q_proj.weight.data.copy_(q_weight)
    new_attn.k_proj.weight.data.copy_(k_weight)
    new_attn.v_proj.weight.data.copy_(v_weight)
    
    # Copy out_proj weights
    new_attn.out_proj.weight.data.copy_(attn.out_proj.weight.data)
    
    # Handle biases if they exist
    if has_bias:
        if attn.in_proj_bias is not None:
            q_bias, k_bias, v_bias = torch.split(attn.in_proj_bias, embed_dim, dim=0)
            new_attn.q_proj.bias.data.copy_(q_bias)
            new_attn.k_proj.bias.data.copy_(k_bias)
            new_attn.v_proj.bias.data.copy_(v_bias)
        new_attn.out_proj.bias.data.copy_(attn.out_proj.bias.data)

    return new_attn


class MultiheadAttention_vanilla(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.batch_first = batch_first

        # Use custom Linear for all projections, including out_proj
        # self.q_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        # self.k_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        # self.v_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        # self.out_proj = Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.q_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        self.k_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        self.v_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False
    ):
        # Handle causal attention if specified
        if is_causal and attn_mask is None:
            # Create causal mask: upper triangle is masked (future tokens cannot be attended to)
            attn_mask = torch.triu(
                torch.ones(query.size(-2), key.size(-2), device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))

        # Shape: [B, N, C] if batch_first else [N, B, C]
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, N, C = query.shape

        # Project Q/K/V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Split into heads
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]
        
        # Apply key_padding_mask if provided
        if key_padding_mask is not None:
            # key_padding_mask shape: [B, N] or [N]
            # Expand to [B, 1, 1, N] for broadcasting
            key_padding_mask = key_padding_mask.view(-1, 1, 1, key_padding_mask.size(-1))
            attn_scores = attn_scores.masked_fill(key_padding_mask == 0, float('-inf'))

        # Apply attn_mask if provided
        if attn_mask is not None:
            # Ensure attn_mask is compatible: [B, 1, N, N] or [N, N]
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            else:
                attn_scores += attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]

        output = self.out_proj(attn_output)

        if not self.batch_first:
            output = output.transpose(0, 1)  # back to [N, B, C]

        # Handle need_weights and average_attn_weights
        if need_weights:
            if average_attn_weights:
                # Average attention weights across heads
                attn_weights = attn_weights.mean(dim=1)  # [B, N, N]
            return output, attn_weights
        return output, None