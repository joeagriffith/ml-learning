import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        assert d_embed % n_heads == 0, f"d_embed {d_embed} must be divisible by n_heads {n_heads}"

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: (Batch_size, Seq_Len, Dim)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape        
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, 3*Dim) -> 3 x (Batch_size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1) 

        # (Batch_size, Seq_Len, Dim) -> (Batch_size, Seq_Len, n_heads, d_head) -> (Batch_size, n_heads, Seq_Len, d_head)
        q = q.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        k = k.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        v = v.view(*interim_shape).permute(0, 2, 1, 3).contiguous()

        # (Batch_size, n_heads, Seq_Len, d_head) -> (Batch_size, n_heads, Seq_Len, Seq_Len)
        scores = q @ k.transpose(-2, -1)

        if causal_mask:
            mask = torch.ones_like(scores, dtype=torch.bool).triu(1)
            scores.masked_fill_(mask, float('-inf'))
        
        scores /= math.sqrt(self.d_head)

        scores = scores.softmax(dim=-1)

        # Batch_size, n_heads, Seq_Len, Seq_Len) @ (Batch_size, n_heads, Seq_Len, d_head) -> (Batch_size, n_heads, Seq_Len, d_head)
        output = scores @ v

        # (Batch_size, n_heads, Seq_Len, d_head) -> (Batch_size, Seq_Len, n_heads, d_head) -> (Batch_size, Seq_Len, Dim)
        output = output.permute(0, 2, 1, 3).contiguous().view(*input_shape)

        output = self.out_proj(output)

        # (Batch_size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()

        assert d_embed % n_heads == 0, f"d_embed {d_embed} must be divisible by n_heads {n_heads}"

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    
    def forward(self, query, keys_n_values):
        # query: (Batch_size, Seq_Len_Q, Dim_Q)
        # keys_n_values: (Batch_size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = query.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply query by Wq
        q = self.q_proj(query)
        k = self.k_proj(keys_n_values)
        v = self.v_proj(keys_n_values)

        q = q.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        k = k.view(*interim_shape).permute(0, 2, 1, 3).contiguous()
        v = v.view(*interim_shape).permute(0, 2, 1, 3).contiguous()

        scores = q @ k.transpose(-1, -2)
        scores /= math.sqrt(self.d_head)
        scores = F.softmax(scores, dim=-1)

        output = scores @ v
        output = output.permute(0, 2, 1, 3).contiguous().view(*input_shape)
        output = self.out_proj(output)
        return output




class VAE_AttentionBlock(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.attention = SelfAttention(1, in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        residual = x

        n, c, h, w = x.shape
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height*Width, Channel)
        x = x.view(n, c, h*w).permute(0, 2, 1).contiguous()

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Height*Width, Channel)
        x = self.attention(x)

        # (Batch_Size, Height*Width, Channel) -> (Batch_Size, Channel, Height, Width)
        x = x.permute(0, 2, 1).contiguous().view(n, c, h, w)

        return x + residual

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)

        residual = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residual)




