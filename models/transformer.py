import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = dilation, groups = groups, bias = False, dilation = dilation)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class E_MHSA(nn.Module):
    """
    Effecient Multi-Head Self-Attention (E-MHSA)
    Parameters
    ----------
    dim : int
        Number of input channels.
    heads : int
        Number of heads.
    inner_dim : int
        Number of hidden channels for each head.
    dropout : float
        Dropout rate.
    stride : int
        Stride of the convolutional block.
    """
    def __init__(self, dim, heads = 8, inner_dim = 64 , dropout = 0.,stride = 2):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.heads = heads
        self.scaled_factor = inner_dim ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(dim)
        self.avg_pool = nn.AvgPool2d(stride, stride = stride)

        self.fc_q = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_k = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_v = nn.Linear(dim, self.inner_dim * self.heads)
        self.fc_o = nn.Linear(self.inner_dim * self.heads, dim)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.bn(x)
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)  # [b, h * w, c]

         # Get q, k, v
        q = self.fc_q(x_reshape)
        # [b, heads, h * w, inner_dim]
        q = q.view(b, h * w, self.heads, self.inner_dim).permute(0, 2, 1, 3).contiguous()

        k = self.fc_k(x_reshape)
        k = k.view(b, self.heads * self.inner_dim, h, w)
        k = self.avg_pool(k)
        # [b, heads, h * w, inner_dim]
        k = rearrange(k, "b (head n) h w -> b head (h w) n", head = self.heads)

        v = self.fc_v(x_reshape)
        v = v.view(b, self.heads * self.inner_dim, h, w)
        v = self.avg_pool(v)
        # [b, heads, h * w, inner_dim]
        v = rearrange(v, "b (head n) h w -> b head (h w) n", head = self.heads)

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, s_h * s_w], s_h = s_h // stride

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.inner_dim)
        result = self.fc_o(result).view(b, self.dim, h, w)
        result = result + x
        return result



class TransformerBlock(nn.Module):
    """
    Implement transformer block in the paper "Attention is all you need"
    Paper: https://arxiv.org/abs/1706.03762
    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    dim_linear_block:
        Number of hidden features of the linear transform
    activation:
        Activation function apply in the linear transform layer
    dropout:
        Dropout rate in the linear transform layers
    mhsa:
        Optional[MultiHeadSelfAttention object | None]
    """
    def __init__(self,
        in_dim,
        out_dim = None,
        stride = 1,
        num_heads = 4,
        dim_head = 32,
        activation = nn.GELU,
        dropout = 0.1
    ):
        super(TransformerBlock, self).__init__()
        out_dim = in_dim if out_dim == None else out_dim
        self.stride = stride
        self.dropout = nn.Dropout(p = dropout)

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mhsa = E_MHSA(in_dim, num_heads, dim_head)
        self.linear_transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation(),
            nn.Dropout(dropout)
        )
        self.down_conv = conv3x3(out_dim, out_dim, stride = 2, groups = 8)

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply LayerNorm then MHSA
        x = rearrange(x, 'B C H W -> B (H W) C')
        x = self.norm1(x)
        x = rearrange(x, 'B (H W) C -> B C H W', H = H, W = W)
        x = self.mhsa(self.dropout(x)) + x

        # Apply LayerNorm then Feed-Forward
        x = rearrange(x, 'B C H W -> B (H W) C')
        x = self.norm2(x)
        x = self.linear_transform(self.norm2(x))
        out = rearrange(x, 'B (H W) C -> B C H W', H = H, W = W)

        # Down Sampling
        if self.stride != 1:
            out = self.down_conv(out)
        return out


if __name__ == "__main__":
    mhsa = E_MHSA(3)
    x = torch.randn(1, 3, 32, 32)
    print(f"mhsa(x).shape = {mhsa(x).shape}")

    transformer = TransformerBlock(in_dim = 3, out_dim = 32, num_heads = 8, dim_head = 64)
    print(transformer(x).shape)


