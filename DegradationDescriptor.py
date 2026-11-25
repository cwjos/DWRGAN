import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, activation='gelu'):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # add
        if activation == 'silu':
            self.act = F.silu
        else:
            self.act = F.gelu

    def forward(self, x):

        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x
class HaarWaveletTransform(nn.Module):
    """
    Channel-wise Haar wavelet decomposition.

    Input:
        x: [B, C, H, W]
    Output:
        ll, lh, hl, hh: [B, C, H/2, W/2]
    """

    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return ll, lh, hl, hh


class InverseHaarWaveletTransform(nn.Module):
    """
    Channel-wise inverse Haar wavelet reconstruction.
    """

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape

        x00 = (ll - lh - hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll + lh + hl + hh) * 0.5

        out = ll.new_zeros(b, c, h * 2, w * 2)

        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11

        return out


class DegradationDescriptor(nn.Module):
    """
    Predicts three spatial degradation descriptors from feature maps:
        illumination: under-exposure / low-frequency degradation
        noise: high-frequency noise corruption
        detail: high-frequency detail deficiency
    """

    def __init__(self, dim, bias=True):
        super().__init__()

        hidden_dim = max(dim // 2, 16)

        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1, bias=True)
        )

    def forward(self, x):
        degradation = torch.sigmoid(self.body(x))

        illumination = degradation[:, 0:1]
        noise = degradation[:, 1:2]
        detail = degradation[:, 2:3]

        return illumination, noise, detail


class DegradationWaveletAttention(nn.Module):
    """
    Degradation-aware Wavelet Structure-Texture Routing Attention.

    Three restoration paths:
    1. Low-frequency structure path:
       restores illumination and global structure using LL features.

    2. High-frequency texture path:
       routes LH/HL/HH subbands according to predicted noise/detail maps.

    3. Channel relation path:
       models global inter-channel correlations on low-frequency features.

    The final fusion weights are spatially predicted from degradation maps,
    instead of using fixed summation or conventional skip concatenation.
    """

    def __init__(self, dim, num_heads=4, bias=True, eps=1e-6):
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.dwt = HaarWaveletTransform()
        self.idwt = InverseHaarWaveletTransform()

        # Predict illumination/noise/detail degradation maps.
        self.degradation_estimator = DegradationDescriptor(dim, bias=bias)

        # Low-frequency structural recovery branch.
        self.low_freq_refine = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=5,
                padding=2,
                groups=dim,
                bias=bias
            ),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )

        # Direction-aware high-frequency restoration branch.
        self.high_freq_refine = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(
                dim * 3,
                dim * 3,
                kernel_size=3,
                padding=1,
                groups=dim * 3,
                bias=bias
            ),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=1, bias=bias)
        )

        # Generates directional high-frequency modulation:
        # LH, HL, HH each receive independent adaptive weights.
        self.high_freq_router = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # Channel relation branch.
        # It is applied to low-frequency feature maps rather than raw x.
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Learn a spatially varying mixture among:
        # structure feature, texture feature, channel-relation feature.
        self.branch_router = nn.Sequential(
            nn.Conv2d(dim + 3, dim, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, 3, kernel_size=1, bias=True)
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Stabilizes training when replacing existing attention weights.
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def _channel_relation(self, x):
        """
        Computes transposed attention on LL features.

        This is intentionally only one branch of the module. It is not the
        claimed novelty; frequency-aware degradation routing is the core.
        """
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(
            q,
            'b (head c) h w -> b head c (h w)',
            head=self.num_heads
        )
        k = rearrange(
            k,
            'b (head c) h w -> b head c (h w)',
            head=self.num_heads
        )
        v = rearrange(
            v,
            'b (head c) h w -> b head c (h w)',
            head=self.num_heads
        )

        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)

        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = attention * self.temperature
        attention = attention.softmax(dim=-1)

        out = torch.matmul(attention, v)

        out = rearrange(
            out,
            'b head c (h w) -> b (head c) h w',
            h=h,
            w=w
        )

        return out

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape

        # Haar decomposition requires even height and width.
        pad_h = h % 2
        pad_w = w % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 1. Infer the local degradation state from full-resolution features.
        illumination, noise, detail = self.degradation_estimator(x)

        # 2. Decompose features into low-frequency structure and three
        # directional high-frequency subbands.
        ll, lh, hl, hh = self.dwt(x)

        low_condition = F.interpolate(
            illumination,
            size=ll.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        high_condition = F.interpolate(
            torch.cat([illumination, noise, detail], dim=1),
            size=ll.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # 3. Low-frequency structural restoration.
        # Stronger illumination deficiency produces stronger correction.
        ll_refined = self.low_freq_refine(ll)
        ll_refined = ll + low_condition * ll_refined

        # 4. Direction-aware high-frequency restoration.
        high_feature = torch.cat([lh, hl, hh], dim=1)
        high_residual = self.high_freq_refine(high_feature)

        lh_residual, hl_residual, hh_residual = high_residual.chunk(3, dim=1)
        lh_gate, hl_gate, hh_gate = self.high_freq_router(
            high_condition
        ).chunk(3, dim=1)

        # Noise suppresses high-frequency amplification.
        noise_gate = 1.0 - F.interpolate(
            noise,
            size=ll.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        detail_gate = F.interpolate(
            detail,
            size=ll.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        texture_gate = noise_gate * detail_gate

        lh_refined = lh + texture_gate * lh_gate * lh_residual
        hl_refined = hl + texture_gate * hl_gate * hl_residual
        hh_refined = hh + texture_gate * hh_gate * hh_residual

        # Reconstruct frequency-aware feature map.
        wavelet_feature = self.idwt(
            ll_refined,
            lh_refined,
            hl_refined,
            hh_refined
        )

        # 5. Global channel relation restoration on low-frequency structure.
        channel_feature = self._channel_relation(ll_refined)
        channel_feature = F.interpolate(
            channel_feature,
            size=wavelet_feature.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        # 6. Spatially varying three-branch routing.
        degradation = torch.cat([illumination, noise, detail], dim=1)

        router_input = torch.cat([x, degradation], dim=1)
        routing_weights = self.branch_router(router_input)
        routing_weights = torch.softmax(routing_weights, dim=1)

        structure_weight = routing_weights[:, 0:1]
        texture_weight = routing_weights[:, 1:2]
        relation_weight = routing_weights[:, 2:3]

        # Structure: reconstructed LL-dominant representation.
        structure_feature = self.idwt(
            ll_refined,
            torch.zeros_like(lh_refined),
            torch.zeros_like(hl_refined),
            torch.zeros_like(hh_refined)
        )

        # Texture: high-frequency reconstruction only.
        texture_feature = wavelet_feature - structure_feature

        out = (
            structure_weight * structure_feature
            + texture_weight * texture_feature
            + relation_weight * channel_feature
        )

        out = self.project_out(out)

        # Crop to original resolution for odd-sized inputs.
        out = out[:, :, :h, :w]

        return residual[:, :, :h, :w] + self.res_scale * out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        ffn_expansion_factor=2.66,
        bias=True,
        LayerNorm_type='WithBias',
        drop_path=0.0,
        layer_scale_init_value=1e-5,
        activation='gelu'
    ):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        self.attn = DegradationWaveletAttention(
            dim=dim,
            num_heads=num_heads,
            bias=bias
        )

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(
            dim=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            activation=activation
        )

        self.drop_path = DropPath(drop_path)

        self.gamma1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )
        self.gamma2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma1.view(1, -1, 1, 1) * self.attn(self.norm1(x))
        )

        x = x + self.drop_path(
            self.gamma2.view(1, -1, 1, 1) * self.ffn(self.norm2(x))
        )

        return x

