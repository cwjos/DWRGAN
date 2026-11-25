import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import functools
from torchvision import transforms
import tensorflow as tf
from hdr_model import HDRPointwiseNN, Args
from DegradationDescriptor import TransformerBlock

from torch.nn.utils import spectral_norm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="munch")


class Generator(nn.Module):
    """Generator network"""
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn):
        super(Generator, self).__init__()
        ###### encoder
        self.enc1 = ConvBlock(in_channels=3,          out_channels=conv_dim*1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*3 --> 256*256*32
        self.enc2 = ConvBlock(in_channels=conv_dim* 1, out_channels=conv_dim*2, kernel_size=3, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*32 --> 128*128*64
        self.enc3 = ConvBlock(in_channels=conv_dim* 2, out_channels=conv_dim*4, kernel_size=3, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*64 --> 64*64*128
        self.enc4 = ConvBlock(in_channels=conv_dim* 4, out_channels=conv_dim*8, kernel_size=3, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*128 --> 32*32*256
        self.enc5 = ConvBlock(in_channels=conv_dim* 8, out_channels=conv_dim*16, kernel_size=3, stride=2, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*256 --> 16*16*512

        ###### decoder
        self.upsample1 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim*16, conv_dim*8, 1, 1, 0, 1, True, use_sn))
        self.upsample2 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 8, conv_dim*4, 1, 1, 0, 1, True, use_sn))
        self.upsample3 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 4, conv_dim*2, 1, 1, 0, 1, True, use_sn))
        self.upsample4 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim* 2, conv_dim*1, 1, 1, 0, 1, True, use_sn))

        self.dec1 = ConvBlock(in_channels=conv_dim*16, out_channels=conv_dim*8, kernel_size=3, stride=1, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 32*32*512 --> 32*32*256
        self.dec2 = ConvBlock(in_channels=conv_dim* 8, out_channels=conv_dim*4, kernel_size=3, stride=1, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 64*64*256 --> 64*64*128
        self.dec3 = ConvBlock(in_channels=conv_dim* 4, out_channels=conv_dim*2, kernel_size=3, stride=1, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 128*128*128 --> 128*128*64
        self.dec4 = ConvBlock(in_channels=conv_dim* 2, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=1, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn) # 256*256*64 --> 256*256*32
        self.dec5 = nn.Sequential(
            SNConv(in_channels=conv_dim*1, out_channels=conv_dim*1, kernel_size=3, stride=1, padding=1, dilation=1, use_bias=True, use_sn=False),
            SNConv(in_channels=conv_dim*1, out_channels=3,          kernel_size=7, stride=1, padding=3, dilation=1, use_bias=True, use_sn=False),
            nn.Tanh()
        )

        self.ga5 = GAM(conv_dim*16, conv_dim*16, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga4 = GAM(conv_dim* 8, conv_dim* 8, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga3 = GAM(conv_dim* 4, conv_dim* 4, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga2 = GAM(conv_dim* 2, conv_dim* 2, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga1 = GAM(conv_dim* 1, conv_dim* 1, reduction=8, bias=False, use_sn=use_sn, norm=True)

        self.hdr = HDRPointwiseNN(Args.params)
        self.enc6 = ConvBlock(in_channels=3, out_channels=conv_dim * 8, kernel_size=3, stride=1, padding=1, dilation=1,
                              use_bias=True, norm_fun=norm_fun, act_fun=act_fun,
                              use_sn=use_sn)  # 256*256*64 --> 256*256*32

        
        self.swin1 = nn.Sequential(
            TransformerBlock(dim=conv_dim * 8, num_heads=8, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias'))
        self.swin2 = nn.Sequential(
            TransformerBlock(dim=conv_dim * 4, num_heads=4, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias'))
        self.swin3 = nn.Sequential(
            TransformerBlock(dim=conv_dim * 2, num_heads=2, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias'))
        self.swin4 = nn.Sequential(
            TransformerBlock(dim=conv_dim * 1, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias'))

        self.guide_reduce = nn.Conv2d(3 + conv_dim * 1, 3, kernel_size=1)
        self.res_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):

        # encoder
        x1 = self.enc1(x)  # 256*256*3 -> 256*256*32
        x1 = self.swin4(x1)
        x2 = self.enc2(x1)  # 256*256*32 -> 128*128*64
        x2 = self.swin3(x2)
        x3 = self.enc3(x2)  # 128*128*64 -> 64*64*128
        x3 = self.swin2(x3)
        x4 = self.enc4(x3)  # 64*64*128 -> 32*32*256
        x4 = self.swin1(x4)

        x5 = self.enc5(x4)  # 32*32*256 --> 16*16*512
        x5 = self.ga5(x5)  #([10, 512, 16, 16])

        guide_map = F.interpolate(x, size=(x4.shape[2], x4.shape[2]), mode='bilinear', align_corners=False)  # 使用输出图像本身作为引导图，保持尺寸一致
        guide_map = torch.cat([guide_map, F.interpolate(x1, size=(x4.shape[2], x4.shape[2]), mode='bilinear')], dim=1)
        guide_map = self.guide_reduce(guide_map)
        #
        hdr_x5 = self.hdr(x5, guide_map)  # HDR refinement 结果
        x5 = self.enc6(hdr_x5)   #([10, 256, 32, 32])

        # decoder
        # y1 = self.upsample1(x5)  #32*32*256
        # y1 = torch.cat([y1, self.ga4(x4)], dim=1)
        # y1 = self.dec1(y1)
        # y1 = self.swin1(y1)

        y2 = self.upsample2(x5)   #64*64*128
        y2 = torch.cat([y2, self.ga3(x3)], dim=1)
        y2 = self.dec2(y2)
        y2 = self.swin2(y2)
       
        y3 = self.upsample3(y2)   #128*128*64
        y3 = torch.cat([y3, self.ga2(x2)], dim=1)
        y3 = self.dec3(y3)
        y3 = self.swin3(y3)
        
        y4 = self.upsample4(y3)  #256*256*32
        y4 = torch.cat([y4, self.ga1(x1)], dim=1)
        y4 = self.dec4(y4)
        y4 = self.swin4(y4)
        
        res = self.dec5(y4.mul(x1))  # 逐像素乘法，增强局部细节
                out = torch.clamp((res + x), min=-1.0, max=1.0)

        return out




class SNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(SNConv, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(
            nn.ReflectionPad2d(self.padding),
            SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn),
        )

    def forward(self, x):
        return self.main(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
        super(ConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        main = []
        main.append(nn.ReflectionPad2d(self.padding))
        main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
        norm_fun = get_norm_fun(norm_fun)
        main.append(norm_fun(out_channels))
        main.append(get_act_fun(act_fun))
        self.main = nn.Sequential(*main)

    def forward(self, x):
        return self.main(x)


def maybe_sn(layer, use_sn):
    # 如果你有自己的 SN wrapper 可以替换这里
    try:
        from torch.nn.utils import spectral_norm as SN
        return SN(layer) if use_sn else layer
    except:
        return layer


class GAM(nn.Module):

    # Improved GAM Model
    def __init__(self, in_nc, out_nc, reduction=8, bias=False, use_sn=False, norm=False,
                 ks_list=(0, 3, 5), eps=1e-5):
        """
        ks_list: 0=全局统计, 其它为局部均值方差的核大小（奇数）
        """
        super().__init__()
        self.in_nc, self.out_nc, self.eps = in_nc, out_nc, eps
        self.ks_list = ks_list

        # 统计特征到 γ/β 的映射（多尺度拼接 → 1×1 → γ/β）
        # 每个尺度提供 2*in_nc（mean,std），总通道 = 2*in_nc*len(ks_list)
        feat_c = 2 * in_nc * len(ks_list)
        conv = lambda c_in, c_out: (maybe_sn(nn.Conv2d(c_in, c_out, 1, bias=bias)) if use_sn
                                    else nn.Conv2d(c_in, c_out, 1, bias=bias))
        self.to_gamma = conv(feat_c, out_nc)
        self.to_beta = conv(feat_c, out_nc)

        # 融合层（在调制后再做一次轻量 1×1），并采用残差 + 可学习缩放
        self.fuse = conv(out_nc, out_nc)
        nn.init.zeros_(self.fuse.weight)  # 零初始化：一开始等价恒等
        if getattr(self.fuse, 'bias', None) is not None:
            nn.init.zeros_(self.fuse.bias)

        self.alpha = nn.Parameter(torch.zeros(1))  # 学习到再放大，避免训练初期破坏分布

        # 可选归一化（建议：如果开 norm，改用 GroupNorm，避免抹平通道幅度）
        self.use_post_norm = norm
        if norm:
            self.post_norm = nn.GroupNorm(num_groups=min(32, out_nc), num_channels=out_nc)

    @staticmethod
    def _moments_local(x, k, eps):
        # 局部均值/方差（可微）：AvgPool 实现，padding 保持尺寸
        pad = k // 2
        mu = F.avg_pool2d(x, k, 1, pad)
        mu2 = F.avg_pool2d(x * x, k, 1, pad)
        var = (mu2 - mu * mu).clamp_min(0.0)
        std = torch.sqrt(var + eps)
        return mu, std

    def _gather_stats(self, x):
        B, C, H, W = x.shape
        feats = []
        for ks in self.ks_list:
            if ks == 0:
                # 全局统计（通道维，空间共享），再 broadcast 成 H×W
                mu = x.mean(dim=(2, 3), keepdim=True)
                std = x.std(dim=(2, 3), keepdim=True) + self.eps
                mu = mu.expand(B, C, H, W)
                std = std.expand(B, C, H, W)
            else:
                mu, std = self._moments_local(x, ks, self.eps)
            feats.append(mu)
            feats.append(std)
        return torch.cat(feats, dim=1)  # [B, 2*C*len(ks), H, W]

    def forward(self, x):
        # 1) 实例归一化（保留内容，去除幅度/对比度偏置）
        mu_c = x.mean(dim=(2, 3), keepdim=True)
        std_c = x.std(dim=(2, 3), keepdim=True) + self.eps
        x_in = (x - mu_c) / std_c

        # 2) 多尺度统计特征 → γ/β（空间可变）
        stats = self._gather_stats(x)  # [B, 2*C*S, H, W]
        gamma = self.to_gamma(stats)
        beta = self.to_beta(stats)

        # 3) AdaIN 调制 + 轻量融合 + 残差
        y = x_in * (1.0 + gamma) + beta
        y = self.fuse(y)
        y = x + torch.tanh(self.alpha) * y  # 受控残差，初始≈恒等

        if self.use_post_norm:
            y = self.post_norm(y)
        return y

    

class Discriminator(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(Discriminator, self).__init__()

        # scale 1 and prediction of scale 1         128
        d_1 = [dis_conv_block(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 2       64
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=7, stride=2, padding=3, dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 3 and prediction of scale 3          32
        d_3 = [dis_conv_block(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim* 4, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 4  16
        d_4 = [dis_conv_block(in_channels=conv_dim*4, out_channels=conv_dim*8, kernel_size=5, stride=2, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        # scale 5 and prediction of scale 5         8
        d_5 = [dis_conv_block(in_channels=conv_dim*8, out_channels=conv_dim*16, kernel_size=5, stride=2, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_5_pred = [dis_pred_conv_block(in_channels=conv_dim*16, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)
        self.d5 = nn.Sequential(*d_5)
        self.d5_pred = nn.Sequential(*d_5_pred)

    def forward(self, x):
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)

        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)

        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)

        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)

        ds5 = self.d5(ds4)
        ds5_pred = self.d5_pred(ds5)

        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred, ds5_pred]


def dis_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
    norm_fun = get_norm_fun(norm_fun)
    main.append(norm_fun(out_channels))
    main.append(get_act_fun(act_fun))
    main = nn.Sequential(*main)
    return main


# 预测图像的真伪
def dis_pred_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias))
    if type in ['ls', 'rals']:
        main.append(nn.Sigmoid())
    elif type in ['hinge', 'rahinge']:
        main.append(nn.Tanh())
    else:
        raise NotImplementedError("Adversarial loss [{}] is not found".format(type))
    main = nn.Sequential(*main)
    return main


def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.data.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=False)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=False)
        elif act_fun_type == 'Swish':
            return Swish()
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=False)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun
