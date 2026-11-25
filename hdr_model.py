from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from slice import bilateral_slice


class L2LOSS(nn.Module):

    def forward(self, x, y):
        return torch.mean((x-y)**2)

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

        if use_bias and not batch_norm:
            self.conv.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        torch.nn.init.kaiming_uniform_(self.conv.weight)#, mode='fan_out',nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None  
        
        if not batch_norm:
            self.fc.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        torch.nn.init.kaiming_uniform_(self.fc.weight)#, mode='fan_out',nonlinearity='relu')

        
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        bilateral_grid = bilateral_grid.permute(0,3,4,2,1)
        guidemap = guidemap.squeeze(1)
        # grid: The bilateral grid with shape (gh, gw, gd, gc).
        # guide: A guide image with shape (h, w). Values must be in the range [0, 1].
        coeefs = bilateral_slice(bilateral_grid, guidemap).permute(0,3,1,2)
        return coeefs
        # Nx12x8x16x16
        # print(guidemap.shape)
        # print(bilateral_grid.shape)
        # device = bilateral_grid.get_device()
        # N, _, H, W = guidemap.shape
        # hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        # if device >= 0:
        #     hg = hg.to(device)
        #     wg = wg.to(device)
        # hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1)# * 2 - 1 # norm to [-1,1] NxHxWx1
        # wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1)# * 2 - 1 # norm to [-1,1] NxHxWx1
        # guidemap = guidemap.permute(0,2,3,1).contiguous()
        # guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        # # When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear. 
        # coeff = F.grid_sample(bilateral_grid, guidemap_guide, 'bilinear')#, align_corners=True)
        # print(coeff.shape)
        # return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        # out_channels = []
        # for chan in range(n_out):
        #     ret = scale[:, :, :, chan, 0]*input_image[:, :, :, 0]
        #     for chan_i in range(1, n_in):
        #         ret += scale[:, :, :, chan, chan_i]*input_image[:, :, :, chan_i]
        #     if has_affine_term:
        #         ret += offset[:, :, :, chan]
        #     ret = tf.expand_dims(ret, 3)
        #     out_channels.append(ret)

        # ret = tf.concat(out_channels, 3)
        """
            R = r1[0]*r2 + r1[1]*g2 + r1[2]*b3 +r1[3]
        """

        # print(coeff.shape)
        # R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        # G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        # B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 9:10, :, :]
        G = torch.sum(full_res_input * coeff[:, 3:6, :, :], dim=1, keepdim=True) + coeff[:, 10:11, :, :]
        B = torch.sum(full_res_input * coeff[:, 6:9, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, params['guide_complexity'], kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(params['guide_complexity'], 1, kernel_size=1, padding=0, activation= nn.Sigmoid) #nn.Tanh nn.Sigmoid

    def forward(self, x):
        return self.conv2(self.conv1(x))  #.squeeze(1)


class Coeffs(nn.Module):

    def __init__(self, nin=4, nout=3, params=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin
        self.nout = nout

        lb = params['luma_bins']  # 8
        cm = params['channel_multiplier']  # default = 1
        sb = params['spatial_bin']  # 16
        bn = params['batch_norm']
        net_input_size = params['net_input_size']  # default=256

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(net_input_size / sb))  # 256：4  32：1
        self.splat_features = nn.ModuleList()
        # prev_ch = 3
        prev_ch = 512
        # for i in range(n_layers_splat):  # 1:3->8   2:8->16  3:16->32   3:32->64   # 1:3->16   2:16->32  3:32->64   3:64->128
        #     use_bn = bn if i > 0 else False
        #     self.splat_features.append(ConvBlock(prev_ch, cm*(2**i)*lb, 3, stride=2, batch_norm=use_bn))
        #     prev_ch = splat_ch = cm*(2**i)*lb

        # global features
        n_layers_global = int(np.log2(sb / 4))  #2
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        self.global_features_fc_32 = nn.ModuleList()
        for i in range(n_layers_global):  # 1:512->64   2:64->64
            self.global_features_conv.append(ConvBlock(prev_ch, cm * 8 * lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm * 8 * lb

        n_total = n_layers_splat + n_layers_global  # 6   32:3
        prev_ch = prev_ch * (net_input_size / 2 ** n_total) ** 2  # 1024
        self.global_features_fc.append(FC(prev_ch, 32 * cm * lb, batch_norm=bn))  # 1024-> 256
        self.global_features_fc.append(FC(32 * cm * lb, 16 * cm * lb, batch_norm=bn))  # 256 -> 128
        self.global_features_fc.append(FC(16 * cm * lb, 8 * cm * lb, activation=None, batch_norm=bn))  # 128 -> 64

        # flatten_dim = 64 * (net_input_size // (2 ** (n_layers_global + 1))) ** 2  # 64 * 8 * 8 = 4096
        flatten_dim = 64 * 8 * 8  # 64 * 8 * 8 = 4096
        self.global_features_fc_32.append(FC(flatten_dim, 32 * 2 * cm * lb, batch_norm=bn))  # 4096-> 512
        self.global_features_fc_32.append(FC(32 * 2 * cm * lb, 32 * cm * lb, batch_norm=bn))  # 512 -> 256
        self.global_features_fc_32.append(FC(32 * cm * lb, 16 * cm * lb, batch_norm=bn))  # 256 -> 128
        self.global_features_fc_32.append(FC(16 * cm * lb, 8 * cm * lb, activation=None, batch_norm=bn))  # 128 -> 64

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(512, 32 * cm * lb, 3, batch_norm=bn))  # 512 -> 256
        self.local_features.append(ConvBlock(32 * cm * lb, 16 * cm * lb, 3, batch_norm=bn))  # 256 -> 128
        self.local_features.append(
            ConvBlock(16 * cm * lb, 8 * cm * lb, 3, activation=None, use_bias=False))  # 128 -> 64

        # predicton
        self.conv_out = ConvBlock(8 * cm * lb, lb * nout * nin, 1, padding=0,
                                  activation=None)  # ,batch_norm=True)   # 64 -> 96

    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']

        x = lowres_input
        # for layer in self.splat_features:
        #     x = layer(x)
        # splat_features = x
        # print(f"Coeffs input shape: {x.shape}")
        for layer in self.global_features_conv:
            x = layer(x)
            # print(f"Coeffs global_features_conv shape : {x.shape}")
        # x = x.view(bs, -1)
        # print(f"Coeffs x.view(bs, -1) shape: {x.shape}")
        # if lowres_input.shape[2] == 16:
        #     for layer in self.global_features_fc:
        #         x = layer(x)
        #         # print(f"Coeffs global_features_fc shape : {x.shape}")
        # else:
        #     for layer in self.global_features_fc_32:
        #         x = layer(x)
                # print(f"Coeffs global_features_fc shape : {x.shape}")
        x = F.adaptive_avg_pool2d(x,(4,4))
        x = x.view(bs,-1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        splat_features = lowres_input
        # print(f"Coeffs splat_features_inputs shape : {splat_features.shape}")
        for layer in self.local_features:
            splat_features = layer(splat_features)
            # print(f"Coeffs splat_features shape : {splat_features.shape}")
        local_features = splat_features
        # print(f"Coeffs local_features shape: {local_features.shape}")
        fusion_grid = local_features
        fusion_global = global_features.view(bs, 8 * cm * lb, 1, 1)
        # print(f"Coeffs fusion_global shape: {fusion_global.shape}")
        fusion = self.relu(fusion_grid + fusion_global)
        # print(f"Coeffs fusion shape: {fusion.shape}")
        x = self.conv_out(fusion)  # ([10, 96, 16, 16])
        # print(f"Coeffs conv_out shape: {x.shape}")
        s = x.shape
        y = torch.stack(torch.split(x, self.nin * self.nout, 1), 2)  # ([10, 12, 8, 16, 16])
        # print(f"Coeffs y shape: {y.shape}")

        # y = torch.stack(torch.split(y, self.nin, 1),3)
        # print(y.shape)
        # x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        # print(x.shape)
        return y


class HDRPointwiseNN(nn.Module):

    def __init__(self, params):
        super(HDRPointwiseNN, self).__init__()
        self.coeffs = Coeffs(params=params)
        self.guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        # self.bsa = bsa.BilateralSliceApply()

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        # print(f"HDRPointwiseNN lowres shape: {lowres.shape}")  #([10, 512, 16, 16])
        # print(f"HDRPointwiseNN fullres shape: {fullres.shape}")  #([10, 3, 32, 32])
        # print(f"HDRPointwiseNN coeffs shape: {coeffs.shape}")  #([10, 12, 8, 16, 16])
        guide = self.guide(fullres)
        # print(f"HDRPointwiseNN guide shape: {guide.shape}")  #([10, 1, 32, 32])
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        # print(f"HDRPointwiseNN out shape: {out.shape}")  #([10, 3, 32, 32])
        # out = bsa.bsa(coeffs,guide,fullres)
        return out


class Args:

    params = {
        'luma_bins': 8,
        'channel_multiplier': 1,
        'spatial_bin': 16,
        'guide_complexity': 16,
        'batch_norm': False,
        'net_input_size': 256,
    }

    # import argparse

    # parser = argparse.ArgumentParser(description='HDRNet Inference')
    # parser.add_argument('--ckpt-path', type=str, default='./ch', help='Model checkpoint path')
    # parser.add_argument('--test-image', type=str, dest="test_image", help='Test image path')
    # parser.add_argument('--test-out', type=str, default='out.png', dest="test_out", help='Output test image path')

    # parser.add_argument('--luma-bins', type=int, default=8)
    # parser.add_argument('--channel-multiplier', default=1, type=int)
    # parser.add_argument('--spatial-bin', type=int, default=16)
    # parser.add_argument('--guide-complexity', type=int, default=16)

    # parser.add_argument('--batch-norm', action='store_true', help='If set use batch norm')
    # parser.add_argument('--net-input-size', type=int, default=256, help='Size of low-res input')
    # parser.add_argument('--net-output-size', type=int, default=512, help='Size of full-res input/output')

    # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # parser.add_argument('--batch-size', type=int, default=6)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--log-interval', type=int, default=10)
    # parser.add_argument('--ckpt-interval', type=int, default=100)
    # parser.add_argument('--dataset', type=str, default='', help='Dataset path with input/output dirs', required=True)
    # parser.add_argument('--dataset-suffix', type=str, default='', help='Add suffix to input/output dirs. Useful when train on different dataset image sizes')

    # params = vars(parser.parse_args())

    # print('PARAMS:')
    # print(params)

    # train(params=params)

