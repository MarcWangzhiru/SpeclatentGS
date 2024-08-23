import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial


def embedding_fn(inputs):
    embed_fns = []
    d = 3
    out_dim = 0
    max_freq = 3  # 3

    freq_bands = 2. ** torch.linspace(0., max_freq, steps=4)  # [2**0, 2**1,...,]
    for freq in freq_bands:
        for p_fn in [torch.sin, torch.cos]:
            embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))  # [torch.sin, torch.cos]
            out_dim += d

    return torch.cat([fn(inputs) for fn in embed_fns], -1)

class BasicConv(nn.Module):
    # Gated_conv
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, relu=True, dilation=1,
                 padding_mode='reflect', act_fun=nn.ELU, normalization=nn.BatchNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        group_normalization = nn.GroupNorm

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)
        self.flag = relu

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                    padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                    padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels),
                'group_norm': group_normalization(num_groups=out_channels, num_channels=out_channels),
            }
        )

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        output = features
        return output


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )
        self.relu = nn.ELU()
        self.norm = nn.GroupNorm(1, out_channel)

    def forward(self, x):
        return self.relu(self.norm(self.main(x) + (x)))

class ResNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResNet, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
            nn.BatchNorm2d(out_channel)

        )
        self.net = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.norm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ELU()

    def forward(self, x):
        output = self.main(x) + self.norm(self.net(x))
        return self.relu(output)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(8, out_plane - 8, kernel_size=3, stride=1, relu=True),
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=1):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=1):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


class MyNet(nn.Module):
    def __init__(
            self,
            num_input_channels=8,
            num_output_channels=3,
            feature_scale=4,
            num_res=1

    ):
        super().__init__()

        self.feature_scale = feature_scale
        base_channel = 8

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        base_channel = 8

        self.resnet = nn.ModuleList([
            ResNet(base_channel, base_channel * 2),
            ResNet(base_channel * 2, base_channel * 4),
            ResNet(base_channel * 4, base_channel * 8),
            ResNet(base_channel * 8, base_channel * 4),
            ResNet(base_channel * 4, base_channel * 2),
            ResNet(base_channel * 2, base_channel),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(8, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=4, relu=True, stride=2),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 1, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 4, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 8 + 24, base_channel * 16, kernel_size=3, relu=True, stride=1),
            # nn.Linear(base_channel * 8 + 24, base_channel*16),
            BasicConv(base_channel * 16, base_channel * 4, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(24, base_channel * 8, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 8, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 8, 20, kernel_size=3, relu=False, stride=1),

            BasicConv(base_channel, base_channel * 8, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 8 + 24, base_channel, kernel_size=3, relu=True, stride=1),
            # BasicConv(base_channel * 8 + 24, base_channel, kernel_size=3, relu=True, stride=1),
            # BasicConv(base_channel * 8 , base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, 3, kernel_size=3, relu=True, stride=1),

            BasicConv(24, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, 20, kernel_size=3, relu=False, stride=1),
            # BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
            # BasicConv(base_channel*2+24, base_channel, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=False, stride=1),

            BasicConv(40, base_channel, kernel_size=3, relu=False, stride=1),

            # 29开始引入hash
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 8, 3, kernel_size=3, relu=False, stride=1),

            # 31 开始MLP:
            nn.Linear(8, 64, bias=True),
            nn.Linear(64 + 24, 64, bias=True),
            nn.Linear(64, 3, bias=True)

        ])

        self.SCM0 = SCM(base_channel * 8)
        self.SCM1 = SCM(base_channel * 4)
        self.SCM2 = SCM(base_channel * 2)

        self.FAM0 = FAM(base_channel * 8)
        self.FAM1 = FAM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 15, base_channel * 1),
            AFF(base_channel * 15, base_channel * 2),
            AFF(base_channel * 15, base_channel * 4),
        ])

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
            EBlock(base_channel * 8, num_res)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 8, num_res),
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 8, base_channel * 8, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 2, kernel_size=1, relu=True, stride=1),

        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    # basic_model
    def forward(self, *inputs, **kwargs):
        inputs = list(inputs)

        n_input = len(inputs)

        x = inputs[0][:, :8, :, :]

        f = inputs[0][:, 8:16, :, :]

        sh_f = list(f.shape)
        sh_f[1] = 3

        tint = inputs[0][:, 16, :, :]

        views = inputs[0][:, 20:, :, :]

        # highlight
        f = self.feat_extract[21](f)
        f = self.Encoder[3](f)
        # print(f.shape, views.shape)
        f = torch.cat((f, views), dim=1)
        f = self.feat_extract[22](f)
        f = self.Decoder[3](f)
        f = self.feat_extract[23](f)

        mask = tint

        highlight = f * mask

        s = x
        z1 = x

        # color
        x = self.feat_extract[1](x)
        x = self.Encoder[1](x)
        z2 = x

        x = self.feat_extract[2](x)
        x = self.Encoder[2](x)
        z3 = x

        x = self.feat_extract[6](x)
        x = self.Encoder[3](x)

        x = self.up(x)

        x = self.Convs[0](x)
        x = F.interpolate(x, z3.shape[-2:])
        x = torch.cat([x, z3], dim=1)
        x = self.Decoder[0](x)

        x = self.feat_extract[9](x)
        x = self.up(x)

        x = self.Convs[1](x)
        x = F.interpolate(x, z2.shape[-2:])
        x = torch.cat([x, z2], dim=1)
        x = self.Decoder[1](x)

        x = self.feat_extract[10](x)
        x = self.up(x)

        x = self.Convs[2](x)
        x = F.interpolate(x, z1.shape[-2:])
        x = torch.cat([x, z1], dim=1)
        x = self.Decoder[2](x)

        x = self.feat_extract[11](x)

        z = self.feat_extract[5](x)

        color = z

        z = z + highlight

        return {'im_out': z,
                's_out': s,
                'mask_out': mask,
                'highlight_out': highlight,
                'f_out': f,
                'color_out': color}


if __name__ == '__main__':
    import pdb
    import time
    import numpy as np

    # model = UNet().to('cuda')
    model = MyNet().to('cuda')
    input = []
    img_sh = [1408, 376]
    sh_unit = 8
    # img_sh = list(map(lambda a: a - a % sh_unit + sh_unit if a % sh_unit != 0 else a, img_sh))

    # print(img_sh)
    down = lambda a, b: a // 2 ** b
    input.append(torch.zeros((1, 43, down(img_sh[0], 0), down(img_sh[1], 0)), requires_grad=True).cuda())
    input.append(F.interpolate(input[0], scale_factor=0.5))
    input.append(F.interpolate(input[1], scale_factor=0.5))
    input.append(F.interpolate(input[2], scale_factor=0.5))
    print(input)

    model.eval()
    st = time.time()
    print(input[0].max(), input[0].min())
    print(input[0].shape, input[1].shape, input[2].shape, input[3].shape)
    with torch.set_grad_enabled(False):
        out = model(*input)
        pdb.set_trace()
        print('model', time.time() - st)
    print(out['im_out'], out['im_out'].shape)
    print(out['s_out'], out['s_out'].shape)
    model.to('cpu')
