import re
import types

import torch.nn
import torch.nn.init

from .common import conv1x1_block, Classifier,conv3x3_dw_blockAll,conv3x3_block
from .SE_Attention import *

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=True) / 6.0


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


def act_layer(name):
    if name is None:
        return nn.Identity()

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name in ("hswish", "hard_swish"):
        return HSwish()
    if name in ("hsigmoid", "hard_sigmoid"):
        return HSigmoid()
    if name in ("swish", "silu"):
        return nn.SiLU(inplace=True)

    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        groups=1,
        dilation=1,
        activation="relu",
        use_bn=True,
    ):
        padding = ((kernel_size - 1) // 2) * dilation
        layers = [
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=not use_bn,
            )
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))

        if activation is not None:
            layers.append(act_layer(activation))

        super().__init__(*layers)


class ECALayer(nn.Module):
    """
    Efficient Channel Attention.
    Nhẹ hơn SE vì không dùng FC/1x1 reduce-expand.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()

        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        k = max(k, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            padding=(k - 1) // 2,
            bias=False,
        )
        self.gate = HSigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                         # B, C, 1, 1
        y = y.squeeze(-1).transpose(-1, -2)          # B, 1, C
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)        # B, C, 1, 1
        return x * self.gate(y)


class GhostModule(nn.Module):
    """
    Thay 1x1 conv thường bằng Ghost module.
    ratio=2 nghĩa là chỉ tạo khoảng một nửa feature bằng conv chính,
    phần còn lại sinh bằng depthwise cheap operation.
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        activation="relu",
        use_bn=True,
    ):
        super().__init__()

        init_ch = math.ceil(out_ch / ratio)
        new_ch = init_ch * (ratio - 1)

        self.out_ch = out_ch

        self.primary = ConvBNAct(
            in_ch=in_ch,
            out_ch=init_ch,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            use_bn=use_bn,
        )

        self.cheap = ConvBNAct(
            in_ch=init_ch,
            out_ch=new_ch,
            kernel_size=dw_size,
            stride=1,
            groups=init_ch,
            activation=activation,
            use_bn=use_bn,
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch, :, :]


class DropPath(nn.Module):
    """
    Stochastic depth nhẹ. Mặc định drop_prob=0.0 nên không ảnh hưởng.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask
class FR_PDP_block_v2(nn.Module):
    """
    FR-PDP-v2 for TickNet.

    Main:
        Ghost/PW expand -> DW kxk -> ECA -> Ghost/PW linear project

    Shortcut:
        Identity hoặc AvgPool + 1x1 linear projection

    Thay đổi quan trọng so với bản cũ:
        1. Có expansion để tăng biểu diễn.
        2. Projection cuối không activation.
        3. Shortcut projection không activation.
        4. ECA đặt ở hidden channels.
        5. Depthwise kernel mặc định 5x5 để tăng spatial receptive field.
        6. Ghost pointwise để giữ nhẹ.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio=2.0,
        kernel_size=5,
        activation="hswish",
        use_ghost=True,
        ghost_ratio=2,
        use_eca=True,
        drop_path=0.0,
    ):
        super().__init__()

        assert stride in (1, 2), "TickNet thường chỉ dùng stride 1 hoặc 2"
        assert kernel_size in (3, 5, 7), "Nên dùng kernel_size 3, 5 hoặc 7"

        hidden_channels = make_divisible(in_channels * expand_ratio, 8)
        hidden_channels = max(hidden_channels, in_channels)

        if use_ghost:
            self.pw_expand = GhostModule(
                in_ch=in_channels,
                out_ch=hidden_channels,
                kernel_size=1,
                ratio=ghost_ratio,
                activation=activation,
            )
        else:
            self.pw_expand = ConvBNAct(
                in_ch=in_channels,
                out_ch=hidden_channels,
                kernel_size=1,
                stride=1,
                activation=activation,
            )

        self.dw = ConvBNAct(
            in_ch=hidden_channels,
            out_ch=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=hidden_channels,
            activation=activation,
        )

        self.attn = ECALayer(hidden_channels) if use_eca else nn.Identity()

        if use_ghost:
            self.pw_project = GhostModule(
                in_ch=hidden_channels,
                out_ch=out_channels,
                kernel_size=1,
                ratio=ghost_ratio,
                activation=None,      # linear bottleneck
            )
        else:
            self.pw_project = ConvBNAct(
                in_ch=hidden_channels,
                out_ch=out_channels,
                kernel_size=1,
                stride=1,
                activation=None,      # linear bottleneck
            )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                ) if stride > 1 else nn.Identity(),

                ConvBNAct(
                    in_ch=in_channels,
                    out_ch=out_channels,
                    kernel_size=1,
                    stride=1,
                    activation=None,  # residual projection nên là tuyến tính
                ),
            )

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.pw_expand(x)
        x = self.dw(x)
        x = self.attn(x)
        x = self.pw_project(x)
        x = self.drop_path(x)

        return x + residual

class TickNet(torch.nn.Module):
    """
    Class for constructing TickNet.    
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block_v2(in_channels=in_channels, out_channels=unit_channels, stride=stride, expand_ratio=2.0, kernel_size=5, use_ghost=True, use_eca=True,)
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

###
#%% model definitions
###
def build_TickNet(num_classes, typesize='small', cifar=False):
    init_conv_channels = 32
    if typesize=='basic':
        channels = [[128],[64],[128],[256],[512]]
    if typesize=='small':
        channels = [[128],[64,128],[256,512,128],[64,128,256],[512]]
    if typesize=='large':
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        if typesize=='basic':
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]
    return  TickNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size)
