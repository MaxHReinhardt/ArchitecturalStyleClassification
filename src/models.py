import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, ch):
        super(ChannelAttentionModule, self).__init__()

        ratio = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(ch, ch//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//ratio, ch, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = self.mlp(x1.view(x1.size(0), -1))
        x1 = x1.view(x1.size(0), x1.size(1), 1, 1)

        x2 = self.max_pool(x)
        x2 = self.mlp(x2.view(x2.size(0), -1))
        x2 = x2.view(x2.size(0), x2.size(1), 1, 1)

        feats = x1 + x2
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()

        kernel_size = 7
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats


class CbamIntegration(nn.Module):
    def __init__(self, channel):
        super(CbamIntegration, self).__init__()

        self.ca = ChannelAttentionModule(channel)
        self.sa = SpatialAttentionModule()

    def forward(self, x):
        residual = x
        x = self.ca(x)
        x = self.sa(x)
        out = x + residual
        return out


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes, with_cbam=False):
        super(MobileNetV1, self).__init__()

        # Define standard convolution with batch norm
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        # Define depth-wise separable convolution
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # depth-wise convolution
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # point-wise convolution
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.convolutional_layers = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1)
        )

        self.with_cbam = with_cbam

        self.cbam_last_layer = CbamIntegration(1024)

        self.final_pool = nn.AdaptiveAvgPool2d(1)

        self.final_linear = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.convolutional_layers(x)
        if self.with_cbam:
            x = self.cbam_last_layer(x)
        x = self.final_pool(x)
        x = x.view(-1, 1024)
        x = self.final_linear(x)
        return x

