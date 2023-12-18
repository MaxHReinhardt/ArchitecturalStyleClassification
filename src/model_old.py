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


class ConvBN(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvBN, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn(x)


class ConvDW(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvDW, self).__init__()
        self.conv_dw = nn.Sequential(
            # depth-wise convolution
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # point-wise convolution
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_dw(x)


class FinalClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(FinalClassifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MobileNetV1_old(nn.Module):
    def __init__(self, ch_in, n_classes, width_multiplier=1, with_cbam=False):
        super(MobileNetV1_old, self).__init__()
        width_divisor = 1/width_multiplier

        self.convolutional_layers = nn.Sequential(
            ConvBN(ch_in, int(32//width_divisor), 2),
            ConvDW(int(32//width_divisor), int(64//width_divisor), 1),
            ConvDW(int(64//width_divisor), int(128//width_divisor), 2),
            ConvDW(int(128//width_divisor), int(128//width_divisor), 1),
            ConvDW(int(128//width_divisor), int(256//width_divisor), 2),
            ConvDW(int(256//width_divisor), int(256//width_divisor), 1),
            ConvDW(int(256//width_divisor), int(512//width_divisor), 2),
            ConvDW(int(512//width_divisor), int(512//width_divisor), 1),
            ConvDW(int(512//width_divisor), int(512//width_divisor), 1),
            ConvDW(int(512//width_divisor), int(512//width_divisor), 1),
            ConvDW(int(512//width_divisor), int(512//width_divisor), 1),
            ConvDW(int(512//width_divisor), int(512//width_divisor), 1),
            ConvDW(int(512//width_divisor), int(1024//width_divisor), 2),
            ConvDW(int(1024//width_divisor), int(1024//width_divisor), 1)
        )

        self.with_cbam = with_cbam
        self.cbam = CbamIntegration(int(1024//width_divisor))

        self.final_classifier = FinalClassifier(int(1024//width_divisor), n_classes)

    def forward(self, x):
        x = self.convolutional_layers(x)
        if self.with_cbam:
            x = self.cbam(x)
        x = self.final_classifier(x)
        return x
