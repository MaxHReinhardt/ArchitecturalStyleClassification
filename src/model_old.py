import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel):
        super(ChannelAttentionModule, self).__init__()
        ratio = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(channel, channel//ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//ratio, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = self.mlp(x1.view(x1.size(0), -1))
        x1 = x1.view(x1.size(0), x1.size(1), 1, 1)

        x2 = self.max_pool(x)
        x2 = self.mlp(x2.view(x2.size(0), -1))
        x2 = x2.view(x2.size(0), x2.size(1), 1, 1)

        attention = x1 + x2
        attention = self.sigmoid(attention)
        refined_x = x * attention

        return refined_x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        kernel_size = 7
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([x1, x2], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        refined_x = x * attention

        return refined_x


class CbamIntegration(nn.Module):
    def __init__(self, channel):
        super(CbamIntegration, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        residual = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        out = x + residual
        return out


class ConvolutionBatchNorm(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ConvolutionBatchNorm, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn(x)


class DepthWiseSeparableConvolution(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(DepthWiseSeparableConvolution, self).__init__()
        self.depthwise_separable_conv = nn.Sequential(
            # depth-wise convolution
            nn.Conv2d(input_channel, input_channel, 3, stride, 1, groups=input_channel, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            # point-wise convolution
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.depthwise_separable_conv(x)


class FinalClassifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super(FinalClassifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fully_connected_layer = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        return x


class MobileNetV1_old(nn.Module):
    def __init__(self, input_channels, n_classes, width_multiplier=1, with_cbam=False):
        super(MobileNetV1_old, self).__init__()
        width_divisor = 1/width_multiplier

        self.convolutional_layers = nn.Sequential(
            ConvolutionBatchNorm(input_channels, int(32//width_divisor), 2),
            DepthWiseSeparableConvolution(int(32 // width_divisor), int(64 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(64 // width_divisor), int(128 // width_divisor), 2),
            DepthWiseSeparableConvolution(int(128 // width_divisor), int(128 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(128 // width_divisor), int(256 // width_divisor), 2),
            DepthWiseSeparableConvolution(int(256 // width_divisor), int(256 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(256 // width_divisor), int(512 // width_divisor), 2),
            DepthWiseSeparableConvolution(int(512 // width_divisor), int(512 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(512 // width_divisor), int(512 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(512 // width_divisor), int(512 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(512 // width_divisor), int(512 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(512 // width_divisor), int(512 // width_divisor), 1),
            DepthWiseSeparableConvolution(int(512 // width_divisor), int(1024 // width_divisor), 2),
            DepthWiseSeparableConvolution(int(1024 // width_divisor), int(1024 // width_divisor), 1)
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
