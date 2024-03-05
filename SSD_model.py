import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

class SSDHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.loc = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)  # 4 bounding box offsets
        self.conf = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)  # class scores

    def forward(self, x):
        print("Shape of input tensor x:", x.shape)  # Print shape of input tensor x
        loc = self.loc(x)
        conf = self.conf(x)
        print("Shape of loc tensor:", loc.shape)  # Print shape of loc tensor
        print("Shape of conf tensor:", conf.shape)  # Print shape of conf tensor
        return loc, conf

class SSDMobileNetV2(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Base MobileNetV2 network
        self.feature_extractor = nn.Sequential(
            ConvBlock(input_shape[0], 32, 3, 2),
            DepthwiseSeparableConv(32, 64),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 512, stride=2)
        )

        # Additional feature layers for SSD
        self.additional_features = nn.ModuleList([
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 128, stride=2),
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 64, stride=2),
        ])
        
        # Prediction heads
        self.prediction_heads = nn.ModuleList([
            SSDHead(512, num_classes),  # Make sure in_channels matches the number of channels produced by the last feature layer
            SSDHead(256, num_classes),  # Make sure in_channels matches the number of channels produced by the corresponding feature layer
            SSDHead(256, num_classes),  # Make sure in_channels matches the number of channels produced by the corresponding feature layer
            SSDHead(128, num_classes),  # Make sure in_channels matches the number of channels produced by the corresponding feature layer
            SSDHead(128, num_classes),  # Make sure in_channels matches the number of channels produced by the corresponding feature layer
            SSDHead(64, num_classes),   # Make sure in_channels matches the number of channels produced by the corresponding feature layer
        ])

    def forward(self, x):
        confidences = []
        locations = []
        
        x = self.feature_extractor(x)
        
        # Apply additional feature layers and prediction heads
        for feature, head in zip(self.additional_features, self.prediction_heads):
            x = feature(x)
            loc, conf = head(x)
            confidences.append(conf.permute(0, 2, 3, 1).contiguous())
            locations.append(loc.permute(0, 2, 3, 1).contiguous())

        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1)
        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1)
        
        confidences = confidences.view(confidences.size(0), -1, self.num_classes)  # [batch, num_priors, num_classes]
        locations = locations.view(locations.size(0), -1, 4)  # [batch, num_priors, 4]
        
        return confidences, locations
