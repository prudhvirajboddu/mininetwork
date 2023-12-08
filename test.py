# import torch 
# import torch.nn as nn


# def _make_layer(in_channels, out_channels, blocks, stride):
#         layers = []
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
#         layers.append(nn.BatchNorm2d(out_channels))
#         layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
#         layers.append(nn.BatchNorm2d(out_channels))
#         layers.append(nn.ReLU(inplace=True))
#         return nn.Sequential(*layers)

# backbone = nn.Sequential(
#         nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#         nn.BatchNorm2d(64),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#         _make_layer(64, 64, 2, stride=1),
#         _make_layer(64, 128, 2, stride=2),
#         _make_layer(128, 256, 2, stride=2)
# )


# x = torch.randn(1, 3, 512, 512)

# out = backbone(x).shape

# print(out)


import torch
import torch.nn as nn

# Custom ResNet-18-like block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x

# Custom ResNet-18 model
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(ResNetBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNetBlock, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


model = CustomResNet18(num_classes=5)

x = torch.randn(1, 3, 512, 512)

out = model(x).shape

print(out)