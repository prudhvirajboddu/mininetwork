import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class CustomObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomObjectDetectionModel, self).__init__()

        # Convolutional layers inspired by ResNet-18
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Bounding box regression
        self.fc_bb1 = nn.Linear(256 * 32 * 32, 256)
        self.relu_bb = nn.ReLU()
        self.fc_bb2 = nn.Linear(256, 4)

        # Classification
        self.fc_cls1 = nn.Linear(256 * 32 * 32, 256)
        self.relu_cls = nn.ReLU()
        self.fc_cls2 = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Reshape before passing through fully connected layers
        x = x.view(x.size(0), -1)

        # Bounding box regression
        x_bb = self.relu_bb(self.fc_bb1(x))
        x_bb = self.fc_bb2(x_bb)

        # Classification
        x_cls = self.relu_cls(self.fc_cls1(x))
        x_cls = self.fc_cls2(x_cls)

        return {"bounding_box": x_bb, "class_label": x_cls}


# x = torch.randn(1, 3, 512, 512)

# model = CustomObjectDetectionModel()(x)

# print(model["bounding_box"])
# print(model["class_label"])