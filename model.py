import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomObjectDetectionModel, self).__init__()

        # Feature extractor using the first five layers of ResNet18
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        # Bounding box regression
        self.conv_bb1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_bb2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc_bb = nn.Linear(512 * 128 * 128, 4)  # 4 for bounding box coordinates (x, y, width, height)

        # Classification
        self.conv_cls1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_cls2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc_cls = nn.Linear(512 * 128 * 128, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)

        # Bounding box regression
        x_bb = F.relu(self.conv_bb1(x))
        x_bb = F.relu(self.conv_bb2(x_bb))
        x_bb = x_bb.view(x_bb.size(0), -1)
        x_bb = self.fc_bb(x_bb)

        # Classification
        x_cls = F.relu(self.conv_cls1(x))
        x_cls = F.relu(self.conv_cls2(x_cls))
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_cls = self.fc_cls(x_cls)

        return {"bounding_box": x_bb, "class_label": x_cls}

    
backbone = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),  
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)

)


x = torch.randn(1, 3, 512,512)

out_features = CustomObjectDetectionModel()(x)

print(out_features['bounding_box'])

print(out_features['class_label'])