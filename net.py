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

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_conv = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.reg_conv = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        cls = self.cls_conv(x)
        reg = self.reg_conv(x)
        return cls, reg

# # Faster R-CNN model combining ResNet-18 and RPN
class FasterRCNN(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(FasterRCNN, self).__init__()
        self.backbone = CustomResNet18()
        self.rpn = RPN(128, num_anchors)
        self.fc_detection = nn.Linear(512, num_classes)

    def forward(self, images, targets=None):
        x = self.backbone(images)
        cls, reg = self.rpn(x)

        # Additional detection head
        detection_output = self.fc_detection(x)

        if self.training:
            assert targets is not None, "During training, targets should not be None."
            # Your training code here, calculate loss, etc.

        return cls, reg, detection_output

# Example usage with random data
num_detection_classes = 5  # Example: 5 classes for detection
num_anchors = 9  # Example: 9 anchors per spatial location
model = FasterRCNN(num_detection_classes, num_anchors)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example forward pass with random data
images = torch.rand((2, 3, 512, 512)).to(device)  # Example batch of images
targets = {'cls_targets': torch.randint(0, num_detection_classes, (2,)), 'reg_targets': torch.rand((2, 4))}  # Example targets
cls_output, reg_output, detection_output = model(images, targets)
print(cls_output.shape, reg_output.shape, detection_output.shape)

# Example usage with random data
num_anchors = 9
in_channels = 64
batch_size = 2
height, width = 512, 512

# Create random input tensor
x = torch.randn((batch_size, in_channels, height, width))

# Initialize and forward pass through RPN
rpn = RPN(in_channels, num_anchors)
cls_output, reg_output = rpn(x)

# Print output shapes
print("CLS Output Shape:", cls_output.shape)
print("REG Output Shape:", reg_output.shape)
