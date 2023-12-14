import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, box_iou

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling for 512x512 images
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Adjust feature map sizes for 512 input
        self.res1 = ResNetBlock(64, 128, 128)
        self.res2 = ResNetBlock(128, 256, 256)
        self.res3 = ResNetBlock(256, 256, 512)
        self.res4 = ResNetBlock(256, 512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Max pooling
        x = self.maxpool(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.avgpool(x)

        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Fixed: Identity mapping
        identity = x
        out = self.layers(x)

        # Ensure identity has the same dimensions as out
        if identity.size() != out.size():
            identity = F.pad(identity, (0, 0, 0, 0, 0, out.size(1) - identity.size(1)))

        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out


NUM_BOXES = 5
NUM_CLASSES = 5

class FaceRecogNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.cnn = ResNet()  

        self.bbox_reg = nn.Linear(512, NUM_BOXES*4)
       
        self.cls_head = nn.Linear(512, NUM_CLASSES)  

    def forward(self, x):

        x = self.cnn(x)

        x = x.view(x.size(0), -1)
        
        predicted_boxes = self.bbox_reg(x) 
        predicted_classes = self.cls_head(x)
        
        predicted_boxes = predicted_boxes.reshape(x.size(0), NUM_BOXES, 4)
        predicted_classes = predicted_classes.reshape(x.size(0), NUM_CLASSES)

        return predicted_boxes, predicted_classes
    
x = torch.randn(8, 3, 512, 512)

model =  FaceRecogNet() (x)

print(model[0].shape)

print(model[1].shape)

    
def compute_loss(predicted_boxes, predicted_classes, target):

    bbox_losses = []
    cls_losses = []
    
    for b, c, t in zip(predicted_boxes, predicted_classes, target):
        
        bbox_loss = nn.MSELoss()(b, t['bounding_box'])
        cls_loss = nn.CrossEntropyLoss()(c, t['class_label'])  
        
        bbox_losses.append(bbox_loss) 
        cls_losses.append(cls_loss)

    total_bbox_loss = sum(bbox_losses) / len(bbox_losses)
    total_cls_loss = sum(cls_losses) / len(cls_losses)
    
    return total_cls_loss, total_bbox_loss
    
        
# # Training
# for epoch in range(num_epochs):

#     batch_bbox_losses = []
#     batch_cls_losses = []
    
#     for x, target in dataloader:
#         predicted_boxes, predicted_classes = model(x)
        
#         cls_loss, bbox_loss = compute_loss(predicted_boxes, predicted_classes, target)
        
#         loss = cls_loss + bbox_loss
#         loss.backward()
        
#         optimizer.step()
        
#         batch_cls_losses.append(cls_loss)  
#         batch_bbox_losses.append(bbox_loss)

#     avg_cls_loss = sum(batch_cls_losses) / len(batch_cls_losses)
#     avg_bbox_loss = sum(batch_bbox_losses) / len(batch_bbox_losses)

#     print(f"Epoch {epoch+1}/{num_epochs}")
#     print(f"Avg Bounding Box Loss: {avg_bbox_loss:.4f}")
#     print(f"Avg Classification Loss: {avg_cls_loss:.4f}")