import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_BOXES = 5
NUM_CLASSES = 5

#Model Definition

class ResNet18(nn.Module):
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

class FaceRecogNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.cnn = ResNet18()  

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

model = FaceRecogNet()

b,l = torch.randn(8,5,4), torch.randn(8,5)

X = torch.randn(8,3,512,512)

pr_b,pr_c = model(X)


from torchvision.ops import box_iou

def calculate_iou(box1, box2):
    # box1 and box2 are tensors representing bounding boxes in [x1, y1, x2, y2] format
    intersection = box_iou(box1.unsqueeze(0), box2.unsqueeze(0))
    return intersection.item()

def calculate_precision_recall(targets, predictions, iou_threshold=0.5):

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_box, pred_label in zip(predictions['boxes'], predictions['labels']):
        iou_max = 0
        matching_target_index = None

        for i, (target_box, target_label) in enumerate(zip(targets['boxes'], targets['labels'])):
            iou = calculate_iou(pred_box, target_box)
            if iou > iou_threshold and iou > iou_max:
                iou_max = iou
                matching_target_index = i

        if matching_target_index is not None:
            true_positives += 1
            # Remove matching target using boolean indexing
            targets['boxes'] = torch.cat([targets['boxes'][:matching_target_index], targets['boxes'][matching_target_index+1:]])
            targets['labels'] = torch.cat([targets['labels'][:matching_target_index], targets['labels'][matching_target_index+1:]])
        else:
            false_positives += 1

    false_negatives = len(targets['boxes'])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall





# print(calculate_precision_recall)

print(pr_b.shape,pr_c.shape)

print(b.shape,l.shape)

targets = {'boxes': b , 'labels': l}

predictions = {'boxes': pr_b, 'labels': pr_c}

print(calculate_precision_recall(targets, predictions))