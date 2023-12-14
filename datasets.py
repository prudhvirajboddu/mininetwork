import os
import cv2
import glob
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as et
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import nms, box_iou

img_size = 512

batch_size = 8

NUM_BOXES = 5
NUM_CLASSES = 5

classes = ['student','Security', 'Staff', 'Facility Worker','Food Service worker']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.num_boxes = 5
        self.num_classes = 5
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.image_paths += glob.glob(f"{self.dir_path}/*.png")
        self.all_images = [image_path.split(
            '/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # capture the corresponding XML file for getting the annotations
        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))

            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        boxes,labels = self.map_to_model_output(boxes, labels , self.num_boxes)

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["bounding_box"] = boxes
        target["class_label"] = labels

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image=image_resized,
                                     bboxes=target['bounding_box'],
                                     labels=labels)
            image_resized = sample['image']
            target['bounding_box'] = torch.Tensor(sample['bboxes'])

        return image_resized, target
    
    def map_to_model_output(self, boxes, classes, num_boxes):
        # Pad the lists to have a fixed number of boxes
        boxes_padded = self.pad_list_to_length(boxes, max_len=num_boxes, pad_value=[0, 0, 512, 512])
        classes_padded = self.pad_list_to_length(classes, max_len=num_boxes, pad_value=-1)
        return boxes_padded, classes_padded

    def pad_list_to_length(self, list_to_pad, max_len, pad_value):
        list_len = len(list_to_pad)
        if list_len >= max_len:
            return list_to_pad[:max_len]
        else:
            # Calculate padding required
            padding = [pad_value] * (max_len - list_len)

            # Pad the list
            padded_list = list_to_pad + padding

            return padded_list

    def __len__(self):
        return len(self.all_images)

train_transforms = A.Compose([
    A.Flip(0.5),
    A.RandomRotate90(0.5),
    A.MotionBlur(p=0.2),
    A.MedianBlur(blur_limit=3, p=0.1),
    A.Blur(blur_limit=3, p=0.1),
    ToTensorV2(p=1.0),
], bbox_params={
    'format': 'pascal_voc',
    'label_fields': ['labels']
})

valid_transforms = A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc','label_fields': ['labels']})


def collate_fn(batch):
    images,targets = zip(*batch)

    bounding_boxes = [target['bounding_box'] for target in targets]
    class_labels = [target['class_label'] for target in targets]

    return images,bounding_boxes,class_labels

train_dir = 'dataset/train'
valid_dir = 'dataset/valid'

train_dataset = FaceDataset(train_dir,img_size,img_size,classes,transforms=train_transforms)

valid_dataset = FaceDataset(valid_dir,img_size,img_size,classes,transforms=valid_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

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

model = FaceRecogNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


def triplet_loss(predicted_boxes, predicted_classes, bounding_boxes, class_labels, alpha=0.2, num_negative_samples=5):
    # Calculate smooth L1 loss for bounding boxes
    loss_bbox = F.smooth_l1_loss(predicted_boxes, bounding_boxes)

    # Apply softmax activation to predicted classes
    predicted_classes = F.softmax(predicted_classes, dim=1)

    # Convert ground truth labels to torch.long
    class_labels = class_labels.long()

    # Calculate triplet loss
    # Get indices for each class
    class_indices = [class_labels == i for i in range(predicted_classes.size(1))]

    # Initialize triplet loss
    triplet_loss = 0

    for i in range(predicted_classes.size(1)):
        # Skip classes without positive samples
        if torch.sum(class_indices[i]) == 0:
            continue

        # Get positive indices for the current class
        positive_indices = class_indices[i].nonzero()[:, 0]

        # Get negative indices for the current class
        negative_indices = torch.cat([class_indices[j].nonzero()[:, 0] for j in range(predicted_classes.size(1)) if j != i])

        # Randomly sample num_negative_samples indices from the negative indices
        negative_indices = torch.randperm(negative_indices.size(0))[:num_negative_samples]

        # Select positive and negative samples
        positive_samples = predicted_boxes[positive_indices]
        negative_samples = predicted_boxes[negative_indices]

        # Check if positive and negative samples have the same size along dimension 0
        if positive_samples.size(0) != negative_samples.size(0):
            continue

        # Calculate pairwise distances
        pairwise_distances = torch.cdist(positive_samples, negative_samples, p=2)

        # Calculate triplet loss for the current class
        triplet_loss += F.relu(pairwise_distances - alpha).mean()

    # Average over classes
        
    print(class_indices)

    print(type(class_indices))
    print(torch.sum(class_indices).float())

    # exit()
    triplet_loss /= torch.sum(class_indices).float()

    # Combine all losses
    total_loss = loss_bbox + triplet_loss

    return total_loss




num_epochs = 1

train_losses = []
valid_losses = []

for epoch in range(num_epochs):

    for images, bounding_boxes, class_labels in train_loader:

        images = torch.stack(images).to(device)
        bounding_boxes = torch.stack(bounding_boxes).to(device)
        class_labels = torch.stack(class_labels).to(device)

        optimizer.zero_grad()

        predicted_boxes, predicted_classes = model(images)

        # Calculate loss
        loss = triplet_loss(predicted_boxes, predicted_classes, bounding_boxes, class_labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        train_losses.append(loss.item())

    print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}")

    # Validation
    with torch.no_grad():
        for images, bounding_boxes, class_labels in valid_loader:

            images = torch.stack(images).to(device)
            bounding_boxes = torch.stack(bounding_boxes).to(device)
            class_labels = torch.stack(class_labels).to(device)

            predicted_boxes, predicted_classes = model(images)

            # Calculate loss
            loss = triplet_loss(predicted_boxes, predicted_classes, bounding_boxes, class_labels)

            valid_losses.append(loss.item())

    print(f"Epoch: {epoch+1}/{num_epochs}, Validation Loss: {loss.item():.4f}")