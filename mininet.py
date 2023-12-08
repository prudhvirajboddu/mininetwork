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
from model import CustomObjectDetectionModel



img_size = 512

batch_size = 8

num_classes = 5

classes = ['student','Security', 'Staff', 'Facility Worker','Food Service worker']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
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
    return tuple(zip(*batch))

train_dir = 'dataset/train'
valid_dir = 'dataset/valid'


train_dataset = FaceDataset(train_dir,img_size,img_size,classes,transforms=train_transforms)

valid_dataset = FaceDataset(valid_dir,img_size,img_size,classes,transforms=valid_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
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

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)

model = CustomObjectDetectionModel(num_classes=num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

triplet_criterion = TripletLoss(margin=1.0)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 1

train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    for data in train_loader:

        images, targets = data

        print(type(images))

        images = list(image.to(device) for image in images)

        targets = {key: value.to(device) for key, value in targets.items()}
  
        prin

        outs = model(images)

        exit()


        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Split predictions for bounding box regression and classification
        pred_bbox = predictions["bounding_box"]
        pred_cls = predictions["class_label"]

        # Calculate triplet loss for bounding box regression
        loss_bbox = triplet_criterion(pred_bbox, pred_bbox, targets["bounding_box"])

        # Calculate triplet loss for classification
        loss_cls = triplet_criterion(pred_cls, pred_cls, targets["class_label"])

        # Combine losses
        total_loss = loss_bbox + loss_cls

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        num_batches = 0

        for val_batch in val_loader:
            val_images, val_targets = val_batch["image"], val_batch["target"]

            # Forward pass for validation
            val_predictions = model(val_images)

            # Split predictions for bounding box regression and classification
            val_pred_bbox = val_predictions["bounding_box"]
            val_pred_cls = val_predictions["class_label"]

            # Calculate triplet loss for bounding box regression
            val_loss_bbox = triplet_criterion(val_pred_bbox, val_pred_bbox, val_targets["bounding_box"])

            # Calculate triplet loss for classification
            val_loss_cls = triplet_criterion(val_pred_cls, val_pred_cls, val_targets["class_label"])

            # Combine losses
            total_val_loss += (val_loss_bbox + val_loss_cls).item()
            num_batches += 1

        average_val_loss = total_val_loss / num_batches
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}, Val Loss: {average_val_loss}")
