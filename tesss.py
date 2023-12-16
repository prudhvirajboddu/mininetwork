from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2

from torchsummary import summary  


model = fasterrcnn_resnet50_fpn_v2(pretrained=False)

print(model)
