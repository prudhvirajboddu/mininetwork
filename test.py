import torch
from torchvision.ops import box_iou

def calculate_iou(box1, box2):
    intersection = box_iou(box1.unsqueeze(0), box2.unsqueeze(0))
    return intersection.item()

def calculate_precision_recall(targets, predictions, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for batch_idx in range(len(predictions['boxes'])):
        pred_boxes = predictions['boxes'][batch_idx]
        pred_labels = predictions['labels'][batch_idx]

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            iou_max = 0
            matching_target_index = None

            for i, (target_box, target_label) in enumerate(zip(targets['boxes'][batch_idx], targets['labels'][batch_idx])):
                iou = calculate_iou(pred_box, target_box)
                if iou > iou_threshold and iou > iou_max:
                    iou_max = iou
                    matching_target_index = i

            if matching_target_index is not None:
                true_positives += 1
                targets['boxes'][batch_idx] = torch.cat([targets['boxes'][batch_idx][:matching_target_index], targets['boxes'][batch_idx][matching_target_index+1:]])
                targets['labels'][batch_idx] = torch.cat([targets['labels'][batch_idx][:matching_target_index], targets['labels'][batch_idx][matching_target_index+1:]])
            else:
                false_positives += 1

    false_negatives = sum(len(targets['boxes'][batch_idx]) for batch_idx in range(len(targets['boxes'])))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

# Example usage for batch of output values:
batch_size = 2


# Example usage:

targets = {'boxes': torch.randn(8,5,4) , 'labels': torch.randn(8,5)}

print(targets['boxes'][0])

predictions = {'boxes': torch.randn(8,5,4) , 'labels': torch.randn(8,5)}

precision, recall = calculate_precision_recall(targets, predictions)

print(f'Precision: {precision}, Recall: {recall}')