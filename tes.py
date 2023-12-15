import torch
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

# Example usage for batch of output values:
batch_size = 2

targets_batch = {'boxes': torch.randn(8,5,4),
                 'labels': torch.randn(8,5)}

predictions_batch = {'boxes': torch.randn(8,5,4),
                     'labels': torch.randn(8,5)}

precision, recall = calculate_precision_recall(targets_batch, predictions_batch)
print(f'Precision: {precision}, Recall: {recall}')
