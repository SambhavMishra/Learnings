import torch

def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculates the Intersection over Union (IoU) score between two sets of bounding boxes.

    Args:
        boxes_preds (torch.Tensor): Predicted bounding boxes in the format (x1, y1, x2, y2)
        boxes_labels (torch.Tensor): Ground truth bounding boxes in the format (x1, y1, x2, y2)

    Returns:
        torch.Tensor: The IoU score between the two sets of boxes.
    """
    # Get the coordinates of the intersection rectangle
    x1 = torch.max(boxes_preds[:, 0], boxes_labels[:, 0])
    y1 = torch.max(boxes_preds[:, 1], boxes_labels[:, 1])
    x2 = torch.min(boxes_preds[:, 2], boxes_labels[:, 2])
    y2 = torch.min(boxes_preds[:, 3], boxes_labels[:, 3])

    # Calculate the area of intersection rectangle
    intersection = torch.clamp((x2 - x1 + 1), min=0) * torch.clamp((y2 - y1 + 1), min=0)

    # Calculate the area of both bounding boxes
    box1_area = (boxes_preds[:, 2] - boxes_preds[:, 0] + 1) * (boxes_preds[:, 3] - boxes_preds[:, 1] + 1)
    box2_area = (boxes_labels[:, 2] - boxes_labels[:, 0] + 1) * (boxes_labels[:, 3] - boxes_labels[:, 1] + 1)

    # Calculate the intersection over union
    iou = intersection / (box1_area + box2_area - intersection)

    return iou
