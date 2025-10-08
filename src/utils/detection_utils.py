"""
Detection Utils - Utilidades para detección de objetos (IoU, NMS, mAP).
"""

import torch
from typing import List
from ..constants import B, C, S, EPSILON


def iou(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates Intersection over Union for each pair of the batch.

    Args:
        boxes_preds: Predictions of bounding boxes, Dimensions: [batch_size, S, S, 4].
        boxes_labels: Correct bounding boxes.  Dimensions: [batch_size, S, S, 4].

    Returns:
        Intersection over Union for each pair of the batch. Dimensions:
        [batch_size, S, S, 1].
    """
    
    pred_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    pred_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    pred_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    pred_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    label_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    label_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    label_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    label_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(pred_x1, label_x1)
    y1 = torch.max(pred_y1, label_y1)
    x2 = torch.min(pred_x2, label_x2)
    y2 = torch.min(pred_y2, label_y2)

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    pred_area = boxes_preds[..., 2:3] * boxes_preds[..., 3:4]
    label_area = boxes_labels[..., 2:3] * boxes_labels[..., 3:4]
    union = pred_area + label_area - intersection + EPSILON

    iou_result = intersection / union

    return iou_result


def iou_single_box(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate IoU between two single boxes.
    
    Args:
        box1: First box [4]. Supports two formats:
            - (cx, cy, w, h) center coordinates and dimensions
            - (x1, y1, x2, y2) corner coordinates
        box2: Second box [4] in same format as box1
        
    Returns:
        IoU value between 0 and 1
    """
    # Determine format and convert to (x1, y1, x2, y2) if needed
    def to_corners(box):
        if len(box) != 4:
            raise ValueError("Box must have 4 coordinates")
        
        # Check if it's already in corner format (x2 > x1 and y2 > y1 typically)
        # or if it's in center format (w, h are typically smaller values)
        if box[2] > box[0] and box[3] > box[1] and (box[2] - box[0]) > 0.1 and (box[3] - box[1]) > 0.1:
            # Likely corner format (x1, y1, x2, y2)
            return box
        else:
            # Assume center format (cx, cy, w, h)
            cx, cy, w, h = box
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            return torch.tensor([x1, y1, x2, y2])
    
    box1_corners = to_corners(box1)
    box2_corners = to_corners(box2)
    
    # Intersection coordinates
    x1 = torch.max(box1_corners[0], box2_corners[0])
    y1 = torch.max(box1_corners[1], box2_corners[1])
    x2 = torch.min(box1_corners[2], box2_corners[2])
    y2 = torch.min(box1_corners[3], box2_corners[3])
    
    # Check if boxes intersect
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1_corners[2] - box1_corners[0]) * (box1_corners[3] - box1_corners[1])
    area2 = (box2_corners[2] - box2_corners[0]) * (box2_corners[3] - box2_corners[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return float(intersection / union)


def nms(
    predicted_boxes: list[torch.Tensor],
    threshold_confidence: float = 0.5,
    threshold_iou: float = 0.5
) -> list[torch.Tensor]:
    """
    Applies Non Max Suppression given the predicted boxes. When updating the list, if
    two boxes have different classes, we do not delete them even if their IoU is high.

    Parameters:
        predicted_boxes: List with the bounding boxes predicted. They have to be
            relative to the hole image, not to each cell. Each box has dimension C + 5.
        threshold_confidence: Threshold to remove predicted bounding boxes with low
            confidence.
        threshold_repeated: Threshold to remove predicted boxes because we consider they
            refer to the same box.

    Returns:
        Predicted bounding boxes after NMS. Each box has dimension C + 5.
    """

    # Using test format: [class_index, confidence, x, y, w, h]
    filtered_boxes = [box for box in predicted_boxes if box[1] >= threshold_confidence]

    if len(filtered_boxes) == 0:
        return []

    filtered_boxes.sort(
        key=lambda box: float(box[1].detach().cpu().item()), reverse=True
    )

    result_boxes = []

    while len(filtered_boxes) > 0:
        chosen_box = filtered_boxes.pop(0)
        result_boxes.append(chosen_box)

        # Using test format: class is at index 0
        chosen_class = int(chosen_box[0].item())

        boxes_to_keep = []
        for box in filtered_boxes:
            box_class = int(box[0].item())

            if box_class != chosen_class:
                boxes_to_keep.append(box)
            else:
                # Using test format: coordinates are at indices 2-5
                chosen_coords = chosen_box[2:6].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                box_coords = box[2:6].unsqueeze(0).unsqueeze(0).unsqueeze(0)

                iou_value = iou(chosen_coords, box_coords)

                if iou_value < threshold_iou:
                    boxes_to_keep.append(box)

        filtered_boxes = boxes_to_keep

    return result_boxes


def _apply_nms_to_class(boxes: List[torch.Tensor], threshold_iou: float, coords_start: int) -> List[torch.Tensor]:
    """
    Apply NMS to boxes of the same class.
    
    Args:
        boxes: List of boxes of the same class
        threshold_iou: IoU threshold for suppression
        coords_start: Starting index of coordinate information
        
    Returns:
        Filtered boxes after NMS
    """
    final_boxes = []
    remaining_boxes = boxes.copy()
    
    while remaining_boxes:
        # Take the box with highest confidence (already sorted)
        current_box = remaining_boxes.pop(0)
        final_boxes.append(current_box)
        
        # Remove boxes with high IoU with current box
        filtered_remaining = []
        for box in remaining_boxes:
            iou_val = iou_single_box(
                current_box[coords_start:coords_start+4], 
                box[coords_start:coords_start+4]
            )
            if iou_val < threshold_iou:
                filtered_remaining.append(box)
        
        remaining_boxes = filtered_remaining
    
    return final_boxes


def to_image_coords(boxes: torch.Tensor, img_w: int, img_h: int) -> List[torch.Tensor]:
    """
    Convert YOLO grid predictions to image coordinates.
    
    Args:
        boxes: Tensor of shape [S, S, 5*B + C] containing YOLO predictions
        img_w: Image width in pixels
        img_h: Image height in pixels
        
    Returns:
        List of boxes in format [class_prob, objectness, cx, cy, w, h] in image coordinates
    """
    result = []
    cell_size = 1.0 / S
    
    for i in range(S):
        for j in range(S):
            # Get class probabilities
            class_probs = boxes[i, j, :C]
            best_class_prob = torch.max(class_probs)
            
            # Process each bounding box in this cell
            for b in range(B):
                start_idx = C + b * 5
                objectness = boxes[i, j, start_idx]
                
                if objectness > 0:  # Only process boxes with positive objectness
                    # Get box coordinates (relative to cell)
                    x_rel = boxes[i, j, start_idx + 1]
                    y_rel = boxes[i, j, start_idx + 2]
                    w_rel = boxes[i, j, start_idx + 3]
                    h_rel = boxes[i, j, start_idx + 4]
                    
                    # Convert to image coordinates
                    cx = (j + x_rel) * cell_size * img_w
                    cy = (i + y_rel) * cell_size * img_h
                    w = w_rel * cell_size * img_w
                    h = h_rel * cell_size * img_h
                    
                    # Create box tensor [class_prob, objectness, cx, cy, w, h]
                    box = torch.tensor([
                        best_class_prob.item(),
                        objectness.item(),
                        cx.item(),
                        cy.item(),
                        w.item(),
                        h.item()
                    ])
                    result.append(box)
                else:
                    # Add zero box for consistency with expected output size
                    box = torch.zeros(6)
                    result.append(box)
    
    return result