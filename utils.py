import os
from typing import Literal

import numpy as np
from skimage import segmentation, color
from skimage.future import graph
from skimage import io
from skimage.feature import selective_search 
ss = selective_search

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.constants import EPSILON, TARGET_SIZE_IMG, B, C, S


def saliency_map(
    model: nn.Module, x: torch.Tensor, target_class: int | None
) -> torch.Tensor:
    """
    Computes gradients of the model output with respect to input. As we want to return a
    saliency map, we must obtain the maximum of the absolute values along the channels
    dimension.

    Args:
        model: Model we want to explain.
        x: Input tensor. Dimensions: [batch, channels, height, width].
        target_class: Class for which backward pass is computed. If None, then we assume
            it is the one with maximum score.

    Returns:
        Saliency map of the model output with respect to input. Dimensions: [batch,
        height, width].

    Raises:
        RuntimeError: If the gradients calculated with the backward are None.
    """

    # TODO
    x.requires_grad_()
    model.zero_grad()
    output = model(x)
    if target_class is None:
        target_class = output.argmax(dim=1)
    loss = output[range(x.shape[0]), target_class].sum()
    loss.backward()
    saliency = x.grad.abs().max(dim=1)[0]
    if saliency is None:
        raise RuntimeError("Gradients are None")
    return saliency


def normalize_tensor(explanation: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the explanation tensor.

    Args:
        explanation: Explanation tensor with dimensions [batch, height, width].

    Returns:
        Normalized explanation with the same dimensions.
    """

    max_ = torch.amax(explanation, dim=(1, 2), keepdim=True)
    min_ = torch.amin(explanation, dim=(1, 2), keepdim=True)

    return (explanation - min_) / (max_ - min_ + 1e-8)


def get_final_layer(model: nn.Module) -> tuple[str, nn.Linear]:
    """
    Retrieves the name and the final fully-connected layer of a PyTorch model.
    This iterates backwards through all named modules
    to find the *last nn.Linear layer*, ignoring final activation or dropout layers.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        tuple: A tuple containing the layer's name (e.g., 'fc' or 'classifier.6') and the layer module itself.

    Raises:
        ValueError: If no nn.Linear layer is found in the model.
    """
    # TODO
    # Hint: Use model.named_modules() to iterate through all modules
    # Iterate backwards through all modules to find the last Linear layer

def iou(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates Intersection over Union for each pair of the batch.

    Args:
        boxes_preds: Predictions of bounding boxes, Dimensions: [batch_size, S, S, 4].
        boxes_labels: Correct bounding boxes.  Dimensions: [batch_size, S, S, 4].

    Returns:
        Intersection over Union for each pair of the batch. Dimensions:
        [batch_size, S, S, 1].
    """

    # TODO

    # Vectorized IoU computation for [batch_size, S, S, 4] tensors
    boxes_preds_xy = torch.empty_like(boxes_preds)
    boxes_labels_xy = torch.empty_like(boxes_labels)

    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes_preds_xy[..., 0] = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    boxes_preds_xy[..., 1] = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    boxes_preds_xy[..., 2] = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    boxes_preds_xy[..., 3] = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

    boxes_labels_xy[..., 0] = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    boxes_labels_xy[..., 1] = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    boxes_labels_xy[..., 2] = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    boxes_labels_xy[..., 3] = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # Intersection
    x1 = torch.max(boxes_preds_xy[..., 0], boxes_labels_xy[..., 0])
    y1 = torch.max(boxes_preds_xy[..., 1], boxes_labels_xy[..., 1])
    x2 = torch.min(boxes_preds_xy[..., 2], boxes_labels_xy[..., 2])
    y2 = torch.min(boxes_preds_xy[..., 3], boxes_labels_xy[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    intersection = inter_w * inter_h

    # Union
    area_pred = (boxes_preds_xy[..., 2] - boxes_preds_xy[..., 0]).clamp(min=0) * (boxes_preds_xy[..., 3] - boxes_preds_xy[..., 1]).clamp(min=0)
    area_label = (boxes_labels_xy[..., 2] - boxes_labels_xy[..., 0]).clamp(min=0) * (boxes_labels_xy[..., 3] - boxes_labels_xy[..., 1]).clamp(min=0)
    union = area_pred + area_label - intersection

    iou_result = (intersection / (union + EPSILON)).unsqueeze(-1)
    return iou_result


def _iou_single_box(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate IoU between two single boxes.
    
    Args:
        box1: First box [x, y, w, h] (center format)
        box2: Second box [x, y, w, h] (center format)
    
    Returns:
        IoU value as float
    """

    box1_c = box1[:2] - box1[2:] / 2
    box1_c2 = box1[:2] + box1[2:] / 2
    box2_c = box2[:2] - box2[2:] / 2
    box2_c2 = box2[:2] + box2[2:] / 2

    x1 = max(box1_c[0], box2_c[0])
    y1 = max(box1_c[1], box2_c[1])
    x2 = min(box1_c2[0], box2_c2[0])
    y2 = min(box1_c2[1], box2_c2[1])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1_c2[0] - box1_c[0]) * (box1_c2[1] - box1_c[1])
    area2 = (box2_c2[0] - box2_c[0]) * (box2_c2[1] - box2_c[1])
    return inter / (area1 + area2 - inter + EPSILON)


def nms(
    predicted_boxes: list[torch.Tensor],
    threshold_confidence: float = 0.5,
    threshold_repeated: float = 0.5,
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

    # TODO

    filtered_boxes = [box for box in predicted_boxes if box[1] >= threshold_confidence]
    if not filtered_boxes:
        return []

    filtered_boxes.sort(key=lambda x: x[1], reverse=True)
    final_boxes = []

    while filtered_boxes:
        curr = filtered_boxes.pop(0)
        final_boxes.append(curr)
        curr_cls = curr[0]
        curr_box = curr[2:6]
        # Only compare boxes of the same class
        filtered_boxes = [
            box for box in filtered_boxes
            if box[0] != curr_cls or _iou_single_box(curr_box, box[2:6]) < threshold_repeated
        ]

    return final_boxes


def to_image_coords(boxes: torch.Tensor, img_w: int, img_h: int) -> list[torch.Tensor]:
    """
    Transforms the coordinates of the bounding boxes from relative to each cell to
    relative to the hole image.

    Args:
        boxes: Predicted bounding boxes. Dimensions: [S, S, 5 * B + C].
        img_w: Width of the image.
        img_h: Height of the image.

    Returns:
        List with the scaled bounding boxes to the hole image. Each tensor has dimension
        C + 5.
    """

    # TODO
    
    cell_size = 1.0 / S
    boxes_list = [
        torch.tensor([
            torch.max(boxes[i, j, :C]).item(),
            boxes[i, j, C + b * 5].item(),
            (j + boxes[i, j, C + b * 5 + 1].item()) * cell_size * img_w,
            (i + boxes[i, j, C + b * 5 + 2].item()) * cell_size * img_h,
            boxes[i, j, C + b * 5 + 3].item() * cell_size * img_w,
            boxes[i, j, C + b * 5 + 4].item() * cell_size * img_h
        ])
        for i in range(S)
        for j in range(S)
        for b in range(B)
    ]
    return boxes_list

# Region Of Interest (ROI) Pooling
def roi_pooling(feature_map: torch.Tensor, rois: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Applies ROI pooling to the feature map given regions of interest.
    Args:
        feature_map: Tensor of shape [N, C, H, W]
        rois: Tensor of shape [num_rois, 5] (batch_idx, x1, y1, x2, y2)
        output_size: (h, w) output size
    Returns:
        Tensor of shape [num_rois, C, h, w]
    """
    # Try to use optimized torchvision ops if available (prefer roi_align)
    N, C, H, W = feature_map.shape
    out_h, out_w = int(output_size[0]), int(output_size[1])

    # Empty rois -> return empty tensor with correct shape
    if rois.numel() == 0:
        return torch.zeros((0, C, out_h, out_w), device=feature_map.device, dtype=feature_map.dtype)

    # Ensure rois on same device
    rois = rois.to(device=feature_map.device)

    # If rois provided as [num_rois, 4] (no batch idx), assume batch 0
    if rois.dim() == 2 and rois.shape[1] == 4:
        batch_idxs = torch.zeros((rois.shape[0], 1), device=rois.device)
        rois = torch.cat([batch_idxs, rois.float()], dim=1)

    # Cast to float for torchvision ops if needed
    rois_float = rois.float()

    try:
        # Prefer roi_align (better interpolation), fallback to roi_pool
        if hasattr(torchvision.ops, "roi_align"):
            # torchvision expects boxes as [num_rois, 5] with floats
            return torchvision.ops.roi_align(feature_map, rois_float, (out_h, out_w), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        elif hasattr(torchvision.ops, "roi_pool"):
            return torchvision.ops.roi_pool(feature_map, rois_float, (out_h, out_w))
    except Exception:
        # If torchvision op failed for any reason, fallback to robust manual implementation
        pass

    # Manual robust fallback implementation (handles float coords, clamps, empty ROIs)
    num_rois = rois_float.shape[0]
    pooled = torch.zeros(num_rois, C, out_h, out_w, device=feature_map.device, dtype=feature_map.dtype)

    for i in range(num_rois):
        batch_idx = int(rois_float[i, 0].item())
        # Read coordinates as floats to support subpixel boxes
        x1f, y1f, x2f, y2f = [float(v) for v in rois_float[i, 1:5].tolist()]

        # Clamp coordinates to feature map bounds
        x1f = max(0.0, min(x1f, W))
        x2f = max(0.0, min(x2f, W))
        y1f = max(0.0, min(y1f, H))
        y2f = max(0.0, min(y2f, H))

        # If ROI degenerate after clamping, leave pooled zeros
        if x2f <= x1f or y2f <= y1f:
            continue

        roi_w = x2f - x1f
        roi_h = y2f - y1f

        # Compute bin edges in image coordinates (float), then map to integer pixel indices
        h_edges = torch.linspace(0.0, roi_h, out_h + 1, device=feature_map.device)
        w_edges = torch.linspace(0.0, roi_w, out_w + 1, device=feature_map.device)

        for ph in range(out_h):
            for pw in range(out_w):
                ys_f = y1f + h_edges[ph].item()
                ye_f = y1f + h_edges[ph + 1].item()
                xs_f = x1f + w_edges[pw].item()
                xe_f = x1f + w_edges[pw + 1].item()

                ys = int(max(0, min(H, int(torch.floor(torch.tensor(ys_f))))))
                ye = int(max(0, min(H, int(torch.ceil(torch.tensor(ye_f))))))
                xs = int(max(0, min(W, int(torch.floor(torch.tensor(xs_f))))))
                xe = int(max(0, min(W, int(torch.ceil(torch.tensor(xe_f))))))

                # Ensure at least one pixel if possible
                if ye <= ys:
                    if ys < H:
                        ye = min(ys + 1, H)
                    else:
                        ye = ys
                if xe <= xs:
                    if xs < W:
                        xe = min(xs + 1, W)
                    else:
                        xe = xs

                if ye <= ys or xe <= xs:
                    # still empty: keep zeros
                    continue

                region = feature_map[batch_idx, :, ys:ye, xs:xe]
                if region.numel() == 0:
                    continue
                pooled[i, :, ph, pw] = region.amax(dim=(-1, -2))

    return pooled

# Selective Search (wrapper, requires skimage)
def selective_search(img: torch.Tensor) -> list:
    """
    Performs selective search on an image.
    Args:
        img: Tensor [H, W, 3] or [3, H, W]
    Returns:
        List of region proposals (x, y, w, h)
    """
    img_np = img.permute(1,2,0).cpu().numpy() if img.shape[0]==3 else img.cpu().numpy()
    _, regions = ss(img_np, scale=500, sigma=0.9, min_size=10) # ss = selective search
    return [r['rect'] for r in regions]

# Cálculo de mAP (mean Average Precision)
def calculate_map(preds, targets) -> float:
    """
    Calculates mean Average Precision (mAP) for detection.
    Args:
        preds: List of predicted boxes and labels
        targets: List of ground truth boxes and labels
    Returns:
        mAP value
    """
    # Manual mAP calculation (simplified, for one class)
    # preds: list of dicts with 'boxes', 'scores', 'labels'
    # targets: list of dicts with 'boxes', 'labels'
    iou_threshold = 0.5
    aps = []
    num_classes = max([t['labels'].max().item() if len(t['labels']) > 0 else 0 for t in targets] + [p['labels'].max().item() if len(p['labels']) > 0 else 0 for p in preds]) + 1
    for cls in range(num_classes):
        all_scores = []
        all_matches = []
        n_gt = 0
        for pred, target in zip(preds, targets):
            pred_mask = (pred['labels'] == cls)
            target_mask = (target['labels'] == cls)
            pred_boxes = pred['boxes'][pred_mask]
            scores = pred['scores'][pred_mask] if 'scores' in pred else torch.ones(len(pred_boxes))
            target_boxes = target['boxes'][target_mask]
            n_gt += len(target_boxes)
            matched = torch.zeros(len(target_boxes), dtype=torch.bool)
            for i, box in enumerate(pred_boxes):
                ious = torch.zeros(len(target_boxes))
                for j, gt_box in enumerate(target_boxes):
                    ious[j] = _iou_single_box(box, gt_box)
                max_iou, idx = (ious.max(0) if len(ious) > 0 else (torch.tensor(0.), torch.tensor(-1)))
                if max_iou > iou_threshold and idx >= 0 and not matched[idx]:
                    all_matches.append(1)
                    matched[idx] = True
                else:
                    all_matches.append(0)
                all_scores.append(scores[i].item())
        if len(all_scores) == 0:
            aps.append(0.0)
            continue
        # Sort by score
        sorted_idx = np.argsort(-np.array(all_scores))
        tp = np.array(all_matches)[sorted_idx]
        fp = 1 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (n_gt + EPSILON)
        precisions = tp_cum / (tp_cum + fp_cum + EPSILON)
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t].max() if np.any(recalls >= t) else 0
            ap += p / 11
        aps.append(ap)
    return float(np.mean(aps))

# Bed of Nails
def bed_of_nails(x: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Bed of nails upsampling (nearest neighbor with zeros).
    Args:
        x: Tensor [N, C, H, W]
        stride: Upsampling factor
    Returns:
        Upsampled tensor
    """
    N, C, H, W = x.shape
    out = torch.zeros(N, C, H*stride, W*stride, device=x.device, dtype=x.dtype)
    out[:,:,::stride,::stride] = x
    return out

# Nearest Neighbor Interpolation
def nearest_neighbor(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Nearest neighbor interpolation.
    Args:
        x: Tensor [N, C, H, W]
        size: (H_out, W_out)
    Returns:
        Upsampled tensor
    """
    # Manual nearest neighbor interpolation (no F.interpolate)
    N, C, H, W = x.shape
    H_out, W_out = size
    # Compute the nearest source indices for each output pixel
    idx_y = torch.floor(torch.linspace(0, H - 1, H_out, device=x.device)).long()
    idx_x = torch.floor(torch.linspace(0, W - 1, W_out, device=x.device)).long()
    out = torch.zeros(N, C, H_out, W_out, device=x.device, dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            out[:, :, i, j] = x[:, :, idx_y[i], idx_x[j]]
    return out
    # return F.interpolate(x, size=size, mode='nearest')

# Bilinear Interpolation
def bilinear_interpolation(x: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Bilinear interpolation.
    Args:
        x: Tensor [N, C, H, W]
        size: (H_out, W_out)
    Returns:
        Upsampled tensor
    """
    # Manual bilinear interpolation (no F.interpolate)
    N, C, H, W = x.shape
    H_out, W_out = size
    # Create normalized grid coordinates
    grid_y = torch.linspace(0, H - 1, H_out, device=x.device)
    grid_x = torch.linspace(0, W - 1, W_out, device=x.device)
    y0 = torch.floor(grid_y).long()
    x0 = torch.floor(grid_x).long()
    y1 = torch.clamp(y0 + 1, max=H - 1)
    x1 = torch.clamp(x0 + 1, max=W - 1)
    wy = grid_y - y0.float()
    wx = grid_x - x0.float()

    out = torch.zeros(N, C, H_out, W_out, device=x.device, dtype=x.dtype)
    for i in range(H_out):
        for j in range(W_out):
            y0i, y1i = y0[i], y1[i]
            x0j, x1j = x0[j], x1[j]
            wy_i = wy[i]
            wx_j = wx[j]
            v00 = x[:, :, y0i, x0j]
            v01 = x[:, :, y0i, x1j]
            v10 = x[:, :, y1i, x0j]
            v11 = x[:, :, y1i, x1j]
            out[:, :, i, j] = (
                (1 - wy_i) * (1 - wx_j) * v00 +
                (1 - wy_i) * wx_j * v01 +
                wy_i * (1 - wx_j) * v10 +
                wy_i * wx_j * v11
            )
    return out
    # return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# Max Unpooling
def max_unpooling(x: torch.Tensor, indices: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Max unpooling operation.
    Args:
        x: Pooled tensor
        indices: Indices from max pooling
        kernel_size: Pooling kernel size
        stride: Pooling stride
    Returns:
        Unpooled tensor
    """
    # unpool = nn.MaxUnpool2d(kernel_size, stride)
    # Manual implementation of Max Unpooling
    # x: [N, C, H, W], indices: [N, C, H, W]
    N, C, H, W = x.shape
    out_H = H * stride
    out_W = W * stride
    out = torch.zeros(N, C, out_H, out_W, device=x.device, dtype=x.dtype)
    # Flatten for easier indexing
    x_flat = x.view(N, C, -1)
    indices_flat = indices.view(N, C, -1)
    out_flat = out.view(N, C, -1)
    for n in range(N):
        for c in range(C):
            out_flat[n, c].scatter_(0, indices_flat[n, c], x_flat[n, c])
    out = out_flat.view(N, C, out_H, out_W)
    return out(x, indices)

# Mapa de profundidad (Depth Map)
def depth_map(disparity: torch.Tensor, focal_length: float, baseline: float) -> torch.Tensor:
    """
    Computes depth map from disparity.
    Args:
        disparity: Disparity tensor [H, W]
        focal_length: Camera focal length
        baseline: Distance between cameras
    Returns:
        Depth map tensor [H, W]
    """
    return (focal_length * baseline) / (disparity + EPSILON)

# Normal de superficie
def surface_normal(depth: torch.Tensor) -> torch.Tensor:
    """
    Estimates surface normals from depth map.
    Args:
        depth: Depth map [H, W]
    Returns:
        Normals [H, W, 3]
    """
    dzdx = torch.gradient(depth, axis=1)[0]
    dzdy = torch.gradient(depth, axis=0)[0]
    normal = torch.stack([-dzdx, -dzdy, torch.ones_like(depth)], dim=-1)
    normal = F.normalize(normal, dim=-1)
    return normal

# Funciones implícitas (Ejemplo: SDF)
def implicit_function_sphere(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Signed Distance Function for a sphere.
    Args:
        x: Points [N, 3]
        center: Center [3]
        radius: Radius
    Returns:
        SDF values [N]
    """
    return torch.norm(x - center, dim=1) - radius

# Voxels
def points_to_voxel(points: torch.Tensor, grid_size: tuple, bounds: tuple) -> torch.Tensor:
    """
    Converts point cloud to voxel grid.
    Args:
        points: [N, 3]
        grid_size: (X, Y, Z)
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    Returns:
        Voxel grid [X, Y, Z]
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]
    idx = ((points - torch.tensor([x_min, y_min, z_min], device=points.device)) /
           torch.tensor([x_max-x_min, y_max-y_min, z_max-z_min], device=points.device) *
           torch.tensor(grid_size, device=points.device)).long()
    idx = torch.clamp(idx, min=0, max=torch.tensor(grid_size, device=points.device)-1)
    voxels = torch.zeros(grid_size, dtype=torch.bool, device=points.device)
    voxels[idx[:,0], idx[:,1], idx[:,2]] = True
    return voxels

# Nube de puntos
def point_cloud_from_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Converts depth map to point cloud.
    Args:
        depth: [H, W]
        intrinsics: [3, 3] camera matrix
    Returns:
        Points [N, 3]
    """
    H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    x = x.flatten().float()
    y = y.flatten().float()
    z = depth.flatten()
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    cx, cy = intrinsics[0,2], intrinsics[1,2]
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    points = torch.stack([X, Y, z], dim=1)
    return points

# Distancia Chamfer
def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> float:
    """
    Computes Chamfer distance between two point clouds.
    Args:
        pc1: [N1, 3]
        pc2: [N2, 3]
    Returns:
        Chamfer distance
    """
    dist1 = torch.cdist(pc1, pc2).min(dim=1)[0]
    dist2 = torch.cdist(pc2, pc1).min(dim=1)[0]
    return dist1.mean().item() + dist2.mean().item()

# F1 Score
def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes F1 score for binary classification.
    Args:
        preds: [N] predicted labels
        targets: [N] true labels
    Returns:
        F1 score
    """
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    return 2 * precision * recall / (precision + recall + EPSILON)

# Capa de fusión
class FusionLayer(nn.Module):
    """
    Simple fusion layer that concatenates and projects features.
    """
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels1 + in_channels2, out_channels)
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        return self.fc(x)

# Early Fusion
def early_fusion(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Early fusion by concatenation.
    Args:
        x1, x2: [N, D]
    Returns:
        [N, D1+D2]
    """
    return torch.cat([x1, x2], dim=-1)

# Late Fusion
def late_fusion(pred1: torch.Tensor, pred2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Late fusion by weighted sum.
    Args:
        pred1, pred2: [N, C]
        alpha: weight for pred1
    Returns:
        [N, C]
    """
    return alpha * pred1 + (1 - alpha) * pred2

# Flujo óptico (Optical Flow)
def optical_flow(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Dummy optical flow estimation (placeholder).
    Args:
        img1, img2: [1, 3, H, W]
    Returns:
        Flow field [1, 2, H, W]
    """
    # For real use, integrate RAFT, PWC-Net, etc.
    return torch.zeros(img1.shape[0], 2, img1.shape[2], img1.shape[3], device=img1.device)

# Red de 2 flujos (Two-Stream Network)
class TwoStreamNet(nn.Module):
    def __init__(self, backbone1, backbone2, fusion_layer):
        super().__init__()
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.fusion = fusion_layer
    def forward(self, x1, x2):
        f1 = self.backbone1(x1)
        f2 = self.backbone2(x2)
        return self.fusion(f1, f2)

# Inflated CNNs (I3D)
class InflatedConv3d(nn.Module):
    """
    Inflated 2D Conv to 3D Conv (I3D style).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        return self.conv3d(x)

# SlowFast Networks (sLOWfAST)
class SlowFastNet(nn.Module):
    """
    Simplified SlowFast network skeleton.
    """
    def __init__(self, slow_path, fast_path, fusion_layer):
        super().__init__()
        self.slow_path = slow_path
        self.fast_path = fast_path
        self.fusion = fusion_layer
    def forward(self, x_slow, x_fast):
        f_slow = self.slow_path(x_slow)
        f_fast = self.fast_path(x_fast)
        return self.fusion(f_slow, f_fast)
