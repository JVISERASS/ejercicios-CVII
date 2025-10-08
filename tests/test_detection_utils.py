"""
Tests for detection utilities comparing with known implementations.
"""

import pytest
import torch
import torchvision.ops as ops
from tests.conftest import assert_tensors_close
from src.utils import iou, iou_single_box, nms, to_image_coords
from src.constants import B, C, S

class TestIoUSingleBox:
    """Test single box IoU calculation."""
    
    def test_iou_single_box_identical(self, device):
        """Test IoU of identical single boxes."""
        box1 = torch.tensor([0.5, 0.5, 0.4, 0.4], device=device)
        box2 = torch.tensor([0.5, 0.5, 0.4, 0.4], device=device)
        
        result = iou_single_box(box1, box2)
        
        assert abs(result - 1.0) < 1e-6
    
    def test_iou_single_box_no_overlap(self, device):
        """Test IoU of non-overlapping single boxes."""
        box1 = torch.tensor([0.2, 0.2, 0.2, 0.2], device=device)
        box2 = torch.tensor([0.8, 0.8, 0.2, 0.2], device=device)
        
        result = iou_single_box(box1, box2)
        
        assert abs(result) < 1e-6
    
    def test_iou_single_box_vs_torchvision(self, device):
        """Compare our single box IoU with torchvision ops."""
        # Create boxes in (x1, y1, x2, y2) format for torchvision
        box1_center = torch.tensor([0.5, 0.5, 0.4, 0.4], device=device)
        box2_center = torch.tensor([0.6, 0.6, 0.4, 0.4], device=device)
        
        # Convert to corner format for torchvision
        def center_to_corner(box):
            cx, cy, w, h = box
            return torch.tensor([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
        box1_corner = center_to_corner(box1_center).unsqueeze(0)
        box2_corner = center_to_corner(box2_center).unsqueeze(0)
        
        # Our implementation
        our_result = iou_single_box(box1_center, box2_center)
        
        # Torchvision implementation
        torch_result = ops.box_iou(box1_corner, box2_corner).item()
        
        assert abs(our_result - torch_result) < 1e-5


# Additional test functions from original implementation
def test_iou() -> None:
    """
    Test for the IoU function.
    """

    torch.manual_seed(42)
    batch = 4
    predictions = torch.randn(batch, S, S, 5 * B + C).clamp(0)
    targets = torch.randn(batch, S, S, 5 * B + C).clamp(0)
    atol = 1e-3

    iou_per_cell = iou(
        predictions[:, :, :, C + 1 : C + 5], targets[:, :, :, C + 1 : C + 5]
    )

    assert iou_per_cell.shape == torch.Size(
        [batch, S, S, 1]
    ), "Incorrect shape of the output."

    expected_max = torch.tensor(0.3730)
    expected_min = torch.tensor(0.0)
    expected_mean = torch.tensor(0.0079)
    expected_std = torch.tensor(0.0470)
    assert (
        torch.isclose(iou_per_cell.max(), expected_max, atol=atol)
        and torch.isclose(iou_per_cell.min(), expected_min, atol=atol)
        and torch.isclose(iou_per_cell.mean(), expected_mean, atol=atol)
        and torch.isclose(iou_per_cell.std(), expected_std, atol=atol)
    ), "Incorrect calculation of IoU."

def test_nms_original_format():
    """
    Test for the NMS function using original format.
    """
    from src.utils import nms

    # Filters by confidence - format [class, confidence, x1, y1, x2, y2]
    boxes = [
        torch.tensor([0, 0.4, 0, 0, 1, 1]),  # < threshold_confidence
        torch.tensor([0, 0.6, 0, 0, 1, 1]),  # >= threshold_confidence
    ]
    result = nms(boxes, threshold_confidence=0.5)
    assert len(result) == 1, "Incorrect filter by confidence."
    assert torch.equal(result[0], boxes[1]), "Incorrect filter by confidence."

    # Overlapping boxes - format [class, confidence, x1, y1, x2, y2]
    boxes = [
        torch.tensor([0, 0.9, 0, 0, 1, 1]),  # chosen_box 
        torch.tensor([0, 0.8, 0, 0, 1, 1]),  # overlaps -> should be removed
        torch.tensor([1, 0.7, 0, 0, 1, 1]),  # different class -> keep
    ]
    result = nms(boxes, threshold_confidence=0.5, threshold_iou=0.5)
    assert len(result) == 2, "Incorrect removing boxes."
    kept_classes = set(box[0].item() for box in result)
    assert 0 in kept_classes and 1 in kept_classes, "Incorrect removing boxes."

    # One single box
    box = torch.tensor([0, 0.9, 0, 0, 1, 1])
    result = nms([box])
    assert len(result) == 1, "Incorrect for one box."
    assert torch.equal(result[0], box), "Incorrect for one box."


def test_to_image_coords():
    """
    Test for the function that transforms coordinates.
    """
    from src.constants import B, C, S
    from src.utils import to_image_coords

    img_w, img_h = 224, 224
    boxes = torch.zeros((S, S, 5 * B + C))
    i, j = 3, 4  # selected cell
    # Fill the cell with some values
    boxes[i, j, 0] = 0.8  # class probability
    boxes[i, j, C] = 0.9  # objectness first box
    boxes[i, j, C + 1] = 0.5  # x relative
    boxes[i, j, C + 2] = 0.5  # y relative  
    boxes[i, j, C + 3] = 0.2  # w relative
    boxes[i, j, C + 4] = 0.3  # h relative
    
    boxes[i, j, C + 5] = 0.6  # objectness second box
    boxes[i, j, C + 6] = 0.1  # x relative
    boxes[i, j, C + 7] = 0.2  # y relative
    boxes[i, j, C + 8] = 0.1  # w relative
    boxes[i, j, C + 9] = 0.1  # h relative

    result = to_image_coords(boxes, img_w, img_h)

    assert len(result) == S * S * B, "Incorrect shape."
    non_zero_boxes = [b for b in result if b[1] > 0]
    assert len(non_zero_boxes) == 2, "Incorrect shape."

    # Test first non-zero box
    box = non_zero_boxes[0]
    # Class prob and p(obj)
    assert torch.isclose(box[0], torch.tensor(0.8)), "Incorrect class."
    assert torch.isclose(box[1], torch.tensor(0.9)), "Incorrect p(obj)."
    
    cell_size = 1.0 / S
    expected_cx = (j + 0.5) * cell_size * img_w
    expected_cy = (i + 0.5) * cell_size * img_h
    expected_w = 0.2 * cell_size * img_w
    expected_h = 0.3 * cell_size * img_h
    
    # Coordinates
    assert torch.isclose(
        box[2], torch.tensor(expected_cx)
    ), "Incorrect coordinates."
    assert torch.isclose(
        box[3], torch.tensor(expected_cy)
    ), "Incorrect coordinates."
    assert torch.isclose(box[4], torch.tensor(expected_w)), "Incorrect coordinates."
    assert torch.isclose(box[5], torch.tensor(expected_h)), "Incorrect coordinates."