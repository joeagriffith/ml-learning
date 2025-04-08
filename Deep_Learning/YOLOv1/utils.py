import torch
import torch.nn.functional as F


def iou(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) of bbox1s to bbox2


    Args:
        bbox1 (..., 4): A tensor containing the bounding boxes with shape (..., 4) with values [center_x, center_y, width, height]
        bbox2 (4,): A tensor containing the bounding boxes with shape (4,) with values [center_x, center_y, width, height]

    Returns:
    torch.Tensor: The IoU of box1 and box2 with shape (B, N, M).
    """
    assert bbox1.shape[-1] == 4, f"bbox1 must have last dimension of 4, got {bbox1.shape}"
    assert bbox2.shape == (4,), f"bbox2 must have shape (4,), got {bbox2.shape}"

    # [cx, cy, w, h] -> [x_min, y_min, x_max, y_max]
    bounds1 = _yolo_to_bounds(bbox1)
    bounds2 = _yolo_to_bounds(bbox2)

    bounds2 = bounds2.expand_as(bounds1)

    # Calculate the coordinates of the intersection rectangle
    inter_x_min = torch.max(bounds1[..., 0], bounds2[..., 0])
    inter_y_min = torch.max(bounds1[..., 1], bounds2[..., 1])
    inter_x_max = torch.min(bounds1[..., 2], bounds2[..., 2])
    inter_y_max = torch.min(bounds1[..., 3], bounds2[..., 3])

    # Calculate the area of the intersection rectangle
    inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * torch.clamp(inter_y_max - inter_y_min, min=0)

    # Calculate the area of both bounding boxes
    bounds1_area = (bounds1[..., 2] - bounds1[..., 0]) * (bounds1[..., 3] - bounds1[..., 1])
    bounds2_area = (bounds2[..., 2] - bounds2[..., 0]) * (bounds2[..., 3] - bounds2[..., 1])

    # Calculate the Intersection over Union (IoU)
    ious = inter_area / (bounds1_area + bounds2_area - inter_area)

    return ious

def _yolo_to_bounds(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert the YOLO format to the bounding box format.
    [x_center, y_center, width, height] -> [x_min, y_min, x_max, y_max]
    """
    cx, cy, w, h = bbox.unbind(dim=-1)
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

def pad_to_square(img: torch.Tensor) -> torch.Tensor:
    """
    Pad the image to a square.
    """
    H, W = img.shape[-2:]
    size = max(H, W)
    return F.pad(img, ((size - W) // 2, (size - W) // 2, (size - H) // 2, (size - H) // 2), "constant", 0), ((size - W) // 2, (size - H) // 2)

class PadToSquare:
    """
    Transform for letterbox padding images to a square.
    """
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        H, W = img.shape[-2:]
        size = max(H, W)
        return F.pad(img, ((size - W) // 2, (size - W) // 2, (size - H) // 2, (size - H) // 2), "constant", 0)

def coco_targets_to_yolo(annotations: list[dict], image_size: int = 448, S: int = 7, B: int = 2, C: int = 20) -> torch.Tensor:
    """
    Convert COCO targets to YOLO targets.

    Args:
        annotations (list[dict]): List of dictionaries containing the annotations for a single image.
        image_size (int): The size of the input image (e.g. 448).
        category_map (dict): A dictionary mapping COCO category IDs to class indices.

    returns: target tensor of shape (S, S, 5B + C)
    """

    target = torch.zeros((S, S, 5 * B + C), dtype=torch.float32)

    for ann in annotations:
        if ann['iscrowd']:
            continue  # Skip crowd regions

        x_min, y_min, box_w, box_h = ann['bbox']
        x_center = x_min + box_w / 2
        y_center = y_min + box_h / 2

        x_center /= image_size
        y_center /= image_size
        box_w /= image_size
        box_h /= image_size

        # Determine which grid cell the object center falls into
        grid_x = int(x_center * S)
        grid_y = int(y_center * S)

        if grid_x >= S: grid_x = S - 1
        if grid_y >= S: grid_y = S - 1

        # Local coords within the cell
        cell_x = x_center * S - grid_x
        cell_y = y_center * S - grid_y

        # Only assign if objectness == 0 (only one object per cell allowed in YOLOv1)
        if target[grid_y, grid_x, 4] == 0:
            # Assign to first bbox slot
            target[grid_y, grid_x, 0:5] = torch.tensor([cell_x, cell_y, box_w, box_h, 1.0])

            # Set class one-hot
            class_idx = category_map[ann['category_id']]
            target[grid_y, grid_x, 5*B + class_idx] = 1.0

        # If that cell already has an object, skip (YOLOv1 limitation)
        # Could add smart heuristics here for dense objects, but that’s outside YOLOv1
        else:
            continue

    return target