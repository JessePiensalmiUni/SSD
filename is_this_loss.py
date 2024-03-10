import torch.nn.functional as F
import torch
from math import sqrt
from torchvision.ops import box_iou
from sklearn.cluster import KMeans
import numpy as np

def decode_predictions(pred_locs, anchors):
    """
    Decode predicted offsets into bounding box coordinates.

    Args:
        pred_locs (Tensor): Predicted offsets (B, N, 4), where B is batch size,
                            N is number of predictions, and 4 represents
                            [Δcx, Δcy, Δw, Δh].
        anchors (Tensor): Anchor boxes (N, 4), where N is number of anchors and
                          4 represents [cx, cy, w, h].

    Returns:
        Tensor: Decoded bounding box coordinates (B, N, 4) in [xmin, ymin, xmax, ymax] format.
    """
    pred_norm = torch.sigmoid(pred_locs)
    pred_coords=pred_norm*320


    return pred_coords

from torchvision.ops import nms

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to avoid overlapping bounding boxes.

    Args:
        boxes (Tensor): Bounding box coordinates (N, 4) in [xmin, ymin, xmax, ymax] format.
        scores (Tensor): Confidence scores for each box.
        iou_threshold (float): IOU threshold for NMS.

    Returns:
        Tensor: Indices of boxes that are kept after NMS.
    """
    keep = nms(boxes, scores, iou_threshold)
    return keep

def encode_ground_truths(anchor_boxes, gt_boxes, gt_labels, iou_threshold=0.2, device="cuda"):
    """
    Encode ground truth boxes and labels relative to anchor boxes.

    Args:
        anchor_boxes (Tensor): Anchor boxes with shape (num_anchors, 4).
        gt_boxes (Tensor): Ground truth boxes with shape (num_gt_boxes, 4).
        gt_labels (Tensor): Ground truth labels with shape (num_gt_boxes,).
        iou_threshold (float): IoU threshold to match ground truth boxes with anchor boxes.

    Returns:
        Tensor, Tensor: Encoded locations (offsets and scales) and labels for anchor boxes.
    """
    num_anchors = anchor_boxes.size(0)
    iou_scores = box_iou(anchor_boxes, gt_boxes)  # Calculate IoU scores between anchors and gt_boxes

    # For each gt box, find the anchor box with the highest IoU score
    best_iou_scores, best_anchor_idx_for_gt = iou_scores.max(dim=0)
    
    # Initialize tensors to hold encoded locations and labels
    encoded_locs = torch.zeros_like(anchor_boxes)
    encoded_labels = torch.zeros((anchor_boxes.size(0),), dtype=torch.long, device=device)

    # Process only matches above the IoU threshold
    for gt_idx, anchor_idx in enumerate(best_anchor_idx_for_gt):
        if best_iou_scores[gt_idx] > iou_threshold:
            # Direct assignment for matched ground truth and anchor boxes
            encoded_locs[anchor_idx] = gt_boxes[gt_idx]

            # Assign label for the best matching anchor
            encoded_labels[anchor_idx] = gt_labels[gt_idx]

    return encoded_locs, encoded_labels
    

def normalize_boxes(gt_boxes, image_width, image_height,device="cuda"):
    normalized_boxes = []
    for box in gt_boxes:
        x_min, y_min, x_max, y_max = box
        normalized_box = [
            x_min / image_width, y_min / image_height,
            x_max / image_width, y_max / image_height
        ]
        normalized_boxes.append(normalized_box)
    return torch.tensor(normalized_boxes, dtype=torch.float32).to(device)

def parse_txt_for_aspect_ratios(txt_file):
    aspect_ratios = set()  # Use a set to avoid duplicates
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        idx = 0
        while idx < len(lines):
            idx += 1  # Skip image path
            num_bounding_boxes = int(lines[idx].strip())
            idx += 1
            if num_bounding_boxes == 0:
                    num_bounding_boxes = 1
            for n in range(num_bounding_boxes):
                box_info = lines[idx+n].strip().split(' ')
                # Calculate width and height of the bounding box
                w = float(box_info[2])
                h = float(box_info[3])
                # Avoid division by zero
                if h > 0:
                    aspect_ratio = w / h
                    if aspect_ratio>0:
                        aspect_ratios.add(aspect_ratio)
                # Adding protection against zero-height to prevent division by zero errors
            idx += num_bounding_boxes
    # Convert set to list if you need to sort or index the aspect ratios
    aspect_ratios_list = sorted(list(aspect_ratios))
    return aspect_ratios_list

def generate_anchors(feature_map_sizes, image_size, optimized_anchors):
    anchors = []
    for feature_size in feature_map_sizes:
        f_w, f_h = feature_size
        # Adjusted to calculate center positions within the image space
        step_x = image_size[0] / f_w
        step_y = image_size[1] / f_h
        
        for i in range(f_h):
            for j in range(f_w):
                # Center coordinates scaled according to the image dimensions
                cx = (j + 0.5) * step_x
                cy = (i + 0.5) * step_y

                for width, height in optimized_anchors:
                    # Width and height scaled against the entire image dimensions
                    w = width
                    h = height

                    # Calculate anchor coordinates in [x_min, y_min, x_max, y_max] format
                    x_min = cx - w / 2.0
                    y_min = cy - h / 2.0
                    x_max = cx + w / 2.0
                    y_max = cy + h / 2.0

                    anchors.append([x_min / image_size[0], y_min / image_size[1], x_max / image_size[0], y_max / image_size[1]])

    return torch.tensor(anchors, dtype=torch.float32)


def find_optimized_anchors(dataset, k=5, random_state=42):
    """
    Collects bounding box dimensions from the dataset and uses K-means clustering
    to find optimized anchor sizes.

    Args:
        dataset: A dataset object that yields images and their corresponding ground truth boxes.
        k (int): The number of clusters for K-means, representing the number of anchor sizes.
        random_state (int): Seed for K-means clustering's random number generator.

    Returns:
        numpy.ndarray: The optimized anchor box sizes as determined by K-means clustering.
    """
    widths = []
    heights = []
    for data in dataset:  # dataset should yield images and their corresponding ground truth boxes
        gt_boxes = data[1]['boxes']  # Assuming gt_boxes is in [x_min, y_min, x_max, y_max] format
        for box in gt_boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            widths.append(width)
            heights.append(height)

    # Reshape widths and heights for clustering
    data = np.array(list(zip(widths, heights))).reshape(-1, 2)

    # Run K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(data)

    # The cluster centers are your optimized anchor box dimensions
    optimized_anchors = kmeans.cluster_centers_
    
    return optimized_anchors
