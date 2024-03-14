import torch.nn.functional as F
import torch
from math import sqrt
from torchvision.ops import box_iou
from sklearn.cluster import KMeans
import numpy as np
from math import sqrt,ceil

def decode_predictions2(pred_locs, anchors, img_width, img_height):
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
    # The predicted locations should be on the same device as anchors
    pred_locs = pred_locs.to(anchors.device)

    # Compute center coordinates of predicted boxes
    pred_cx = (pred_locs[..., 0] * anchors[..., 2]) + anchors[..., 0]
    pred_cy = (pred_locs[..., 1] * anchors[..., 3]) + anchors[..., 1]

    # Compute width and height of predicted boxes
    pred_w = anchors[..., 2] * torch.exp(pred_locs[..., 2])
    pred_h = anchors[..., 3] * torch.exp(pred_locs[..., 3])

    # Compute corner coordinates of predicted boxes
    xmin = pred_cx - (pred_w / 2)
    ymin = pred_cy - (pred_h / 2)
    xmax = pred_cx + (pred_w / 2)
    ymax = pred_cy + (pred_h / 2)

    # Concatenate and clamp the coordinates to make sure they're within the image boundaries
    decoded_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1).clamp(min=0, max=img_width)
    original_dims = torch.tensor([img_width, img_height, img_width, img_height], device=decoded_boxes.device, dtype=torch.float32)
    scaled_boxes = decoded_boxes * original_dims
    return scaled_boxes

def decode_predictions(pred_locs, img_width, img_height):
    """
    Scale predicted bounding box coordinates to the original image size.

    Args:
        pred_locs (Tensor): Predicted bounding box coordinates (B, N, 4), where B is batch size,
                            N is number of predictions, and 4 represents [xmin, ymin, xmax, ymax],
                            normalized to [0, 1] range.
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.

    Returns:
        Tensor: Scaled bounding box coordinates (B, N, 4) in [xmin, ymin, xmax, ymax] format,
                adjusted to the original image size.
    """
    # Scale coordinates to the original image dimensions
    scaled_xmin = pred_locs[..., 0] * img_width
    scaled_ymin = pred_locs[..., 1] * img_height
    scaled_xmax = pred_locs[..., 2] * img_width
    scaled_ymax = pred_locs[..., 3] * img_height

    # Stack the scaled coordinates
    scaled_boxes = torch.stack([scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax], dim=-1)

    return scaled_boxes

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

def convert_cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (xmin, ymin, xmax, ymax) format.

    Parameters:
    - boxes: A list of lists or a 2D array where each inner list or array represents a bounding box in (cx, cy, w, h) format.

    Returns:
    - A list of lists where each inner list represents a bounding box in (xmin, ymin, xmax, ymax) format.
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    # Convert to xmin, ymin, xmax, ymax
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    
    # Stack the converted coordinates
    converted_bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    
    return converted_bboxes

def encode_ground_truths2(anchor_boxes, gt_boxes, gt_labels, iou_threshold=0.2, device="cuda"):
    # Convert gt_boxes from (x, y, w, h) to (cx, cy, w, h) if necessary
    gt_boxes_conv=convert_cxcywh_to_xyxy(gt_boxes)
    anchor_boxes_conv=convert_cxcywh_to_xyxy(anchor_boxes)
    num_anchors = anchor_boxes.size(0)
    iou_scores = box_iou(anchor_boxes_conv, gt_boxes_conv)  # Calculate IoU scores between anchors and gt_boxes
    
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
    anchor_boxes_norm=anchor_boxes/320
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
    for feature_size,boxes in zip(feature_map_sizes,optimized_anchors):
        f_w, f_h = feature_size
        # Adjusted to calculate center positions within the image space
        step_x = image_size[0] / f_w
        step_y = image_size[1] / f_h
        
        for i in range(f_h):
            for j in range(f_w):
                # Center coordinates scaled according to the image dimensions
                cx = (j + 0.5) * step_x
                cy = (i + 0.5) * step_y

                for (w, h) in boxes:

                    # Calculate anchor coordinates in [x_min, y_min, x_max, y_max] format
                    x_min = cx - w / 2.0
                    y_min = cy - h / 2.0
                    x_max = cx + w / 2.0
                    y_max = cy + h / 2.0

                    anchors.append([x_min / image_size[0], y_min / image_size[1], x_max / image_size[0], y_max / image_size[1]])

    return torch.tensor(anchors, dtype=torch.float32)

def generate_anchors2(feature_map_sizes, image_size, optimized_anchors):
    #anchors are in 320x320
    anchors = []
    img_width, img_height = image_size
    anchors=[]
    for feature_size,boxes in zip(feature_map_sizes,optimized_anchors):
        f_w, f_h = feature_size
        #widths and heights of the feature boxes
        step_x = img_width / f_w
        step_y = img_height / f_h
        
        for i in range(f_h):
            for j in range(f_w):
                #center coords for each feature box
                cx = (j + 0.5) * step_x
                cy = (i + 0.5) * step_y
                
                for (w, h) in boxes:
                    # Normalize anchor dimensions
                    norm_w = w  / img_width
                    norm_h = h  / img_height

                    anchors.append([cx / img_width, cy / img_height, norm_w , norm_h ])
    all_anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
    return torch.tensor(all_anchors_tensor, dtype=torch.float32)

def find_optimized_anchors(dataset, k=[4,4,6,6,4,4,4], random_state=42):
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
    data_for_kmeans = np.array(list(zip(widths, heights)))
    anchors=[]
    for box in k:
    # Perform K-means clustering
        kmeans = KMeans(n_clusters=box, random_state=random_state).fit(data_for_kmeans)
        anchors.append(kmeans.cluster_centers_)
    
    return anchors

def find_optimized_anchors2(dataset, k=[4,4,6,6,4,4,4], random_state=42):
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
    #width and height is in 320x320
    widths = []
    heights = []
    for data in dataset:
        gt_boxes = data[1]['boxes']
        for box in gt_boxes:
            width = box[2].item()  # Convert tensor to scalar
            height = box[3].item()  # Convert tensor to scalar
            widths.append(width)
            heights.append(height)

    # Prepare data for K-means
    data_for_kmeans = np.array(list(zip(widths, heights)))
    anchors=[]
    for box in k:
    # Perform K-means clustering
        kmeans = KMeans(n_clusters=box, random_state=random_state).fit(data_for_kmeans)
        anchors.append(kmeans.cluster_centers_)

    #anchors are in 320x320
    return anchors

def create_default_boxes():
        '''
            Create 8732 default boxes in center-coordinate,
            a tensor of dimensions (8732, 4)
        '''
        fmap_wh = {"conv4_3": 38, "conv7": 19, "conv8_2": 10, "conv9_2": 5,
                   "conv10_2": 3, "conv11_2": 1}
        
        scales = {"conv4_3": 0.1, "conv7": 0.2, "conv8_2": 0.375,
                  "conv9_2": 0.55, "conv10_2": 0.725, "conv11_2": 0.9}
        
        aspect_ratios= {"conv4_3": [1., 2., 0.5], "conv7": [1., 2., 3., 0.5, 0.3333],
                        "conv8_2": [1., 2., 3., 0.5, 0.3333], 
                        "conv9_2": [1., 2., 3., 0.5, 0.3333],
                        "conv10_2": [1., 2., 0.5], "conv11_2": [1., 2., 0.5]}
        
        fmaps = list(fmap_wh.keys())
        
        default_boxes = []
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_wh[fmap]):
                for j in range(fmap_wh[fmap]):
                    cx = (j + 0.5) / fmap_wh[fmap]
                    cy = (i + 0.5) / fmap_wh[fmap]
                    
                    for ratio in aspect_ratios[fmap]:
                        default_boxes.append([cx, cy, scales[fmap]* sqrt(ratio), 
                                              scales[fmap]/sqrt(ratio)]) #(cx, cy, w, h)
                        
                        if ratio == 1:
                            try:
                                add_scale = sqrt(scales[fmap]*scales[fmaps[k+1]])
                            except IndexError:
                                #for the last feature map
                                add_scale = 1.
                            default_boxes.append([cx, cy, add_scale, add_scale])
        
        default_boxes = torch.FloatTensor(default_boxes).to("cuda") #(8732, 4)
        default_boxes.clamp_(0, 1)
        assert default_boxes.size(0) == 8732
        assert default_boxes.size(1) == 4
        return default_boxes

def soft_negative_mining(cls_loss_all, labels, alpha=1/2, beta=0.5,neg_pos_ratio_cap=1):
    """
    Soft negative mining.

    Args:
    - losses (Tensor): A tensor containing the loss of each bounding box.
    - labels (Tensor): A tensor containing the true labels of the bounding boxes (0 for background/negative).
    - alpha (float): The proportion of hard negatives to include.
    - beta (float): The threshold to include soft negatives (as a fraction of the max loss).

    Returns:
    - selected_indices (Tensor): Indices of selected negatives for backpropagation.
    """
    pos_mask = labels > 0
    neg_mask = ~pos_mask

    n_positives = pos_mask.sum().item()
    # Calculate the max number of negatives to include based on the cap
    max_negatives_allowed = ceil(n_positives * neg_pos_ratio_cap)

    # Filter losses for positive and negative examples
    cls_loss_neg = cls_loss_all[neg_mask]

    # Sort negative losses in descending order to prioritize higher losses
    sorted_neg_indices = torch.argsort(cls_loss_neg, descending=True)

    # Adjust the number of hard negatives based on initial_alpha
    # Ensure we do not exceed the maximum negatives allowed
    num_hard_negatives = min(int(alpha * n_positives), max_negatives_allowed)
    hard_negative_indices = sorted_neg_indices[:num_hard_negatives]

    # For soft negatives, instead of using initial_beta directly,
    # we dynamically adjust the number included based on remaining allowance after selecting hard negatives
    remaining_negatives_allowance = max_negatives_allowed - num_hard_negatives
    # The approach to select soft negatives can remain unchanged,
    # but ensure the number of soft negatives does not exceed the remaining allowance
    loss_threshold = beta * cls_loss_neg[hard_negative_indices[-1]]  # Use the last hard negative loss as threshold
    soft_negative_indices = sorted_neg_indices[num_hard_negatives:][cls_loss_neg[sorted_neg_indices[num_hard_negatives:]] < loss_threshold]
    soft_negative_indices = soft_negative_indices[:remaining_negatives_allowance]  # Ensure we respect the remaining allowance

    # Combine selected hard and soft negatives
    selected_negatives = torch.cat((hard_negative_indices, soft_negative_indices), dim=0)

    # Compute the final combined loss for backpropagation
    selected_negative_losses = cls_loss_neg[selected_negatives]
    cls_loss_pos = cls_loss_all[pos_mask]
    final_loss = (cls_loss_pos.sum() + selected_negative_losses.sum()) / (n_positives + len(selected_negatives))

    return final_loss