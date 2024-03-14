import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
from math import sqrt
from is_this_loss import parse_txt_for_aspect_ratios

class SSD(nn.Module):
    def __init__(self, num_classes=2, num_boxes=4,feature_map_size=(38, 38),anchors=None):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.feature_map_size = feature_map_size
        self.anchors=anchors
        # Use MobileNetV3 Large as the backbone
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])  # Remove the classifier
        feature_channels = [256, 512, 128, 256, 128, 256 ,64 , 128]
        # Additional convolutional layers for SSD
        self.loc_pred_layers = nn.ModuleList()
        self.class_pred_layers = nn.ModuleList()
        self.additional_layers = nn.ModuleList()
        
        num_anchors=[4, 6, 6, 6, 4, 4]
        feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5),(3,3),(1,1)]
        num_ancs=[]
        for i in range(0,len(num_anchors)):
            num_ancs.append(num_anchors[i]*feature_map_sizes[i][0]*feature_map_sizes[i][1])
        prev_channel_size = 960

        for channels, n_anchors in zip(feature_channels, num_ancs):
            self.additional_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_channel_size, channels, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(),
                )
            )
            self.loc_pred_layers.append(nn.Conv2d(channels, n_anchors * 4, kernel_size=3, padding=1))
            self.class_pred_layers.append(nn.Conv2d(channels, n_anchors * num_classes, kernel_size=3, padding=1))
            
            # Update the channel size for the next layer
            prev_channel_size = channels

    
    def forward(self, x):
    
        x = self.backbone(x)
        # Pass through additional layers
        locations=[]
        confidences=[]
        for i, layer in enumerate(self.additional_layers):
            x = layer(x)
            
            # Location prediction
            loc_pred = self.loc_pred_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(x.size(0), -1, 4)  # [batch_size, num_anchors * H * W, 4]
            locations.append(loc_pred)

            # Class prediction
            class_pred = self.class_pred_layers[i](x)
            class_pred = class_pred.permute(0, 2, 3, 1).contiguous()
            class_pred = class_pred.view(x.size(0), -1, self.num_classes)  # [batch_size, num_anchors * H * W, num_classes]
            confidences.append(class_pred)

        # Concatenate predictions from all layers
        locations = torch.cat(locations, dim=1)  # [batch_size, total_num_anchors, 4]
        confidences = torch.cat(confidences, dim=1)  # [batch_size, total_num_anchors, num_classes]

        return locations, confidences
    
    def postprocess(self, pred_locs, pred_scores, anchors, score_threshold=0.5, iou_threshold=0.5):
        """
        Decode predictions, apply NMS, and filter boxes by score.

        Args:
            pred_locs (Tensor): Predicted offsets for each anchor (N, 4).
            pred_scores (Tensor): Predicted class scores for each anchor (N, num_classes).
            anchors (Tensor): Anchor boxes (N, 4).
            score_threshold (float): Threshold for class scores.
            iou_threshold (float): IOU threshold for NMS.

        Returns:
            List of Tensors: Filtered bounding boxes, scores, and class labels.
        """
        device = pred_locs.device
        pred_boxes = self.decode_predictions(pred_locs, anchors).to(device)
        pred_scores = torch.softmax(pred_scores, dim=-1)

        # Filter out boxes with class score below the threshold and apply NMS
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        # Assuming the first class is background
        for class_idx in range(1, pred_scores.size(1)):
            scores = pred_scores[:, class_idx]
            mask = scores > score_threshold
            nms_boxes, nms_scores = pred_boxes[mask], scores[mask]
            nms_idx = ops.nms(nms_boxes, nms_scores, iou_threshold)

            filtered_boxes.append(nms_boxes[nms_idx])
            filtered_scores.append(nms_scores[nms_idx])
            filtered_labels.append(torch.full_like(nms_idx, class_idx, dtype=torch.int64))

        return torch.cat(filtered_boxes, dim=0), torch.cat(filtered_scores, dim=0), torch.cat(filtered_labels, dim=0)

    def decode_predictions(self, pred_locs, anchors):
        """
        Decode predicted offsets into bounding box coordinates.
        """
        # Ensure anchors is expanded to match the batch size of pred_locs
        num_batches = pred_locs.shape[0]
        anchors = anchors.unsqueeze(0).expand(num_batches, -1, -1)

        # Calculate the center coordinates (cx, cy) and the size (w, h) of the bounding boxes
        boxes_center = anchors[..., :2] + pred_locs[..., :2] * anchors[..., 2:]
        boxes_wh = anchors[..., 2:] * torch.exp(pred_locs[..., 2:])

        # Convert from center coordinates to box coordinates [xmin, ymin, xmax, ymax]
        boxes_min = boxes_center - 0.5 * boxes_wh
        boxes_max = boxes_center + 0.5 * boxes_wh
        boxes = torch.cat((boxes_min, boxes_max), dim=-1)

        return boxes