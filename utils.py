import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted_locs, predicted_scores, target_locs, target_labels):
        """
        Compute the custom loss function.

        Args:
            predicted_locs (Tensor): Predicted bounding box offsets of shape (N, num_priors, 4).
            predicted_scores (Tensor): Predicted class scores of shape (N, num_priors, 1).
            target_locs (Tensor): Target bounding box offsets of shape (N, num_priors, 4).
            target_labels (Tensor): Target labels of shape (N, num_priors).

        Returns:
            loss (Tensor): Custom loss value.
        """
        # Smooth L1 Loss for localization
        smooth_l1_loss = F.smooth_l1_loss(predicted_locs, target_locs, reduction='sum')
        
        # Cross-Entropy Loss for confidence
        cross_entropy_loss = F.binary_cross_entropy_with_logits(predicted_scores.squeeze(dim=-1), target_labels.float(), reduction='sum')
        
        # Total loss
        loss = smooth_l1_loss + cross_entropy_loss
        return loss