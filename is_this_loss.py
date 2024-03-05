import torch.nn.functional as F
import torch

def bbox_regression_loss(predictions, targets):
    # Compute smooth L1 loss (Huber loss) for bounding box regression
    # predictions: (batch_size, num_boxes, 4, 1)
    # targets: Ground truth bounding box coordinates (list of dictionaries)
    loss = 0
    batch_size = predictions.size(0)
    num_boxes = predictions.size(1)
    for batch_idx in range(batch_size):
        pred_boxes = predictions[batch_idx]  # Predicted bounding box offsets
        target_boxes = targets[batch_idx]['boxes']  # Ground truth bounding box coordinates
        
        # Iterate over pairs of predicted and target bounding boxes
        for target_box in target_boxes:
            # Initialize the minimum loss
            min_loss = float('inf')
            
            # Iterate over predicted bounding boxes
            for pred_box in pred_boxes:
                # Expand dimensions of target_box to create a column vector
                target_box_expanded = target_box.unsqueeze(1)
                # Compute smooth L1 loss for the current pair
                curr_loss = F.smooth_l1_loss(pred_box, target_box_expanded)  # Compute smooth L1 loss
                
                # Update the minimum loss if the current loss is smaller
                min_loss = min(min_loss, curr_loss)
            
            # Add the loss for the best prediction to the total loss
            loss += min_loss
    
    return loss

def classification_loss(predictions, targets):
    """
    Compute Cross-Entropy Loss for classification.

    Args:
        predictions (Tensor): Predicted class scores (batch_size, num_boxes, num_classes, 1).
        targets (list of dictionaries): Ground truth class labels and boxes.

    Returns:
        Tensor: Cross-Entropy Loss.
    """
    loss = 0
    for batch_idx in range(len(targets)):
        pred_classes = predictions[batch_idx]  # Predicted class scores
        target_classes = targets[batch_idx]['labels']  # Ground truth class labels
        
        num_boxes = pred_classes.size(0)
        for target_idx, target_class in enumerate(target_classes):
            # Initialize the maximum probability for class 1 (face)
            max_face_prob = -float('inf')
            
            # Iterate through each predicted class score and find the maximum probability for class 1
            for pred_idx in range(num_boxes):
                pred_class = pred_classes[pred_idx]  # Predicted class score
                
                # Compute softmax to get probabilities
                pred_prob = F.softmax(pred_class, dim=0)[1]  # Index 1 corresponds to the face class
                
                # Update the maximum probability if the current probability for class 1 is greater
                max_face_prob = max(max_face_prob, pred_prob)
            
            # Add the negative log probability of the maximum prediction for class 1 to the total loss
            loss += -torch.log(max_face_prob)
    
    return loss
