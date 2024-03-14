import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from DeepMAD_model import OptimizedSSD
import numpy as np
from torchvision.models.detection.ssdlite import MobileNet_V3_Large_Weights
from is_this_loss import find_optimized_anchors,generate_anchors,decode_predictions
from custom_dataset import CustomDataset

def calculate_iou(box1, box2):
    # Calculate the intersection coordinates
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def calculate_accuracy(ground_truth_boxes, predicted_boxes, iou_threshold=0.22):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    if len(predicted_boxes) == 0:
        return 0, 0, 0  # No predicted boxes, return 0 precision, recall, and accuracy

    for gt_box in ground_truth_boxes:
        found_match = False

        for pred_box in predicted_boxes:
            iou = calculate_iou(gt_box, pred_box)
            
            if iou >= iou_threshold:
                true_positives += 1
                found_match = True
                break

        if not found_match:
            false_negatives += 1

    false_positives = len(predicted_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = true_positives / len(ground_truth_boxes)

    return precision, recall, accuracy


# Set up the paths and parameters
data_folder = "datasets/WIDER_val\images"
txt_file = "datasets/wider_face_val_bbx_gt.txt"

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Create a custom dataset
custom_dataset = CustomDataset(txt_file=txt_file, transform=transform,train=0)

feature_map_sizes = [(40, 40), (20, 20), (10, 10), (5, 5)]  # Feature map sizes for SSD
image_size = (320, 320)  # Input image size
scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]  # Example scales
aspect_ratios = [1, 2, 1/2, 3, 1/3]  # Example aspect ratios
optimal_boxes=find_optimized_anchors(custom_dataset)
opt_anchors=generate_anchors(feature_map_sizes,image_size,optimal_boxes).to("cuda")

output_size = 2
# Load your trained ResNet-18 model

model = OptimizedSSD(num_classes=output_size,anchors=opt_anchors)  # Adjust as necessary
model.load_state_dict(torch.load('trained_modelSSD_epoch_8.pth'))
model.eval()
model.to("cuda")  # If using GPU

# Evaluation loop
final_precision = 0
final_recall = 0
final_accuracy = 0
num_images = len(custom_dataset)

for i in range(num_images):
    image, targets = custom_dataset[i]  # Assuming targets include ground truth boxes
    input_tensor = image.unsqueeze(0).to("cuda")  # Add batch dimension and move to GPU if using

    with torch.no_grad():
        pred_locs, pred_scores = model(input_tensor)
    width, height = image.size
    # Decode and process predictions
    decoded_boxes = decode_predictions(pred_locs,width,height).squeeze(0) # Adjust for batch dimension if necessary

    # Convert logits to probabilities (if pred_scores are logits)
    pred_probs = torch.softmax(pred_scores.squeeze(0), dim=-1)  # Adjust for batch dimension if necessary

    # Apply confidence threshold and select boxes (this part may need adjustment based on your output structure)
    confidence_threshold = 0.6
    keep_boxes = pred_probs[:, 1] > confidence_threshold  # Assuming background class is at index 0

    final_boxes = decoded_boxes[keep_boxes].cpu().numpy()
    ground_truth_boxes = np.array(targets['boxes'])  # Adjust as necessary for your dataset structure

    precision, recall, accuracy = calculate_accuracy(ground_truth_boxes, final_boxes, iou_threshold=0.5)
    final_precision += precision
    final_recall += recall
    final_accuracy += accuracy

# Calculate averages
average_precision = final_precision / num_images
average_recall = final_recall / num_images
average_accuracy = final_accuracy / num_images

print(f'Final Precision: {average_precision}, Final Recall: {average_recall}, Final Accuracy: {average_accuracy}')
