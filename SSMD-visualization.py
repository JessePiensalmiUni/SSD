import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import numpy as np
from torchvision.models.detection.ssdlite import MobileNet_V3_Large_Weights
from SSD_model import SSD
from is_this_loss import find_optimized_anchors2,generate_anchors2,decode_predictions,convert_cxcywh_to_xyxy
from custom_dataset import CustomDataset
from torchvision.ops import nms
from DeepMAD_model import OptimizedSSD

output_size = 2
# Load your trained ResNet-18 model
data_folder = "datasets/WIDER_val\images"
txt_file = "datasets/wider_face_val_bbx_gt.txt"
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

custom_dataset = CustomDataset(txt_file=txt_file, transform=transform,train=0)
feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5),(3,3),(1,1)]  # Feature map sizes for SSD
image_size = (320, 320)  # Input image size
anchors=[4,6,6,6,4,4]
optimal_boxes=find_optimized_anchors2(custom_dataset,k=anchors)
opt_anchors=generate_anchors2(feature_map_sizes,image_size,optimal_boxes).to("cuda")
#model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=output_size, weights_backbone=MobileNet_V3_Large_Weights)
model=OptimizedSSD(num_classes=output_size,anchors=opt_anchors)

# Load the saved weights
model.load_state_dict(torch.load('trained_modelSSD_epoch_8.pth'))
# Set the model to evaluation mode

model.eval()

# Load and preprocess the input image
#input_image_path = "datasets/WIDER_val/images/14--Traffic/14_Traffic_Traffic_14_380.jpg"
input_image_path = "C:\\Users\\hel71\\Downloads\\images\\images\\group2.jpg"
image = Image.open(input_image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    # Add any other preprocessing steps used during training
])

input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
inputs = [img for img in input_tensor]
# Forward pass
with torch.no_grad():
    pred_locs, pred_scores = model(input_tensor)

width, height = image.size
# Decode and process predictions
decoded_boxes = decode_predictions(pred_locs,width,height).squeeze(0).to("cuda")
pred_probs = torch.softmax(pred_scores.squeeze(0), dim=-1)
face_likelihood_probs = pred_probs[:, 1].to("cuda")
decode_boxes_filter=decoded_boxes[face_likelihood_probs>0.6].to("cuda")
face_likelihood_probs_filtered=face_likelihood_probs[face_likelihood_probs>0.6].to("cuda")
# Apply NMS
nms_indices = nms(decode_boxes_filter, face_likelihood_probs_filtered, iou_threshold=0.4).to("cuda")
nms_boxes = decoded_boxes[nms_indices].to("cuda")
nms_scores = face_likelihood_probs[nms_indices].to("cuda")
# Calculate the width and height of each box
box_widths = nms_boxes[:, 2] - nms_boxes[:, 0]
box_heights = nms_boxes[:, 3] - nms_boxes[:, 1]

# Define minimum size criteria (these values are examples, adjust as necessary)
min_width = 5  # Minimum width to consider a box valid
min_height = 5  # Minimum height to consider a box valid

# Filter out boxes that don't meet the criteria
valid_boxes_mask = (box_widths >= min_width) & (box_heights >= min_height)
filtered_boxes = nms_boxes[valid_boxes_mask]

# Optionally, filter scores in the same way if you need them
if nms_scores is not None:
    filtered_scores = nms_scores[valid_boxes_mask]

# Filter boxes by confidence after NMS


# Visualization
draw = ImageDraw.Draw(image)
for box in filtered_boxes:
    box_coords = [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
    draw.rectangle(box_coords, outline="red", width=2)

image.show()
# image.save('path/to/save/output_image.jpg')