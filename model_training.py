import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import MobileNet_V3_Large_Weights
from custom_dataset import CustomDataset
from is_this_loss import encode_ground_truths
from SSD_model import SSD
import torch.nn.functional as F
from is_this_loss import normalize_boxes,generate_anchors, find_optimized_anchors
from torch.optim.lr_scheduler import StepLR

output_size = 2
pretrained = False

# Set up the paths and parameters
data_folder = "datasets/WIDER_train\images"
txt_file = "datasets/wider_face_train_bbx_gt.txt"

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# Create a custom dataset
custom_dataset = CustomDataset(txt_file=txt_file, transform=transform,train=1)

feature_map_sizes = [(40, 40), (20, 20), (10, 10), (5, 5)]  # Feature map sizes for SSD
image_size = (320, 320)  # Input image size
scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]  # Example scales
aspect_ratios = [1, 2, 1/2, 3, 1/3]  # Example aspect ratios
optimal_boxes=find_optimized_anchors(custom_dataset)
opt_anchors=generate_anchors(feature_map_sizes,image_size,optimal_boxes).to("cuda")

# Create a data loader
batch_size = 6
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
# Set up the model
#model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=output_size, weights_backbone=MobileNet_V3_Large_Weights)
model  = SSD(num_classes=output_size,anchors=opt_anchors)
# Set up optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9)



# Training loop
num_epochs = 10 # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
print("starting training")

for epoch in range(num_epochs):
    model.train()
    total_loss_accumulated = 0.0  # To track loss over the epoch
    
    for images, targets in train_loader:
        images = torch.stack(images).to(device)  # Stack and move images to the correct device
        
        # Initialize encoded ground truths as tensors with the correct shape and device
        gt_locs = torch.zeros((len(images), len(opt_anchors), 4), device=device)
        gt_labels = torch.zeros((len(images), len(opt_anchors)), dtype=torch.long, device=device)
        
        # Encode ground truths for each image in the batch
        for i in range(len(images)):
            gt_boxes = normalize_boxes(targets[i]['boxes'], 320, 320).to(device)
            encoded_locs, encoded_labels = encode_ground_truths(opt_anchors, gt_boxes, targets[i]['labels'].to(device))
            gt_locs[i] = encoded_locs
            gt_labels[i] = encoded_labels
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_locs, pred_scores = model(images)
        
        # Localization loss (only for positive matches)
        pos_mask = gt_labels > 0
        loc_loss = F.smooth_l1_loss(pred_locs[:][pos_mask], gt_locs[:][pos_mask], reduction='sum')
        
        # Classification loss with Hard Negative Mining
        cls_loss = F.cross_entropy(pred_scores.view(-1, model.num_classes), gt_labels.view(-1), reduction='none').view_as(gt_labels)
        # Only faces 
        cls_loss_pos = cls_loss[:][pos_mask]
        
        # Combine losses
        total_loss = loc_loss + cls_loss_pos.sum()
        total_loss /= len(images)  # Normalize by batch size
        
        total_loss.backward()
        optimizer.step()
        
        total_loss_accumulated += total_loss.item()
    
    avg_loss = total_loss_accumulated / len(train_loader)
    print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss}")

# Save the trained model
torch.save(model.state_dict(), 'trained_modelSSD1.pth')
