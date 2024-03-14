import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import MobileNet_V3_Large_Weights
from custom_dataset import CustomDataset
from is_this_loss import encode_ground_truths
from SSD_model import SSD
import torch.nn.functional as F
from is_this_loss import normalize_boxes,generate_anchors, find_optimized_anchors,create_default_boxes,soft_negative_mining
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from DeepMAD_model import OptimizedSSD

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

num_anchors_per_layer = [4, 6, 6, 6, 4, 4]
aspect_ratios = [1, 2, 3, 1/2, 1/3]  # Extend or modify according to your needs
scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]  # Adjust scales if necessary
feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5),(3,3),(1,1)]  # Feature map sizes for SSD
image_size = (320, 320)  # Input image size
anchors=[4,6,6,6,4,4]
optimal_boxes=find_optimized_anchors(custom_dataset,k=anchors)
#default_boxes=create_default_boxes()
opt_anchors=generate_anchors(feature_map_sizes,image_size,optimal_boxes).to("cuda")
opt_anchors = opt_anchors.to("cuda")
n_default_boxes=0
for i in range(0,len(anchors)):
            n_default_boxes+=anchors[i]*feature_map_sizes[i][0]*feature_map_sizes[i][1]
# Create a data loader
batch_size = 12
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
# Set up the model
#model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=output_size, weights_backbone=MobileNet_V3_Large_Weights)
#model  = SSD(num_classes=output_size,anchors=opt_anchors)
model=OptimizedSSD(num_classes=output_size,anchors=opt_anchors)
# Set up optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005) 
scheduler = StepLR(optimizer, step_size=5,gamma=0.1)

# Training loop
num_epochs = 15 # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
print("starting training")
loc_loss_weight = 1.0
class_loss_weight = 1.0
running_ratio = 1.0  # Initial ratio
alpha = 0.1  
negative_mining=True
neg_pos_ratio=3

for epoch in range(num_epochs):
    model.train()
    total_loss_accumulated = 0.0
    total_loc_loss_accumulated = 0.0
    total_class_loss_accumulated = 0.0
    
    for images, targets in train_loader:
        images = torch.stack(images).to(device)
        
        gt_locs = torch.zeros((len(images), len(opt_anchors), 4), device=device)
        gt_labels = torch.zeros((len(images), len(opt_anchors)), dtype=torch.long, device=device)
        
        for i in range(len(images)):
            gt_boxes = normalize_boxes(targets[i]['boxes'], 320, 320).to(device)
            encoded_locs, encoded_labels = encode_ground_truths(opt_anchors, gt_boxes, targets[i]['labels'].to(device))
            gt_locs[i] = encoded_locs
            gt_labels[i] = encoded_labels
        
        optimizer.zero_grad()
        
        pred_locs, pred_scores = model(images)
        
        pos_mask = gt_labels > 0
        loc_loss = F.smooth_l1_loss(pred_locs[pos_mask], gt_locs[pos_mask], reduction='sum')
        
        cls_loss_all = F.cross_entropy(pred_scores.view(-1, model.num_classes), gt_labels.view(-1), reduction='none').view(gt_labels.size())
        
        cls_loss_pos = cls_loss_all[pos_mask]
        n_positives = pos_mask.sum(dim=1)  # Number of positives per batch item
        n_hard_negatives=torch.ceil(n_positives * neg_pos_ratio)
        if negative_mining:
            """confidence_neg_loss = cls_loss_all.clone()    #(N, 8732)
            confidence_neg_loss[pos_mask ] = 0.
            confidence_neg_loss, _ = confidence_neg_loss.sort(dim= 1, descending= True)
            
            hardness_ranks = torch.LongTensor(range(n_default_boxes)).unsqueeze(0).expand_as(confidence_neg_loss).to(device)  # (N, 8732)
            
            hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
            
            confidence_hard_neg_loss = confidence_neg_loss[hard_negatives]
            
            final_cls_loss = (confidence_hard_neg_loss.sum() + cls_loss_pos.sum()) / n_positives.sum().float()"""

            final_cls_loss=soft_negative_mining(cls_loss_all , gt_labels, alpha=0.5, beta=0.5)
        else:
            final_cls_loss= cls_loss_pos.sum()/n_positives.sum()
        
        total_loc_loss = loc_loss_weight * loc_loss
        total_class_loss = class_loss_weight * final_cls_loss
        total_loss = total_loc_loss + total_class_loss
        
        total_loss.backward()
        
        optimizer.step()
        
        total_loss_accumulated += total_loss.item()
        total_loc_loss_accumulated += loc_loss.item()
        total_class_loss_accumulated += total_class_loss.item()

    avg_loss = total_loss_accumulated / len(train_loader)
    print(f'Validation Loss: {avg_loss}')
    # Step the scheduler with average validation loss
    scheduler.step()
    print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss}, Loc Loss Weight: {loc_loss_weight}, Class Loss Weight: {class_loss_weight}")
    current_learning_rate = optimizer.param_groups[0]['lr']
    print(f'Epoch: {epoch+1}, Current Learning Rate: {current_learning_rate}')
    torch.save(model.state_dict(), f'trained_modelSSD_epoch_{epoch+1}.pth')

# Save the trained model
torch.save(model.state_dict(), 'trained_modelSSD13DeepMADneg2.pth')