import torch
from torch.utils.data import DataLoader
from SSD_model import SSDMobileNetV2
from preprocessor import FaceDataset, collate_fn
import torch.optim as optim
import utils as ut

input_shape = (3, 300, 300)
num_classes = 1  # Number of classes, in this case, 1 for face detection
num_anchor_boxes = 4  # Number of anchor boxes per spatial location
num_epochs = 10

# Define your dataset and data loader
train_dir = 'D:/SSD_pytorch/datasets/WIDER_train/images'
ground_truth_file_train = 'D:/SSD_pytorch/datasets/wider_face_train_bbx_gt.txt'

face_dataset = FaceDataset(root_dir=train_dir, ground_truth_file=ground_truth_file_train, transform=None)
data_loader = DataLoader(face_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your model and move it to the appropriate device
model = SSDMobileNetV2(input_shape, num_classes).to(device)

# Define your optimizer and learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Custom loss function for bounding box detection
loss_function = ut.CustomLoss()

for epoch in range(num_epochs):
    for images, targets in data_loader:
        images = images.to(device)
        # Iterate over each target dictionary in the list
        for target_dict in targets:
            # Filter out invalid bounding boxes (width or height is 0)
            valid_boxes = [box for box in target_dict if box['w'] > 0 and box['h'] > 0]
            if not valid_boxes:
                continue  # Skip if there are no valid boxes
            
            num_boxes = len(valid_boxes)
            boxes = torch.tensor([[box['x'], box['y'], box['w'], box['h']] for box in valid_boxes]).to(device)
            
            # Generate target labels tensor
            target_labels = torch.zeros(num_boxes, num_anchor_boxes, num_classes).to(device)

            # Forward pass
            confidences, locations = model(images)

            # Compute your loss
            loss = loss_function(confidences, locations, boxes, target_labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch+1} completed")

print("Training finished")