import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import MobileNet_V3_Large_Weights
from custom_dataset import CustomDataset
from is_this_loss import classification_loss,bbox_regression_loss
from SSD_model import SSD

#print(torch.cuda.is_available())
#print(torch.version.cuda)
#cuda_id = torch.cuda.current_device()
#print(torch.cuda.current_device())
       
#print(torch.cuda.get_device_name(cuda_id))

output_size = 2
pretrained = False

# Set up the paths and parameters
data_folder = "datasets/WIDER_train\images"
txt_file = "datasets/wider_face_train_bbx_gt.txt"

# Set up transformations
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# Create a custom dataset
custom_dataset = CustomDataset(txt_file=txt_file, transform=transform)

# Create a data loader
batch_size = 1
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
# Set up the model
#model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=output_size, weights_backbone=MobileNet_V3_Large_Weights)
model  = SSD(num_classes=output_size)
# Set up optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 5 # Adjust as needed
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
print("starting training")
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs = [img.to(device) for img in inputs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to the same device    

        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)

        # Compute the losses
        bbox_loss = bbox_regression_loss(outputs['bbox'], targets)
        classification_loss_ad = classification_loss(outputs['classification'], targets)
        
        # Total loss
        loss = bbox_loss + classification_loss_ad

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
