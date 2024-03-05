import torch
import torch.nn as nn
import torchvision.models as models

class SSD(nn.Module):
    def __init__(self, num_classes=2, num_boxes=4):
        super(SSD, self).__init__()

        # Backbone: MobileNetV3 Large
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove the last classifier layer

        # Additional SSD layers
        self.conv1 = nn.Conv2d(960, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)

        # Customize your SSD layers here

        # For demonstration, let's add a default SSD prediction layer
        self.prediction = nn.Conv2d(256, num_boxes * (4 + num_classes), kernel_size=3, padding=1)

    def forward(self, x, num_classes=2):
        batched_tensor = torch.stack(x, dim=0)

        # Pass the tensor through the backbone network
        x = self.backbone(batched_tensor)

        # Continue with the rest of the forward pass
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)

        # Apply prediction layer
        predictions = self.prediction(x)

        # Reshape predictions
        # Assuming num_boxes is the number of predicted bounding boxes per feature map cell
        # and num_classes is the number of object classes
        batch_size = predictions.size(0)
        num_boxes = predictions.size(1) // (4 + num_classes)
        bbox_predictions = predictions[:, :num_boxes * 4, :, :]  # Extract bounding box predictions
        class_predictions = predictions[:, num_boxes * 4:, :, :]  # Extract class predictions

        # Reshape bounding box predictions to (batch_size, num_boxes, 4, H, W)
        bbox_predictions = bbox_predictions.view(batch_size, num_boxes, 4, -1)

        # Reshape class predictions to (batch_size, num_boxes, num_classes, H, W)
        class_predictions = class_predictions.view(batch_size, num_boxes, num_classes, -1)

        return {'bbox': bbox_predictions, 'classification': class_predictions}

# Example usage
ssd = SSD()
print(ssd)
