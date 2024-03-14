import torch
import torch.nn as nn
import torchvision.models as models

class OptimizedSSD(nn.Module):
    def __init__(self, num_classes=2, anchors=None, optimized_channels=[176, 152, 39, 256, 133, 256, 74, 167], optimized_depths=[1, 1, 1]):
        super(OptimizedSSD, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        
        # Adjusted backbone - you might want to fine-tune or replace according to the optimized widths and depths
        mobilenet = models.mobilenet_v3_large(pretrained=True)  # Consider adjusting this based on your optimization
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])  # Still removing the classifier
        
        # Adjust layers according to optimized_channels
        feature_channels = optimized_channels  # This now directly reflects your optimization results
        
        # Assuming num_anchors and feature_map_sizes remain constant, but you might adjust these based on optimization
        num_anchors = [4, 6, 6, 6, 4, 4]  # Example setup
        feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        
        # Layers
        self.loc_pred_layers = nn.ModuleList()
        self.class_pred_layers = nn.ModuleList()
        self.additional_layers = nn.ModuleList()

        prev_channel_size = 960  # Starting point from the backbone
        
        # Example way to integrate optimized depths, by simply using the first value for now
        for channels, fm_size in zip(feature_channels, feature_map_sizes):
            self.additional_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_channel_size, channels, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(),
                )
            )
            n_anchors = num_anchors[feature_map_sizes.index(fm_size)] * fm_size[0] * fm_size[1]
            self.loc_pred_layers.append(nn.Conv2d(channels, n_anchors * 4, kernel_size=3, padding=1))
            self.class_pred_layers.append(nn.Conv2d(channels, n_anchors * num_classes, kernel_size=3, padding=1))
            
            prev_channel_size = channels  # Prepare for the next iteration

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