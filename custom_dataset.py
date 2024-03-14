import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, txt_file, transform=None,train=1):
        self.train=train
        self.data = self.parse_txt(txt_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, boxes_info = self.data[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert to tensor
        return image, boxes_info

    def parse_txt(self, txt_file):
        data = []
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            idx = 0
            while idx < len(lines):
                if self.train:
                    image_path = "datasets/WIDER_train/images/" + lines[idx].strip()
                else:
                    image_path = "datasets/WIDER_val/images/" + lines[idx].strip()
                idx += 1
                num_bounding_boxes = int(lines[idx].strip())
                idx += 1
                image = Image.open(image_path).convert('RGB')
                width, height = image.size
                scale_x = 320/width
                scale_y = 320/height
                if num_bounding_boxes == 0:
                    num_bounding_boxes = 1
                # Read bounding box information
                boxes_info = [lines[i].strip() for i in range(idx, idx + num_bounding_boxes)]
                # x y w h
                boxes = []
                labels=[]
                for box_info in boxes_info:
                    box_info = box_info.split(' ')
                    x = 0
                    y = 0
                    if(float(box_info[2]) * float(scale_x) < 1):
                        x = 1
                    if(float(box_info[3]) * float(scale_y) < 1):
                        y = 1
                    bounding_boxes = [int(float(box_info[0]) * float(scale_x)),int(float(box_info[1]) * float(scale_y)),int((float(box_info[0])+float(box_info[2])) * float(scale_x))+x,int((float(box_info[1])+float(box_info[3])) * float(scale_y))+y]
                    boxes.append(bounding_boxes)
                    labels.append(1)
                idx += num_bounding_boxes
                boxes_info = torch.tensor(boxes, dtype=torch.float32)
                box_dict = {}
                box_dict['boxes'] = boxes_info
                
                labels_info = torch.tensor(labels, dtype=torch.int64)
                box_dict['labels'] = labels_info
                # Append data tuple
                data.append((image_path, box_dict))

        return data

    def parse_txt2(self, txt_file):
        data = []
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            idx = 0
            while idx < len(lines):
                if self.train:
                    image_path = "datasets/WIDER_train/images/" + lines[idx].strip()
                else:
                    image_path = "datasets/WIDER_val/images/" + lines[idx].strip()
                idx += 1
                
                num_bounding_boxes = int(lines[idx].strip())
                idx += 1
                image = Image.open(image_path).convert('RGB')
                width, height = image.size
                scalex=320/width
                scaley=320/height
                if num_bounding_boxes == 0:
                    num_bounding_boxes = 1
                # Read bounding box information
                boxes_info = [lines[i].strip() for i in range(idx, idx + num_bounding_boxes)]
                boxes, labels = [], []
                for box_info in boxes_info:
                    box_info = box_info.split(' ')
                    
                    x, y, w, h = [float(coord) for coord in box_info[:4]]
                    
                    # Convert (x, y, w, h) to (cx, cy, w, h) with scaling
                    cx = ((x + w) / 2)*scalex
                    cy = ((y + h) / 2)*scaley
                    scaled_w = w*scalex
                    scaled_h = h*scaley
                    
                    boxes.append([cx, cy, scaled_w, scaled_h])
                    labels.append(1)  # Assuming all objects are of a single class
                idx += num_bounding_boxes
                boxes_info = torch.tensor(boxes, dtype=torch.float32)
                labels_info = torch.tensor(labels, dtype=torch.int64)
                #Scaled to 320x320 
                box_dict = {'boxes': boxes_info, 'labels': labels_info}
                data.append((image_path, box_dict))

        return data

