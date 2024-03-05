from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import PIL.Image as Image
import os
import torch

class FaceDataset(Dataset):
    def __init__(self, root_dir, ground_truth_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.parse_ground_truth(ground_truth_file,4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        boxes = self.data[idx]['boxes']  # List of dictionaries

        if self.transform:
            image = self.transform(image)

        return image, boxes

    @staticmethod
    def parse_ground_truth(file_path, num_anchor_boxes):
        """
        Parse the ground truth text file and extract image filenames along with bounding box information.
        Args:
            file_path: Path to the ground truth text file.
            num_anchor_boxes: Number of anchor boxes used per spatial location.
        Returns:
            List of dictionaries containing filename and bounding box information.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = []
        i = 0
        while i < len(lines):
            filename = lines[i].strip()
            num_boxes = int(lines[i + 1])
            if num_boxes > 0:
                boxes = []
                for j in range(num_boxes):
                    box_info = list(map(int, lines[i + 2 + j].split()))
                    if box_info[7] == 0:  # Checking if the box is valid (invalid = 1)
                        # Replicate each bounding box for the number of anchor boxes
                        for k in range(num_anchor_boxes):
                            boxes.append({
                                'x': box_info[0],
                                'y': box_info[1],
                                'w': box_info[2],
                                'h': box_info[3],
                            })
                data.append({'filename': filename, 'boxes': boxes})
                i += 2 + num_boxes
            else:
                i += 3

        return data

class Rescale(object):
    """Rescale the image in a sample to a given size and adjust the bounding boxes accordingly."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']
        w, h = image.size
        new_h, new_w = self.output_size if isinstance(self.output_size, tuple) else (self.output_size, self.output_size)
        
        img = transforms.functional.resize(image, (new_h, new_w))
        
        # Rescale bounding boxes
        boxes_rescaled = []
        for box in boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            boxes_rescaled.append({
                'x': x * new_w / w,
                'y': y * new_h / h,
                'w': w * new_w / w,
                'h': h * new_h / h
            })

        return {'image': transforms.ToTensor()(img), 'boxes': boxes_rescaled}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']
        # Swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': transforms.ToTensor()(image),
                'boxes': torch.as_tensor(boxes)}

def collate_fn(batch):
    # Extract images and boxes from the batch
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]

    # Resize images to a fixed size
    resized_images = [transforms.Resize((300, 300))(image) for image in images]

    # Scale bounding box coordinates
    scaled_boxes = []
    for orig_boxes, image in zip(boxes, images):
        orig_width, orig_height = image.size
        new_width, new_height = 300, 300
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height
        scaled_boxes.append([{
            'x': int(box['x'] * scale_x),
            'y': int(box['y'] * scale_y),
            'w': int(box['w'] * scale_x),
            'h': int(box['h'] * scale_y)
        } for box in orig_boxes])

    # Convert resized images to tensors
    images = [transforms.ToTensor()(image) for image in resized_images]
    images = torch.stack(images, dim=0)

    return images, scaled_boxes