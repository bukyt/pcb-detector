import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image

class OBBDataset(Dataset):
    def __init__(self, annotation_path, image_root, transforms=None):
        self.image_root = image_root
        self.transforms = transforms
        with open(annotation_path) as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']
        self.categories = data['categories']

        # Group annotations by image_id
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.ids = list(self.image_to_anns.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_info = self.images[image_id]
        img_path = os.path.join(self.image_root, image_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        anns = self.image_to_anns[image_id]
        boxes = []
        labels = []
        angles = []

        for ann in anns:
            # Expected bbox format: [cx, cy, w, h, angle]
            cx, cy, w, h, angle = ann['bbox']
            boxes.append([cx, cy, w, h, angle])
            labels.append(ann['category_id'] + 1)


        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
