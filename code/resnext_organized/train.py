import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
from torchvision import transforms
from orientedRCNN import CustomOrientedRCNN
from resnextbackbone import ResNeXtBackbone
from obb_dataset import OBBDataset
from utils import obb_collate_fn
from torch.amp import autocast
from torch.amp import GradScaler
import gc
gc.collect()
from torch.optim.lr_scheduler import StepLR


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
lr = 1e-5
batch_size = 1
num_classes = 2  # 1 class + background

# Dataset
train_dataset = OBBDataset(
    annotation_path="obb_data/train.json",
    image_root="newdata_868/data/images/train",
    transforms=transforms.ToTensor()
)
val_dataset = OBBDataset(
    annotation_path="obb_data/val.json",
    image_root="newdata_868/data/images/train",
    transforms=transforms.ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=obb_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=obb_collate_fn)

# Model
backbone = ResNeXtBackbone(config_name=50)
model = CustomOrientedRCNN(backbone=backbone, num_classes=num_classes).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by a factor of 10 every 10 epochs

# Mixed Precision Scaler
scaler = GradScaler()  # No argument needed

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        # Move images and targets to device
        images = images.to(device)
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast("cuda"):
            loss = model(images, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")
