import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms.v2 as T
from tqdm import tqdm
import os

from model import OBBFasterRCNN
from obb_dataset import OBBDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    train_ann = "obb_data/train.json"
    val_ann = "obb_data/val.json"
    image_root = "newdata_868/data/images/train"

    # Dataset and Dataloader
    train_dataset = OBBDataset(train_ann, image_root, transforms=get_transform(train=True))
    val_dataset = OBBDataset(val_ann, image_root, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = OBBFasterRCNN(num_classes=2)  # 1 class + background
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
    save_path = 'obb_fasterrcnn_full.pth'

    # Save the entire model
    torch.save(model, save_path)

    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train()
