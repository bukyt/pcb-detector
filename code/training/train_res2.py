import torch
import torchvision
import torch.utils.data
from torchvision.transforms import ToTensor, Compose
from obb_dataset import OBBDataset
from model import OBBFasterRCNN  # You need to implement this
import os
from torchvision.transforms import functional as F
from shapely.geometry import box as shapely_box
from shapely import affinity

def collate_fn(batch):
    return tuple(zip(*batch))

def resize_rotated_boxes(boxes, old_size, new_size):
    # Unpack the boxes [cx, cy, w, h, angle]
    cx, cy, w, h, angle = boxes.unbind(1)
    
    # Resize the bounding box properties (center, width, height) based on the image size
    cx = cx * new_size[1] / old_size[1]  # Width scaling
    cy = cy * new_size[0] / old_size[0]  # Height scaling
    w = w * new_size[1] / old_size[1]
    h = h * new_size[0] / old_size[0]
    
    # Return the resized boxes in [cx, cy, w, h, angle] format
    return torch.stack([cx, cy, w, h, angle], dim=1)

def transform_images_and_boxes(image, target, new_size):
    # Resize the image using torchvision.transforms
    image = F.resize(image, new_size)
    
    # If the image is a tensor, use the shape to get the new dimensions
    if isinstance(image, torch.Tensor):
        old_size = (image.shape[1], image.shape[0])  # height, width
    else:
        old_size = (image.size[1], image.size[0])  # PIL image size (width, height)

    boxes = target['boxes']
    boxes_resized = resize_rotated_boxes(boxes, old_size, new_size)
    target['boxes'] = boxes_resized
    return image, target


def main():
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    train_dataset = OBBDataset(
        annotation_path='obb_data/train.json',
        image_root='newdata_868/data/images/train',
        transforms=Compose([ToTensor()])
    )
    val_dataset = OBBDataset(
        annotation_path='obb_data/val.json',
        image_root='newdata_868/data/images/val',
        transforms=Compose([ToTensor()])
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model
    model = OBBFasterRCNN(num_classes=2)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            
            # Apply the resize and box transformation here
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            for i in range(len(images)):
                image, target = transform_images_and_boxes(images[i], targets[i], new_size=(800, 800))
                images[i] = image  # Update the image with resized one
                targets[i] = target  # Update the target with resized boxes

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {losses.item()}")

        # Optionally save model
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), 'checkpoints/model_last.pth')

if __name__ == "__main__":
    main()
