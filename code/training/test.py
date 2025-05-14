import torch
import torchvision
from torchvision.transforms import ToTensor, Compose
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from obb_dataset import OBBDataset
from model import OBBFasterRCNN
from shapely.geometry import box as shapely_box
from shapely import affinity
import numpy as np
from PIL import Image
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def rotated_box_to_polygon(cx, cy, w, h, angle_deg):
    rect = shapely_box(-w/2, -h/2, w/2, h/2)
    rotated = affinity.rotate(rect, angle_deg, origin=(0, 0), use_radians=False)
    moved = affinity.translate(rotated, cx, cy)
    return np.array(moved.exterior.coords)

def visualize_predictions(image, predictions, save_path=None):
    image = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if predictions and len(predictions['boxes']) > 0:
        for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
            if score < 0.0:
                continue
            box = box.tolist()
            if len(box) == 5:
                cx, cy, w, h, angle = box
                poly = rotated_box_to_polygon(cx, cy, w, h, angle)
                patch = patches.Polygon(poly, linewidth=2, edgecolor='r', facecolor='none')
            elif len(box) == 4:
                xmin, ymin, xmax, ymax = box
                w = xmax - xmin
                h = ymax - ymin
                cx = xmin + w / 2
                cy = ymin + h / 2
                angle = 0  # assume upright
                poly = rotated_box_to_polygon(cx, cy, w, h, angle)
                patch = patches.Polygon(poly, linewidth=2, edgecolor='b', facecolor='none')
            else:
                continue  # skip invalid box

            ax.add_patch(patch)
            ax.text(cx, cy, f'{score:.2f}', color='white', fontsize=8, ha='center')
            ax.text(cx, cy - 10, f'Class: {label}', color='yellow', fontsize=8, ha='center')

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load validation dataset
    val_dataset = OBBDataset(
        annotation_path='obb_data/val.json',
        image_root='newdata_868/data/images/train',
        transforms=Compose([ToTensor()])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    # Load model
    model = OBBFasterRCNN(num_classes=2)
    model.load_state_dict(torch.load('checkpoints/model_last.pth', map_location=device))
    model.to(device)
    model.eval()

    # Output directory
    os.makedirs("predictions", exist_ok=True)

    # Inference loop
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            predictions = model(images)
            for pred in predictions:
                print(pred)
            # Visualize and save predictions
            visualize_predictions(images[0].cpu(), predictions[0], save_path=f"predictions/pred_{idx}.png")

            if idx == 4:  # Show only first 5 examples
                break

if __name__ == '__main__':
    main()
