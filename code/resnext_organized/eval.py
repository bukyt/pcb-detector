import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from obb_dataset import OBBDataset
from orientedRCNN import CustomOrientedRCNN
from resnextbackbone import ResNeXtBackbone
from utils import obb_collate_fn
from torchvision import transforms
from shapely.geometry import Polygon
import math
import cv2

# Convert [cx, cy, width, height, angle] to 4 corners (polygon points)
def convert_to_corners(cx, cy, w, h, angle):
    angle_rad = math.radians(angle)
    hw = w / 2
    hh = h / 2
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    corners = [
        (cx-hw, cy-hh), (cx+hw, cy-hh), (cx+hw, cy-hh), (cx-hw, cx+hh)
    ]
    return [
        (cos_theta * dx - sin_theta * dy + cx, sin_theta * dx + cos_theta * dy + cy)
        for (dx, dy) in corners
    ]

# IoU calculation
def compute_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0

# Score threshold filter
def filter_predictions_by_score(predictions, threshold):
    filtered_batch = []
    for predClump in predictions:
        for preds in predClump:  # each `preds` is a dict with 'box', 'label', 'score'
            boxes = preds['box']
            labels = preds['label']
            scores = preds['score']
            filtered = []
            if scores<threshold:
                filtered.append({'box': boxes, 'label': labels, 'score': scores})
            filtered_batch.append(filtered)
    print(filtered_batch)
    return filtered_batch  # list of lists: filtered predictions per image

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
score_threshold = 0.51
iou_threshold = 0.5

# Model and dataset
backbone = ResNeXtBackbone(config_name=50)
model = CustomOrientedRCNN(backbone=backbone, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model.pth"), strict=False)
model.eval()

val_dataset = OBBDataset(
    annotation_path="obb_data/val.json",
    image_root="newdata_868/data/images/train",
    transforms=transforms.ToTensor()
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=obb_collate_fn)

all_preds, all_labels = [], []

with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        for t in targets:
            t['boxes'], t['labels'] = t['boxes'].to(device), t['labels'].to(device)

        outputs = model(images)  # outputs is a list: one per image
        filtered_preds_batch = filter_predictions_by_score(outputs, score_threshold)

        for preds, target in zip(filtered_preds_batch, targets):
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            matched_gt = np.zeros(len(gt_boxes), dtype=bool)

            for pred in preds:
                cx, cy, w, h, angle = pred['box']
                pred_corners = convert_to_corners(cx, cy, w, h, angle)
                pred_label = pred['label']
                matched = False

                for i, gt_box in enumerate(gt_boxes):
                    if matched_gt[i]:
                        continue
                    gt_corners = convert_to_corners(*gt_box)
                    if compute_iou(pred_corners, gt_corners) > iou_threshold and pred_label == gt_labels[i]:
                        all_preds.append(pred_label)
                        all_labels.append(gt_labels[i])
                        #print(pred_label)
                        matched_gt[i] = True
                        matched = True
                        break

                if not matched:
                    # False positive: predicted something incorrect
                    all_preds.append(pred_label)
                    all_labels.append(0)

            for i, matched in enumerate(matched_gt):
                if not matched:
                    # False negative: missed a ground truth
                    all_preds.append(0)
                    all_labels.append(gt_labels[i])



print(len(all_preds))
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
class_names = ['object','background']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add numbers inside the squares
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
