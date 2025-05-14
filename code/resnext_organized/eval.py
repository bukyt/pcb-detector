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
        (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
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
    for preds in predictions:  # each `preds` is a dict with 'boxes', 'labels', 'scores'
        boxes = preds['boxes'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()

        filtered = []
        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                filtered.append({'box': box, 'label': label, 'score': score})
        filtered_batch.append(filtered)
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


cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
class_names = ['background', 'object']
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Visualization
font = cv2.FONT_HERSHEY_SIMPLEX
example_count = 0
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=obb_collate_fn)

for images, targets in val_loader:
    if example_count >= 5: break
    image = images[0].permute(1, 2, 0).cpu().numpy() * 255
    image = image.astype(np.uint8).copy()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    filtered_preds = filter_predictions_by_score(outputs, score_threshold)

    for pred in filtered_preds:
        cx, cy, w, h, angle = pred['box']
        print("Predicted Box:", pred['box'])
        print("Score:", pred['score'])
        print("Label:", pred['label'])

        corners = np.array(convert_to_corners(cx, cy, w, h, angle), dtype=np.int32)
        cv2.polylines(image_bgr, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image_bgr, f"P:{pred['label']} {pred['score']:.2f}", (int(cx), int(cy)), font, (0, 255, 0), 1)

    for t in targets:
        gt_boxes = t['boxes'].cpu().numpy()
        gt_labels = t['labels'].cpu().numpy()
        for box, label in zip(gt_boxes, gt_labels):
            corners = np.array(convert_to_corners(*box), dtype=np.int32)
            cv2.polylines(image_bgr, [corners], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.putText(image_bgr, f"GT:{label}", (int(box[0]), int(box[1]) - 10), font , (255, 0, 0), 1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Sample {example_count+1}")
    plt.show()
    example_count += 1
