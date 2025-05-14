import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np

from model import OBBFasterRCNN
from obb_dataset import OBBDataset
import torchvision.transforms.v2 as T

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = [
        T.ToTensor(),
        T.ConvertImageDtype(torch.float),
    ]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def angle_diff(a1, a2):
    """Returns the smallest difference between two angles in degrees."""
    return abs((a1 - a2 + 180) % 360 - 180)

def match_prediction(pred_box, gt_boxes, iou_thresh=1, angle_thresh=30):
    px, py, _, _, p_angle = pred_box
    for gt_box in gt_boxes:
        gx, gy, _, _, g_angle = gt_box
        dist = ((gx - px)**2 + (gy - py)**2) ** 0.5
        angle_error = angle_diff(p_angle, g_angle)
        if dist < iou_thresh and angle_error < angle_thresh:
            return True
    return False

def evaluate_background_and_poles(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                print(output)
                gt_boxes = target["boxes"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy() if "boxes" in output else []

                matched_gts = set()
                tp = 0
                for pred_box in pred_boxes:
                    matched = False
                    for i, gt_box in enumerate(gt_boxes):
                        if i in matched_gts:
                            continue
                        print(pred_box)
                        print(gt_box)
                        if match_prediction(pred_box, [gt_box]):
                            tp += 1
                            matched_gts.add(i)
                            matched = True
                            break
                    if not matched:
                        all_preds.append(1)  # predicted pole but no match (FP)
                        all_targets.append(0)

                for i in range(len(gt_boxes)):
                    if i not in matched_gts:
                        all_preds.append(0)  # no prediction for GT (FN)
                        all_targets.append(1)

                for _ in range(tp):
                    all_preds.append(1)
                    all_targets.append(1)

    labels = [0, 1]
    label_names = ["background", "capacitor_poles"]
    cm = confusion_matrix(all_targets, all_preds, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix: Background vs Capacitor Poles")
    plt.savefig("confusion_matrix_background_vs_poles.png")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "checkpoints/model_last.pth"
    val_ann = "obb_data/val.json"
    image_root = "newdata_868/data/images/train"

    val_dataset = OBBDataset(val_ann, image_root, transforms=get_transform(train=False))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = OBBFasterRCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    evaluate_background_and_poles(model, val_loader, device)

if __name__ == "__main__":
    main()
