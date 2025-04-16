import torch
import numpy as np
from shapely.geometry import Polygon
import cv2
def box_iou_rotated(pred_boxes, gt_boxes):
    # Ensure that both pred_boxes and gt_boxes are on the CPU
    pred_boxes = pred_boxes.cpu().numpy() if pred_boxes.is_cuda else pred_boxes.numpy()
    gt_boxes = gt_boxes.cpu().numpy() if gt_boxes.is_cuda else gt_boxes.numpy()

    iou_matrix = torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]))

    for i, pred_box in enumerate(pred_boxes):
        # Check if pred_box has the expected 5 values: [cx, cy, w, h, angle]
        if len(pred_box) == 4:
            pred_box = np.append(pred_box, 0)  # Default angle to 0 if missing
        cx1, cy1, w1, h1, angle1 = pred_box

        for j, gt_box in enumerate(gt_boxes):
            # Check if gt_box has the expected 5 values: [cx, cy, w, h, angle]
            if len(gt_box) == 4:
                gt_box = np.append(gt_box, 0)  # Default angle to 0 if missing
            cx2, cy2, w2, h2, angle2 = gt_box

            # Convert rotated bounding boxes to polygons
            poly1 = rotate_box(cx1, cy1, w1, h1, angle1)
            poly2 = rotate_box(cx2, cy2, w2, h2, angle2)

            # Calculate IOU using Shapely's polygon intersection
            poly1 = Polygon(poly1)
            poly2 = Polygon(poly2)

            if poly1.is_valid and poly2.is_valid:
                intersection = poly1.intersection(poly2).area
                union = poly1.area + poly2.area - intersection
                iou_matrix[i, j] = intersection / union if union != 0 else 0.0

    return iou_matrix

def rotate_box(cx, cy, w, h, angle_deg):
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    # Get the four corners of the box using OpenCV
    box_points = cv2.boxPoints(((cx, cy), (w, h), angle_deg))  # Returns 4 points (corners of the box)
    return box_points
