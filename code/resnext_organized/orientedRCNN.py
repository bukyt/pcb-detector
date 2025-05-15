import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import time
from torchvision.ops import nms
import math
timenow=time.time()
def compute_losses(class_logits, bbox_preds, targets, proposals):
    # Match GT boxes to proposals
    global timenow
    bbox_targets, cls_targets = match_proposals_to_targets(proposals, targets)

    cls_loss = F.cross_entropy(class_logits, cls_targets)
    reg_loss = fpdiou_loss(bbox_preds, bbox_targets)
    if time.time()>timenow+1:
        print(f"Classification loss: {cls_loss.item()}, Regression loss: {0.5*reg_loss.item()}")
        timenow=time.time()
    return cls_loss + 5 * reg_loss 


def match_proposals_to_targets(proposals, targets, max_distance=50.0):
    matched_boxes = []
    matched_labels = []

    for props, target in zip(proposals, targets):
        gt_boxes = target['boxes']  # shape: [N_gt, 5]
        gt_labels = target['labels']  # shape: [N_gt]

        if gt_boxes.numel() == 0:
            matched_boxes.append(torch.zeros((props.size(0), 5), device=props.device))
            matched_labels.append(torch.zeros((props.size(0),), dtype=torch.long, device=props.device))
            continue

        matched = []
        labels = []

        for p in props:
            dists = torch.norm(gt_boxes[:, :2] - p[:2], dim=1)
            min_idx = torch.argmin(dists)
            if dists[min_idx] < max_distance:
                matched.append(gt_boxes[min_idx])
                labels.append(gt_labels[min_idx])
            else:
                matched.append(torch.zeros(5, device=p.device))  # background
                labels.append(torch.tensor(0, dtype=torch.long, device=p.device))

        matched_boxes.append(torch.stack(matched))
        matched_labels.append(torch.stack(labels))

    return torch.cat(matched_boxes), torch.cat(matched_labels)


def fpdiou_loss(pred_boxes, target_boxes, angle_weight=1.0, eps=1e-6):
    assert not torch.isnan(pred_boxes).any()
    assert not torch.isnan(target_boxes).any()

    # Unpack the predicted and target boxes
    cx_p, cy_p, w_p, h_p, angle_p = pred_boxes.unbind(dim=1)
    cx_t, cy_t, w_t, h_t, angle_t = target_boxes.unbind(dim=1)

    # Center distance loss (Euclidean distance)
    center_dist = torch.sqrt((cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2 + eps)

    # Size loss (difference in areas of width * height)
    size_loss = (w_p - w_t) ** 2 + (h_p - h_t) ** 2

    # Angle loss (normalized angular difference)
    angle_diff = torch.abs(angle_p - angle_t) % 360
    angle_diff = torch.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    angle_loss = angle_diff / 180.0
    angle_loss = torch.clamp(angle_loss, 0, 1)

    # Focal Point Distance-IoU loss
    total_loss = 0.5 * center_dist + size_loss + angle_weight * angle_loss

    # Handle NaN values
    total_loss = torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)

    return total_loss.mean()

def rotated_roi_align(features, boxes, output_size=(7, 7)):
    """
    features: Tensor[B, C, H, W] - feature map
    boxes: List of [cx, cy, w, h, angle] in image coords, one list per image
    output_size: (H, W) output spatial resolution
    """
    pooled_features = []
    B, C, H, W = features.shape

    for batch_idx, boxes_per_image in enumerate(boxes):
        for box in boxes_per_image:
            cx, cy, w, h, angle = box
            angle_rad = -angle * torch.pi / 180.0  # Note: negative for rotation direction

            # Normalize coords to [-1, 1]
            theta = torch.tensor([
                [w / W * torch.cos(angle_rad), -h / H * torch.sin(angle_rad), 0],
                [w / W * torch.sin(angle_rad),  h / H * torch.cos(angle_rad), 0]
            ], dtype=torch.float32, device=features.device)

            # Translate to center
            theta[0, 2] = (2 * cx / W - 1)
            theta[1, 2] = (2 * cy / H - 1)

            # Grid sample
            grid = F.affine_grid(theta.unsqueeze(0), size=(1, C, *output_size), align_corners=False)
            patch = F.grid_sample(features[batch_idx:batch_idx+1], grid, align_corners=False)

            pooled_features.append(patch)

    return torch.cat(pooled_features, dim=0)  # [N, C, H, W]


class CustomOrientedRCNN(nn.Module):
    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.backbone = backbone  # Your ResNeXtBackbone
        self.rpn = CustomOrientedRPN(in_channels=backbone.out_channels)
        self.roi_align = CustomRotatedRoIAlign(output_size=(7, 7))
        self.head = OrientedRoIHead(in_channels=backbone.out_channels, num_classes=num_classes)
        self.max_proposals = 100
    def forward(self, images, targets=None):
        features = self.backbone(images)
        features = F.interpolate(features, scale_factor=0.5, mode='bilinear')
        proposals = self.rpn(features)
        proposals = [p[:self.max_proposals] for p in proposals]
        pooled = self.roi_align(features, proposals)
        class_logits, obb_preds = self.head(pooled)
        if self.training:
            loss = compute_losses(class_logits, obb_preds, targets, proposals)
            return loss
        if not self.training:
            batch_size = images.size(0)
            results = postprocess_outputs(class_logits, obb_preds, proposals, batch_size)

            for i, preds in enumerate(results):
                print(f"\nImage {i} predictions:")
                for j, pred in enumerate(preds):
                    print(f"  [#{j}] Label: {pred['label']}, Score: {pred['score']:.2f}, Box: {pred['box']}")

            return results



class CustomOrientedRPN(nn.Module):
    def __init__(self, in_channels, num_anchors=27, stride=32):
        super().__init__()
        self.num_anchors = num_anchors  # total combinations of (scale, ratio, angle)
        self.stride = stride

        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.obj_logits = nn.Conv2d(256, self.num_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(256, self.num_anchors * 5, kernel_size=1)  # [cx, cy, w, h, angle]

    def forward(self, feature_map):
        t = F.relu(self.conv(feature_map))  # [B, 256, H, W]
        logits = self.obj_logits(t)         # [B, A, H, W]
        deltas = self.bbox_deltas(t)        # [B, A*5, H, W]
        return self.generate_proposals(logits, deltas)

    def generate_proposals(self, logits, deltas):
        B, _, H, W = logits.shape
        A = self.num_anchors

        # [B, A*5, H, W] -> [B, H, W, A, 5] -> [B, H*W*A, 5]
        deltas = deltas.view(B, A, 5, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 5)
        # [B, A, H, W] -> [B, H, W, A] -> [B, H*W*A]
        logits = logits.permute(0, 2, 3, 1).reshape(B, -1)

        # Generate anchors: [H*W*A, 5]
        anchors = self.generate_anchors((H, W), self.stride, deltas.device)
        #print(anchors.shape[0])
        #print(deltas.shape[1])
        assert anchors.shape[0] == deltas.shape[1], "Anchor and delta count mismatch"

        # Decode proposals for each batch
        proposals = []
        for b in range(B):
            deltas_b = deltas[b]            # [N, 5]
            boxes_b = self.decode_deltas_to_boxes(anchors, deltas_b)  # [N, 5]
            proposals.append(boxes_b)

        return proposals  # List of length B, each [N, 5]

    def generate_anchors(self, feat_size, stride, device):
        H, W = feat_size
        shift_x = torch.arange(0, W * stride, step=stride, device=device)
        shift_y = torch.arange(0, H * stride, step=stride, device=device)
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shifts = torch.stack((shift_x, shift_y), dim=-1).reshape(-1, 2)  # [H*W, 2]

        # Define anchors with combinations of scale, ratio, angle
        scales = [32, 64, 128]
        ratios = [0.5, 1.0, 2.0]
        angles = [0.0, 45.0, 90.0]  # in degrees

        base_anchors = []
        for scale in scales:
            for ratio in ratios:
                w = scale * (ratio ** 0.5)
                h = scale / (ratio ** 0.5)
                for angle in angles:
                    base_anchors.append([w, h, angle])

        base_anchors = torch.tensor(base_anchors, device=device)  # [A, 3]
        A = base_anchors.shape[0]
        shifts = shifts[:, None, :].expand(-1, A, 2)               # [H*W, A, 2]
        wha = base_anchors[None, :, :].expand(shifts.size(0), -1, -1)  # [H*W, A, 3]
        anchors = torch.cat([shifts, wha], dim=-1).reshape(-1, 5)  # [H*W*A, 5]

        return anchors

    def decode_deltas_to_boxes(self, anchors, deltas):
        cx, cy, w, h, a = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3], anchors[:, 4]
        dx, dy, dw, dh, da = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3], deltas[:, 4]

        pred_cx = dx * w + cx
        pred_cy = dy * h + cy
        pred_w = torch.exp(dw) * w
        pred_h = torch.exp(dh) * h
        pred_angle = a + da  # assuming angle in degrees

        return torch.stack([pred_cx, pred_cy, pred_w, pred_h, pred_angle], dim=1)


class CustomRotatedRoIAlign(nn.Module):
    def __init__(self, output_size=(7, 7)):
        super().__init__()
        self.output_size = output_size

    def forward(self, feature_map, proposals):
        # Normally use torchvision.ops.roi_align or a custom CUDA-based rotated ROIAlign
        # Here we simulate pooled features
        B = len(proposals)
        pooled = rotated_roi_align(feature_map, proposals, output_size=self.output_size)
        return pooled
    
class OrientedRoIHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, 5)  # [cx, cy, w, h, angle]

    def forward(self, pooled_features):
        x = pooled_features.view(pooled_features.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)


def convert_to_corners(cx, cy, w, h, angle):
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    dx = w / 2
    dy = h / 2

    corners = [
        (-dx, -dy),
        ( dx, -dy),
        ( dx,  dy),
        (-dx,  dy)
    ]

    return [
        (
            cx + x * cos_theta - y * sin_theta,
            cy + x * sin_theta + y * cos_theta
        )
        for x, y in corners
    ]

def obb_to_aabb(boxes):
    # boxes: [N, 5] -> cx, cy, w, h, angle
    aabbs = []
    for box in boxes:
        cx, cy, w, h, angle = box.tolist()
        corners = convert_to_corners(cx, cy, w, h, angle)
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        aabbs.append([xmin, ymin, xmax, ymax])
    return torch.tensor(aabbs, dtype=torch.float32, device=boxes.device)


def postprocess_outputs(class_logits, obb_preds, proposals, batch_size, score_thresh=0.1, top_k=100):
    scores = F.softmax(class_logits, dim=1)
    confidences, labels = scores.max(dim=1)

    keep = confidences > score_thresh
    boxes = obb_preds[keep]
    labels = labels[keep]
    confidences = confidences[keep]

    # Convert to axis-aligned boxes for NMS
    boxes_xyxy = obb_to_aabb(boxes)  # implemented above

    keep_nms = nms(boxes_xyxy, confidences, iou_threshold=0.5)

    boxes = boxes[keep_nms]
    labels = labels[keep_nms]
    confidences = confidences[keep_nms]

    results = [[] for _ in range(batch_size)]

    # Need to assign predictions back to images.
    # We assume original proposal structure [P1, P2, ..., PN], flattened in same order.
    # Let's reconstruct a map from flat index to image index.
    image_indices = []
    for i, props in enumerate(proposals):
        image_indices.extend([i] * len(props))
    image_indices = torch.tensor(image_indices, device=boxes.device)
    image_indices = image_indices[keep][keep_nms]

    for i in range(batch_size):
        inds = (image_indices == i).nonzero(as_tuple=True)[0]
        top_inds = inds[:top_k]
        for idx in top_inds:
            results[i].append({
                "box": boxes[idx].detach().cpu().numpy(),
                "label": labels[idx].item(),
                "score": confidences[idx].item()
            })
    return results
