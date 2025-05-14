import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead
import torch.nn.functional as F
from torchvision.ops.boxes import box_convert
from shapely.geometry import box as shapely_box
from shapely import affinity
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import numpy as np
from torchvision.models.detection import image_list

class CustomTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__(min_size, max_size, image_mean, image_std)

    def forward(self, images, targets=None):
        if targets is not None:
            targets = self.sanitize_obbs(targets)
        images, targets = super().forward(images, targets)
        return images, targets

    def sanitize_obbs(self, targets):
        for target in targets:
            if target["boxes"].shape[1] == 5:
                obbs = target["boxes"]
                obbs[:, 4] = ((obbs[:, 4] + 180) % 360) - 180
                obbs = torch.where(torch.isnan(obbs), torch.zeros_like(obbs), obbs)
                obbs[:, 2] = torch.clamp(obbs[:, 2], min=1e-3)
                obbs[:, 3] = torch.clamp(obbs[:, 3], min=1e-3)
                target["obbs"] = obbs
                target["boxes"] = self.obb_to_aabb(obbs)
        return targets

    def postprocess(self, result, image_shapes, original_image_sizes):
        for res in result:
            if "boxes" in res and res["boxes"].shape[1] == 4 and "obbs" in res:
                res["boxes"] = res["obbs"]  # Replace AABB boxes with OBBs for output
        return result

    @staticmethod
    def obb_to_aabb(boxes):
        cx, cy, w, h, _ = boxes.unbind(1)
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)

def fpdiou_loss(pred_boxes, target_boxes, angle_weight=1.0):
    assert not torch.isnan(pred_boxes).any()
    assert not torch.isnan(target_boxes).any()
    cx_p, cy_p, w_p, h_p, angle_p = pred_boxes.unbind(dim=1)
    cx_t, cy_t, w_t, h_t, angle_t = target_boxes.unbind(dim=1)
    center_dist = torch.sqrt((cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2 + 1e-6)
    size_loss = torch.abs(w_p - w_t) + torch.abs(h_p - h_t)
    angle_diff = torch.abs(angle_p - angle_t) % 360
    angle_diff = torch.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    angle_loss = angle_diff / 180.0
    angle_loss = torch.clamp(angle_loss, 0, 1)
    total_loss = center_dist + size_loss + angle_weight * angle_loss
    total_loss = torch.where(torch.isnan(total_loss), torch.zeros_like(total_loss), total_loss)
    return total_loss.mean()

def rotated_box_to_polygon(cx, cy, w, h, angle_deg):
    rect = shapely_box(-w/2, -h/2, w/2, h/2)
    rotated = affinity.rotate(rect, angle_deg, origin=(0, 0))
    moved = affinity.translate(rotated, cx, cy)
    return moved

def rotated_iou(box1, box2):
    poly1 = rotated_box_to_polygon(*box1)
    poly2 = rotated_box_to_polygon(*box2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    try:
        result = inter / union
        return abs(result)
    except:
        return 0
def custom_nms_rotated(boxes, scores, iou_threshold=0.5):
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        current = idxs[0].item()
        keep.append(current)
        if idxs.numel() == 1:
            break
        current_box = boxes[current].tolist()
        rest = boxes[idxs[1:]].tolist()
        ious = torch.tensor([rotated_iou(current_box, b) for b in rest])
        idxs = idxs[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)

class OBBRoIHeads(RoIHeads):
    def __init__(self, box_roi_pool, box_head, box_predictor, num_classes,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 proposal_matcher, score_thresh, nms_thresh, detections_per_img):
        super().__init__(box_roi_pool, box_head, box_predictor,
                         fg_iou_thresh, bg_iou_thresh,
                         batch_size_per_image, positive_fraction,
                         proposal_matcher, score_thresh, nms_thresh, detections_per_img)
        in_features = box_head.fc6.out_features if isinstance(box_head, TwoMLPHead) else box_head.out_channels
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 5)
        self.num_classes = num_classes
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, features, proposals, image_shapes, targets=None):
        result = []
        losses = {}

        if self.training:
            labels, matched_idxs, regression_targets = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)
        box_regression = torch.where(torch.isnan(box_regression), torch.zeros_like(box_regression), box_regression)
        box_regression[..., 2:4] = torch.clamp(box_regression[..., 2:4], min=1e-3, max=1e3)
        box_regression[..., 4] = torch.remainder(box_regression[..., 4], 360)

        if self.training:
            loss_classifier, loss_box_reg, missed_obj_penalty = self.compute_loss(
                class_logits, box_regression, labels, regression_targets, matched_idxs, targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "missed_penalty": missed_obj_penalty
            }
            return [], losses
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            
            for i in range(len(boxes)):
                result.append({
                    "obbs": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i]
                })
            return result, {}

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[1]
        pred_scores = torch.softmax(class_logits, -1)
        boxes_per_image = [len(p) for p in proposals]
        pred_boxes = box_regression.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores in zip(pred_boxes, pred_scores):
            scores = scores.view(-1, num_classes)
            image_boxes, image_scores, image_labels = [], [], []
            for j in range(1, num_classes):
                cls_scores = scores[:, j]
                inds = torch.where(cls_scores > self.score_thresh)[0]
                cls_scores = cls_scores[inds]
                cls_boxes = boxes[inds, j * 5:(j + 1) * 5]
                if cls_boxes.numel() == 0:
                    continue
                cls_boxes[:, 4] = cls_boxes[:, 4] % 360
                keep = custom_nms_rotated(cls_boxes, cls_scores, self.nms_thresh)
                cls_boxes = cls_boxes[keep]
                cls_scores = cls_scores[keep]
                image_boxes.append(cls_boxes)
                image_scores.append(cls_scores)
                image_labels.append(torch.full_like(cls_scores, j, dtype=torch.int64))
            if image_boxes:
                all_boxes.append(torch.cat(image_boxes, dim=0))
                all_scores.append(torch.cat(image_scores, dim=0))
                all_labels.append(torch.cat(image_labels, dim=0))
            else:
                all_boxes.append(torch.empty((0, 5), device=device))
                all_scores.append(torch.empty((0,), device=device))
                all_labels.append(torch.empty((0,), dtype=torch.int64, device=device))
        return all_boxes, all_scores, all_labels

    def compute_loss(self, class_logits, box_regression, labels, regression_targets, matched_idxs, targets):
        device = class_logits.device
        N, total_dims = box_regression.shape
        num_classes = self.num_classes
        loss_cls = F.cross_entropy(class_logits, labels)

        fg_inds = labels > 0
        if fg_inds.any():
            box_regression = box_regression.view(N, num_classes, 5)
            regression_targets = regression_targets.view(N, num_classes, 5)
            indices = torch.arange(N, device=device)
            box_regression = box_regression[indices, labels]
            regression_targets = regression_targets[indices, labels]

            box_regression_fg = box_regression[fg_inds]
            regression_targets_fg = regression_targets[fg_inds]

            assert not torch.isnan(box_regression_fg).any(), "NaNs in box_regression_fg"
            assert not torch.isnan(regression_targets_fg).any(), "NaNs in regression_targets_fg"

            loss_box_reg = fpdiou_loss(box_regression_fg, regression_targets_fg, angle_weight=1.0)
        else:
            loss_box_reg = torch.tensor(5.0, device=device)

        missed_obj_penalty = 0.0
        for matched, target in zip(matched_idxs, targets):
            gt_labels = target["labels"]
            matched_gt_inds = matched[matched >= 0]
            num_gt = len(gt_labels)
            num_matched = len(torch.unique(matched_gt_inds))
            num_missed = num_gt - num_matched
            missed_obj_penalty += num_missed * 5
        missed_obj_penalty = torch.tensor(missed_obj_penalty, device=device, dtype=torch.float)

        print(f"✅ loss_cls: {loss_cls.item()}, loss_box_reg: {loss_box_reg.item()}, missed_penalty: {missed_obj_penalty.item()}")

        return loss_cls + loss_box_reg + missed_obj_penalty, loss_cls, loss_box_reg, missed_obj_penalty

class OBBFasterRCNN(FasterRCNN):
    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', weights=None)
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        temp_faster_rcnn = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
        roi_pool = temp_faster_rcnn.roi_heads.box_roi_pool
        box_head = temp_faster_rcnn.roi_heads.box_head
        in_features = box_head.fc6.out_features if isinstance(box_head, TwoMLPHead) else box_head.out_channels
        obb_roi_heads = OBBRoIHeads(
            roi_pool,
            box_head,
            FastRCNNPredictor(in_features, num_classes),
            num_classes=num_classes,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            proposal_matcher=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )
        super().__init__(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            roi_heads=obb_roi_heads
        )
        self.transform = CustomTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def forward(self, images, targets=None):
        if self.training:
            assert targets is not None
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, _ = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, images.image_sizes)
        if self.training:
            return detector_losses
        return detections

