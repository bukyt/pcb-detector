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

def fpdiou_loss(pred_boxes, target_boxes, angle_weight=1.0):
    """
    Computes the FPDIoU loss between predicted and target bounding boxes.
    https://arxiv.org/html/2405.09942v1
    Parameters:
    - pred_boxes: Tensor of shape (N, 5) containing predicted boxes in (cx, cy, w, h, angle_deg)
    - target_boxes: Tensor of shape (N, 5) containing ground truth boxes in the same format
    - angle_weight: Weighting factor for the angle loss component

    Returns:
    - loss: Scalar tensor representing the FPDIoU loss
    """
    # Extract components
    cx_p, cy_p, w_p, h_p, angle_p = pred_boxes.unbind(dim=1)
    cx_t, cy_t, w_t, h_t, angle_t = target_boxes.unbind(dim=1)

    # Center distance loss
    center_dist = torch.sqrt((cx_p - cx_t) ** 2 + (cy_p - cy_t) ** 2)

    # Size loss
    size_loss = torch.abs(w_p - w_t) + torch.abs(h_p - h_t)

    # Angle loss (convert degrees to radians)
    angle_diff = torch.abs(angle_p - angle_t) % 360
    angle_diff = torch.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    angle_loss = angle_diff / 180  # Normalize to [0, 1]
    
    # Combine losses
    loss = center_dist + size_loss + angle_weight * angle_loss
    return loss.mean()

def rotated_box_to_polygon(cx, cy, w, h, angle_deg):
    rect = shapely_box(-w/2, -h/2, w/2, h/2)  # centered at origin
    rotated = affinity.rotate(rect, angle_deg, origin=(0, 0))
    moved = affinity.translate(rotated, cx, cy)
    return moved

def rotated_iou(box1, box2):
    poly1 = rotated_box_to_polygon(*box1)
    poly2 = rotated_box_to_polygon(*box2)
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0

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
        self.bbox_pred = nn.Linear(in_features, num_classes * 5)  # cx, cy, w, h, angle
        self.num_classes = num_classes

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            labels, matched_idxs, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            matched_idxs = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)

        result = []
        losses = {}
        if self.training:
            loss_classifier, loss_box_reg = self.compute_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return result, losses

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[1]

        pred_scores = torch.softmax(class_logits, -1)
        boxes_per_image = [len(p) for p in proposals]

        pred_boxes = self.decode_rotated_boxes(box_regression, proposals)

        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []

        for boxes, scores in zip(pred_boxes, pred_scores):
            boxes = boxes.view(-1, 5)  # cx, cy, w, h, angle
            scores = scores.view(-1, num_classes)

            image_boxes = []
            image_scores = []
            image_labels = []

            for j in range(1, num_classes):  # skip background
                cls_scores = scores[:, j]
                inds = torch.where(cls_scores > self.score_thresh)[0]
                cls_scores = cls_scores[inds]
                cls_boxes = boxes[inds]

                if cls_boxes.numel() == 0:
                    continue

                keep = custom_nms_rotated(cls_boxes, cls_scores, self.nms_thresh)
                cls_boxes = cls_boxes[keep]
                cls_scores = cls_scores[keep]

                image_boxes.append(cls_boxes)
                image_scores.append(cls_scores)
                image_labels.append(torch.full_like(cls_scores, j, dtype=torch.int64))

            if image_boxes:
                image_boxes = torch.cat(image_boxes)
                image_scores = torch.cat(image_scores)
                image_labels = torch.cat(image_labels)
            else:
                image_boxes = torch.zeros((0, 5), device=device)
                image_scores = torch.zeros((0,), device=device)
                image_labels = torch.zeros((0,), dtype=torch.int64, device=device)

            all_boxes.append(image_boxes)
            all_scores.append(image_scores)
            all_labels.append(image_labels)

        return all_boxes, all_scores, all_labels

    
    def compute_loss(self, class_logits, box_regression, labels, regression_targets):
        device = class_logits.device
        N, total_dims = box_regression.shape
        num_classes = self.num_classes

        # --- Classification loss ---
        loss_cls = F.binary_cross_entropy_with_logits(class_logits, labels.float())  # Sigmoid for binary classification

        # --- Regression loss: only for foreground samples ---
        fg_inds = labels > 0
        if fg_inds.any():
            # Reshape to (N, num_classes, 5)
            box_regression = box_regression.view(N, num_classes, 5)
            regression_targets = regression_targets.view(N, num_classes, 5)

            indices = torch.arange(N, device=device)
            box_regression = box_regression[indices, labels]
            regression_targets = regression_targets[indices, labels]

            # Only select foreground examples
            box_regression_fg = box_regression[fg_inds]
            regression_targets_fg = regression_targets[fg_inds]

            # Compute FPDIoU loss
            loss_box_reg = fpdiou_loss(box_regression_fg, regression_targets_fg, angle_weight=1.0)
        else:
            loss_box_reg = torch.tensor(5.0, device=device)

        # Combine
        total_loss = loss_cls + loss_box_reg

        return total_loss, loss_cls, loss_box_reg



class OBBFasterRCNN(FasterRCNN):
    def __init__(self, num_classes):
        # Load ResNet+FPN backbone
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)

        # Define custom anchor sizes and aspect ratios
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),  # One size per feature map level
            aspect_ratios=((0.5, 1.0, 2.0),) * 5          # Same aspect ratios for each level
        )

        # Instantiate a FasterRCNN with the backbone and RPN
        temp_faster_rcnn = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

        # Get the necessary modules from the default FasterRCNN
        roi_pool = temp_faster_rcnn.roi_heads.box_roi_pool
        box_head = temp_faster_rcnn.roi_heads.box_head

        # Determine the in_features for the predictor
        in_features = box_head.fc6.out_features if isinstance(box_head, TwoMLPHead) else box_head.out_channels

        # Create the custom OBB RoI Heads
        obb_roi_heads = OBBRoIHeads(
            roi_pool,
            box_head,
            FastRCNNPredictor(in_features, num_classes),
            num_classes=num_classes,
            fg_iou_thresh=temp_faster_rcnn.roi_heads.fg_iou_thresh if hasattr(temp_faster_rcnn.roi_heads, 'fg_iou_thresh') else 0.5,
            bg_iou_thresh=temp_faster_rcnn.roi_heads.bg_iou_thresh if hasattr(temp_faster_rcnn.roi_heads, 'bg_iou_thresh') else 0.5,
            batch_size_per_image=temp_faster_rcnn.roi_heads.batch_size_per_image if hasattr(temp_faster_rcnn.roi_heads, 'batch_size_per_image') else 512,
            positive_fraction=temp_faster_rcnn.roi_heads.positive_fraction if hasattr(temp_faster_rcnn.roi_heads, 'positive_fraction') else 0.25,
            proposal_matcher=temp_faster_rcnn.roi_heads.proposal_matcher if hasattr(temp_faster_rcnn.roi_heads, 'proposal_matcher') else None,
            score_thresh=temp_faster_rcnn.roi_heads.score_thresh if hasattr(temp_faster_rcnn.roi_heads, 'score_thresh') else 0.05,
            nms_thresh=temp_faster_rcnn.roi_heads.nms_thresh if hasattr(temp_faster_rcnn.roi_heads, 'nms_thresh') else 0.5,
            detections_per_img=temp_faster_rcnn.roi_heads.detections_per_img if hasattr(temp_faster_rcnn.roi_heads, 'detections_per_img') else 100,
        )

        super().__init__(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=None,
            box_head=None,
            box_predictor=None, # Don't pass a default predictor
            roi_heads=obb_roi_heads # Use the custom RoI heads
        )

    def forward(self, images, targets=None):
        if self.training:
            assert targets is not None

        # This does transforms, backbone + RPN
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, _ = self.rpn(images, features, targets)

        # RoI heads with custom bbox + angle
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes, images.image_sizes)

        losses = {}
        if self.training:
            losses.update(detector_losses)
            return losses
        return detections