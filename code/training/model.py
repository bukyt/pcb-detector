import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead


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

    def compute_loss(self, class_logits, box_regression, labels, regression_targets):
        loss_cls = nn.functional.cross_entropy(class_logits, labels)

        # Assuming regression_targets and box_regression are shaped (N, num_classes * 5)
        # We reshape them to (N, num_classes, 5)
        N, total_dims = box_regression.shape
        num_classes = self.num_classes
        box_regression = box_regression.view(N, num_classes, 5)
        regression_targets = regression_targets.view(N, num_classes, 5)

        # Select only the regression targets for the correct class
        device = box_regression.device
        indices = torch.arange(N, device=device)
        labels = labels.to(device)
        box_regression = box_regression[indices, labels]
        regression_targets = regression_targets[indices, labels]

        # Split components
        cx_pred, cy_pred, w_pred, h_pred, angle_pred = box_regression.split(1, dim=1)
        cx_tgt,  cy_tgt,  w_tgt,  h_tgt,  angle_tgt  = regression_targets.split(1, dim=1)

        # Normalize angle from degrees [-180, 180] to [0, 1]
        angle_pred_norm = (angle_pred + 180.0) / 360.0
        angle_tgt_norm = (angle_tgt + 180.0) / 360.0

        # Compute loss components with weights
        loss_cx = nn.functional.smooth_l1_loss(cx_pred, cx_tgt, beta=1.0, reduction="mean")
        loss_cy = nn.functional.smooth_l1_loss(cy_pred, cy_tgt, beta=1.0, reduction="mean")
        loss_w  = nn.functional.smooth_l1_loss(w_pred,  w_tgt,  beta=1.0, reduction="mean")
        loss_h  = nn.functional.smooth_l1_loss(h_pred,  h_tgt,  beta=1.0, reduction="mean")
        loss_angle = nn.functional.smooth_l1_loss(angle_pred_norm, angle_tgt_norm, beta=1.0, reduction="mean")

        # Weighted sum (cx and cy more important, w/h less, angle moderate)
        loss_box_reg = (2.0 * loss_cx +
                        2.0 * loss_cy +
                        0.5 * loss_w +
                        0.5 * loss_h +
                        1.0 * loss_angle)

        return loss_cls, loss_box_reg


class OBBFasterRCNN(FasterRCNN):
    def __init__(self, num_classes):
        # Load ResNet+FPN backbone
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)

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