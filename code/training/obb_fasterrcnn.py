import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch import nn
import torch

class OBBFasterRCNN(FasterRCNN):
    def __init__(self, num_classes):
        # Load a standard backbone
        backbone = torchvision.models.resnet50(weights="DEFAULT")
        backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove FC
        backbone.out_channels = 2048

        # Anchor generator
        rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )

        # RoI align
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        # Initialize the super class (FasterRCNN) with the custom backbone and RoI components
        super().__init__(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
        )

        # Replace the box predictor head with [cx, cy, w, h, θ]
        self.roi_heads.box_predictor = OBBBoxPredictor(1024, num_classes)

class OBBBoxPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 5)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
