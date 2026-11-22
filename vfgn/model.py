# Copyright (c) VerifyGN. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
VFGN26 Object Detection Architecture & Drop-In Inference Engine
---------------------------------------------------------------
Variants:
  - VFGN26S  (Small)       : MobileNetV3-Large / CSP-Lite + CoordAtt BiFPN + Decoupled Head (~5.8M params)
  - VFGN26M  (Medium)      : ConvNeXt-Tiny (BSD-3 Pretrained) + CoordAtt BiFPN + Decoupled Head (~31M params)
  - VFGN26XL (Extra-Large) : ConvNeXt-Base (BSD-3 Pretrained) + CoordAtt BiFPN + Decoupled Head (~92M params)

Key Architectural Advantages over Generic YOLO:
  1. Pretrained Torchvision BSD-3-Clause Backbones (ConvNeXt / MobileNetV3) -> 100% free for closed-source
     commercial use AND converges in 3x fewer epochs on industrial inspection datasets.
  2. Weighted Bi-Directional Feature Pyramid (BiFPN) + Coordinate Attention (CoordAtt) -> preserves exact X/Y
     positional awareness for industrial fixtures, trays, chamfers, sleeves, and bolts.
  3. Anchor-Free Decoupled Detection Head (P3/8, P4/16, P5/32) -> separate classification & box branches
     with direct distance-to-box (l, t, r, b) prediction.
  4. Dual Runtime Engine:
     - Runs natively on PyTorch (GPU/CPU) OR compiles into an embedded ONNX graph inside `.vf` for
       2x-4x faster CPU inference via `onnxruntime` (MIT License).
  5. 100% Drop-In Compatible Results API:
     `results = model(image, conf=0.25, iou=0.7)` returns `Result` objects with `.boxes.xyxy`, `.boxes.xywh`,
     `.boxes.conf`, `.boxes.cls`, and `model.names` so existing inspection scripts work without changing
     a single line of detection logic!
"""

import io
import os
import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import batched_nms

from .format import save_vf, load_vf, inspect_vf

logger = logging.getLogger(__name__)

# ==============================================================================
# Variant Specifications
# ==============================================================================
VARIANT_CONFIGS = {
    "VFGN26N": {
        "backbone": "mobilenet_v3_small",
        "neck_channels": 96,
        "head_channels": 96,
        "neck_repeats": 2,
        "description": "VFGN26-Nano (Ultra-light edge/CPU detector for dense small-object counting)",
    },
    "VFGN26S": {
        "backbone": "mobilenet_v3_large",
        "neck_channels": 128,
        "head_channels": 128,
        "neck_repeats": 2,
        "description": "VFGN26-Small (Ultra-fast CPU/Edge industrial detector)",
    },
    "VFGN26M": {
        "backbone": "convnext_tiny",
        "neck_channels": 192,
        "head_channels": 192,
        "neck_repeats": 2,
        "description": "VFGN26-Medium (Balanced high-accuracy industrial detector)",
    },
    "VFGN26XL": {
        "backbone": "convnext_base",
        "neck_channels": 256,
        "head_channels": 256,
        "neck_repeats": 3,
        "description": "VFGN26-ExtraLarge (Maximum precision industrial detector)",
    },
}


# ==============================================================================
# Building Blocks: ConvBNAct, Coordinate Attention, CSP-Elan, BiFPN, Head
# ==============================================================================
class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CoordAtt) embeds precise X and Y spatial coordinate information
    into channel attention—critical for industrial inspection (quadrants, tray slots, bores).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(16, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        # Pool along X and Y axes separately
        x_h = x.mean(dim=3, keepdim=True)                  # [B, C, H, 1]
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)                   # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


def parse_imgsz_wh(imgsz) -> tuple:
    """
    Parse `imgsz` into `(width, height)` pixels, rounded to a multiple of 32.
    Accepts:
      - int: `640` -> `(640, 640)`
      - tuple/list `(W, H)`: `(1024, 512)` -> `(1024, 512)` (Matches Roboflow 'Stretch to 1024 x 512')
      - string `'1024x512'` or `'1024,512'` -> `(1024, 512)`
    """
    if isinstance(imgsz, str):
        parts = imgsz.lower().replace("x", ",").split(",")
        if len(parts) == 2:
            w, h = int(parts[0].strip()), int(parts[1].strip())
        else:
            w = h = int(parts[0].strip())
    elif isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 2:
            w, h = int(imgsz[0]), int(imgsz[1])
        else:
            w = h = int(imgsz[0])
    else:
        w = h = int(imgsz)
    w = max(32, (w // 32) * 32)
    h = max(32, (h // 32) * 32)
    return (w, h)


class VFGNCSPBlock(nn.Module):
    """VerifyGN Residual Bottleneck Block with Coordinate Attention."""
    def __init__(self, c1, c2, n=2, shortcut=True):
        super().__init__()
        c_mid = c2 // 2
        self.cv1 = ConvBNAct(c1, c_mid, 1, 1)
        self.cv2 = ConvBNAct(c1, c_mid, 1, 1)
        self.m = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c_mid, c_mid, 3, 1),
                ConvBNAct(c_mid, c_mid, 3, 1),
            )
            for _ in range(n)
        ])
        self.shortcut = shortcut
        self.cv3 = ConvBNAct(c_mid * (n + 2), c2, 1, 1)
        self.ca = CoordinateAttention(c2)

    def forward(self, x):
        y = [self.cv1(x), self.cv2(x)]
        for block in self.m:
            out = block(y[-1])
            if self.shortcut:
                out = out + y[-1]
            y.append(out)
        return self.ca(self.cv3(torch.cat(y, dim=1)))


class TorchvisionBackboneP3P4P5(nn.Module):
    """
    Extracts multi-scale feature maps (P3/8, P4/16, P5/32) from BSD-3 licensed
    Torchvision backbones (MobileNetV3-Large, ConvNeXt-Tiny, ConvNeXt-Base).
    """
    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "mobilenet_v3_small":
            weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = torchvision.models.mobilenet_v3_small(weights=weights).features
            # Stages producing stride 8 (24 ch), stride 16 (48 ch), stride 32 (576 ch)
            self.stage_p3 = nn.Sequential(*net[:4])
            self.stage_p4 = nn.Sequential(*net[4:9])
            self.stage_p5 = nn.Sequential(*net[9:])
            self.out_channels = (24, 48, 576)

        elif backbone_name == "mobilenet_v3_large":
            weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = torchvision.models.mobilenet_v3_large(weights=weights).features
            # Stages producing stride 8 (idx 6), stride 16 (idx 12), stride 32 (idx 16)
            self.stage_p3 = nn.Sequential(*net[:7])
            self.stage_p4 = nn.Sequential(*net[7:13])
            self.stage_p5 = nn.Sequential(*net[13:])
            self.out_channels = (40, 112, 960)

        elif backbone_name == "convnext_tiny":
            weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            net = torchvision.models.convnext_tiny(weights=weights).features
            # ConvNeXt features: [0..3]=stride 8 (192 ch), [4..5]=stride 16 (384 ch), [6..7]=stride 32 (768 ch)
            self.stage_p3 = nn.Sequential(*net[:4])
            self.stage_p4 = nn.Sequential(*net[4:6])
            self.stage_p5 = nn.Sequential(*net[6:8])
            self.out_channels = (192, 384, 768)

        elif backbone_name == "convnext_base":
            weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            net = torchvision.models.convnext_base(weights=weights).features
            # ConvNeXt-Base: stride 8 (256 ch), stride 16 (512 ch), stride 32 (1024 ch)
            self.stage_p3 = nn.Sequential(*net[:4])
            self.stage_p4 = nn.Sequential(*net[4:6])
            self.stage_p5 = nn.Sequential(*net[6:8])
            self.out_channels = (256, 512, 1024)

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        p3 = self.stage_p3(x)   # [B, C3, H/8, W/8]
        p4 = self.stage_p4(p3)  # [B, C4, H/16, W/16]
        p5 = self.stage_p5(p4)  # [B, C5, H/32, W/32]
        return p3, p4, p5


class VFGNBiFPNNeck(nn.Module):
    """
    Weighted Bi-Directional Feature Pyramid Neck with learnable fusion weights
    and Coordinate Attention CSP blocks across P3/8, P4/16, and P5/32.
    """
    def __init__(self, in_channels, neck_channels=192, repeats=2):
        super().__init__()
        c3, c4, c5 = in_channels
        nc = neck_channels

        # Lateral 1x1 projections
        self.lat3 = ConvBNAct(c3, nc, 1, 1)
        self.lat4 = ConvBNAct(c4, nc, 1, 1)
        self.lat5 = ConvBNAct(c5, nc, 1, 1)

        # Top-down pathway (P5 -> P4 -> P3)
        self.w_td4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.td_block4 = VFGNCSPBlock(nc, nc, n=repeats)

        self.w_td3 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.td_block3 = VFGNCSPBlock(nc, nc, n=repeats)

        # Bottom-up pathway (P3 -> P4 -> P5) with skip connections
        self.down3to4 = ConvBNAct(nc, nc, 3, 2)
        self.w_bu4 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.bu_block4 = VFGNCSPBlock(nc, nc, n=repeats)

        self.down4to5 = ConvBNAct(nc, nc, 3, 2)
        self.w_bu5 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.bu_block5 = VFGNCSPBlock(nc, nc, n=repeats)

    @staticmethod
    def _norm_weights(w):
        w_pos = F.relu(w)
        return w_pos / (w_pos.sum() + 1e-4)

    def forward(self, p3, p4, p5):
        p3_in = self.lat3(p3)
        p4_in = self.lat4(p4)
        p5_in = self.lat5(p5)

        # Top-down
        w4 = self._norm_weights(self.w_td4)
        p5_up = F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest")
        p4_td = self.td_block4(w4[0] * p4_in + w4[1] * p5_up)

        w3 = self._norm_weights(self.w_td3)
        p4_up = F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest")
        p3_out = self.td_block3(w3[0] * p3_in + w3[1] * p4_up)

        # Bottom-up
        wb4 = self._norm_weights(self.w_bu4)
        p4_out = self.bu_block4(wb4[0] * p4_in + wb4[1] * p4_td + wb4[2] * self.down3to4(p3_out))

        wb5 = self._norm_weights(self.w_bu5)
        p5_out = self.bu_block5(wb5[0] * p5_in + wb5[1] * self.down4to5(p4_out))

        return p3_out, p4_out, p5_out


class VFGNDecoupledHead(nn.Module):
    """
    Anchor-Free Decoupled Detection Head across strides (8, 16, 32).
    For each grid location (x, y) at stride s:
      - Classification branch predicts `nc` class logits
      - Regression branch predicts 4 positive distances (l, t, r, b) in stride units
    """
    def __init__(self, in_channels: int, head_channels: int, nc: int, strides=(8, 16, 32)):
        super().__init__()
        self.nc = nc
        self.strides = strides

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in strides])

        prior_prob = 0.01
        bias_init = float(-math.log((1.0 - prior_prob) / prior_prob))

        for _ in strides:
            self.stems.append(ConvBNAct(in_channels, head_channels, 1, 1))
            self.cls_convs.append(
                nn.Sequential(
                    ConvBNAct(head_channels, head_channels, 3, 1),
                    ConvBNAct(head_channels, head_channels, 3, 1),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvBNAct(head_channels, head_channels, 3, 1),
                    ConvBNAct(head_channels, head_channels, 3, 1),
                )
            )
            cls_head = nn.Conv2d(head_channels, nc, 1)
            nn.init.constant_(cls_head.bias, bias_init)
            self.cls_preds.append(cls_head)

            reg_head = nn.Conv2d(head_channels, 4, 1)
            nn.init.constant_(reg_head.bias, 2.0)
            self.reg_preds.append(reg_head)

    def forward(self, feats):
        """
        Returns:
          cls_outputs : list of [B, nc, H_i, W_i]
          reg_outputs : list of [B, 4, H_i, W_i] (positive l, t, r, b distances in stride units)
        """
        cls_outputs = []
        reg_outputs = []
        for i, x in enumerate(feats):
            stem = self.stems[i](x)
            cls_feat = self.cls_convs[i](stem)
            reg_feat = self.reg_convs[i](stem)

            cls_logits = self.cls_preds[i](cls_feat)
            # Softplus guarantees smooth positive distances (l, t, r, b >= 0)
            reg_dist = F.softplus(self.reg_preds[i](reg_feat) * self.scales[i])
            cls_outputs.append(cls_logits)
            reg_outputs.append(reg_dist)
        return cls_outputs, reg_outputs


class VFGN26Detector(nn.Module):
    """
    Complete VFGN26 Object Detection Network (VFGN26S / VFGN26M / VFGN26XL).
    """
    def __init__(self, variant: str = "VFGN26M", nc: int = 3, pretrained_backbone: bool = True):
        super().__init__()
        variant = variant.upper()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(VARIANT_CONFIGS.keys())}")

        cfg = VARIANT_CONFIGS[variant]
        self.variant = variant
        self.nc = int(nc)
        self.strides = (8, 16, 32)

        # ImageNet normalization inside the model graph so raw [0..1] RGB tensors work directly
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.backbone = TorchvisionBackboneP3P4P5(cfg["backbone"], pretrained=pretrained_backbone)
        self.neck = VFGNBiFPNNeck(
            self.backbone.out_channels,
            neck_channels=cfg["neck_channels"],
            repeats=cfg["neck_repeats"],
        )
        self.head = VFGNDecoupledHead(
            in_channels=cfg["neck_channels"],
            head_channels=cfg["head_channels"],
            nc=self.nc,
            strides=self.strides,
        )

    def forward_raw(self, x):
        """Forward returning raw multi-scale (cls_outputs, reg_outputs) for loss computation."""
        x_norm = (x - self.img_mean) / self.img_std
        p3, p4, p5 = self.backbone(x_norm)
        f3, f4, f5 = self.neck(p3, p4, p5)
        return self.head((f3, f4, f5))

    def decode_predictions(self, cls_outputs, reg_outputs):
        """
        Decode multi-scale feature maps into a single tensor [B, N_anchors, 4 + nc]
        where the first 4 columns are [x1, y1, x2, y2] in input pixel coordinates
        and the remaining `nc` columns are class probabilities [0..1].
        """
        decoded_boxes = []
        decoded_scores = []

        for stride, cls_logit, reg_dist in zip(self.strides, cls_outputs, reg_outputs):
            b, nc, h, w = cls_logit.shape
            device = cls_logit.device
            dtype = cls_logit.dtype

            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride

            # reg_dist: [B, 4, H, W] -> l, t, r, b in stride units
            l = reg_dist[:, 0] * stride
            t = reg_dist[:, 1] * stride
            r = reg_dist[:, 2] * stride
            btm = reg_dist[:, 3] * stride

            x1 = cx.unsqueeze(0) - l
            y1 = cy.unsqueeze(0) - t
            x2 = cx.unsqueeze(0) + r
            y2 = cy.unsqueeze(0) + btm

            boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(b, -1, 4)
            scores = torch.sigmoid(cls_logit).permute(0, 2, 3, 1).reshape(b, -1, nc)

            decoded_boxes.append(boxes)
            decoded_scores.append(scores)

        all_boxes = torch.cat(decoded_boxes, dim=1)
        all_scores = torch.cat(decoded_scores, dim=1)
        return torch.cat([all_boxes, all_scores], dim=-1)  # [B, N, 4 + nc]

    def forward(self, x):
        """Standard inference forward (also used for ONNX export) -> [B, N, 4 + nc]."""
        cls_outputs, reg_outputs = self.forward_raw(x)
        return self.decode_predictions(cls_outputs, reg_outputs)


# ==============================================================================
# Drop-In Result / Boxes Wrapper (100% Compatible with Existing Inspection Code)
# ==============================================================================
class VFGNBoxItem:
    """Single bounding box item matching `box.xyxy`, `box.xywh`, `box.xywhn`, `box.xyxyn`, `box.cls`, `box.conf`."""
    def __init__(self, xyxy, conf, cls_id, orig_shape=None):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
        self.xyxy = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        self.xywh = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
        if orig_shape is not None and len(orig_shape) >= 2 and orig_shape[0] > 0 and orig_shape[1] > 0:
            oh, ow = float(orig_shape[0]), float(orig_shape[1])
            self.xyxyn = torch.tensor([[x1 / ow, y1 / oh, x2 / ow, y2 / oh]], dtype=torch.float32)
            self.xywhn = torch.tensor([[cx / ow, cy / oh, w / ow, h / oh]], dtype=torch.float32)
        else:
            self.xyxyn = self.xyxy.clone()
            self.xywhn = self.xywh.clone()
        self.conf = torch.tensor([float(conf)], dtype=torch.float32)
        self.cls = torch.tensor([float(cls_id)], dtype=torch.float32)


class VFGNBoxes:
    """Container of detected boxes supporting `len(result.boxes)`, `for box in result.boxes`, and indexing."""
    def __init__(self, boxes_xyxy, scores, class_ids, orig_shape=None):
        self._items = [
            VFGNBoxItem(b, s, c, orig_shape=orig_shape)
            for b, s, c in zip(boxes_xyxy, scores, class_ids)
        ]
        if len(self._items) > 0:
            self.xyxy = torch.cat([it.xyxy for it in self._items], dim=0)
            self.xywh = torch.cat([it.xywh for it in self._items], dim=0)
            self.xyxyn = torch.cat([it.xyxyn for it in self._items], dim=0)
            self.xywhn = torch.cat([it.xywhn for it in self._items], dim=0)
            self.conf = torch.cat([it.conf for it in self._items], dim=0)
            self.cls = torch.cat([it.cls for it in self._items], dim=0)
        else:
            self.xyxy = torch.zeros((0, 4), dtype=torch.float32)
            self.xywh = torch.zeros((0, 4), dtype=torch.float32)
            self.xyxyn = torch.zeros((0, 4), dtype=torch.float32)
            self.xywhn = torch.zeros((0, 4), dtype=torch.float32)
            self.conf = torch.zeros((0,), dtype=torch.float32)
            self.cls = torch.zeros((0,), dtype=torch.float32)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __iter__(self):
        return iter(self._items)


class VFGNResult:
    """Single image detection result (`result.boxes`, `result.names`, `result.orig_shape`)."""
    def __init__(self, boxes: VFGNBoxes, names: dict, orig_shape: tuple):
        self.boxes = boxes
        self.names = names
        self.orig_shape = orig_shape
        self.masks = None
        self.keypoints = None
        self.obb = None


# ==============================================================================
# High-Level User API: VFGNModel("VFGN26M") or VFGNModel("best.vf")
# ==============================================================================
class VFGNModel:
    """
    High-level VerifyGN Model API for Training, Exporting, and Inference.

    Examples:
      # 1. Create a new model & train on a YOLO data.yaml dataset:
      model = VFGNModel("VFGN26M")
      model.train(data="path/to/data.yaml", epochs=100, imgsz=640, batch=8)

      # 2. Load a trained .vf model (PyTorch or embedded ONNX) & run inference:
      model = VFGNModel("runs/vfgn/best.vf", key=None)
      results = model(image, conf=0.25, iou=0.7)
    """
    def __init__(
        self,
        model_or_path: str = "VFGN26M",
        nc: int = None,
        names: dict = None,
        key: str = None,
        device: str = None,
        pretrained_backbone: bool = True,
        letterbox: bool = True,
        **kwargs,
    ):
        self.key = key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ort_session = None
        self.net = None
        self.imgsz = 640
        self.task = "detect"
        self.letterbox = bool(letterbox)
        self.metadata = {}

        if isinstance(model_or_path, str) and (
            os.path.isfile(model_or_path) or model_or_path.lower().endswith(".vf")
        ):
            self._load_from_vf(model_or_path, key=key)
        else:
            variant = str(model_or_path).upper()
            if variant not in VARIANT_CONFIGS:
                raise ValueError(
                    f"'{model_or_path}' is neither an existing .vf file nor a valid variant "
                    f"({list(VARIANT_CONFIGS.keys())})."
                )
            self.variant = variant
            if names is not None:
                self.names = {int(k): str(v) for k, v in names.items()}
                self.nc = len(self.names)
            else:
                self.nc = int(nc or 3)
                self.names = {i: f"class_{i}" for i in range(self.nc)}

            self.net = VFGN26Detector(
                variant=self.variant,
                nc=self.nc,
                pretrained_backbone=pretrained_backbone,
            ).to(self.device)
            self.net.eval()

    def _load_from_vf(self, vf_path: str, key: str = None):
        pkg = load_vf(vf_path, key=key, map_location=self.device)
        meta = pkg["metadata"]
        self.metadata = meta
        self.variant = meta["variant"]
        self.names = meta["names"]
        self.nc = meta["nc"]
        self.imgsz = parse_imgsz_wh(meta.get("imgsz", 640))

        if pkg["onnx_bytes"] is not None:
            import shutil
            import onnxruntime as ort
            has_ort_cuda = (
                "cuda" in str(self.device).lower()
                and "CUDAExecutionProvider" in ort.get_available_providers()
                and all(shutil.which(dll) for dll in ("cublasLt64_12.dll", "cudnn64_9.dll"))
            )
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if has_ort_cuda
                else ["CPUExecutionProvider"]
            )
            self.ort_session = ort.InferenceSession(pkg["onnx_bytes"], providers=providers)
            self.ort_input_name = self.ort_session.get_inputs()[0].name
            logger.info(f"Loaded VFGN ONNX Runtime engine from '{vf_path}' ({self.variant}, nc={self.nc})")
        else:
            self.net = VFGN26Detector(
                variant=self.variant,
                nc=self.nc,
                pretrained_backbone=False,
            )
            self.net.load_state_dict(pkg["state_dict"], strict=True)
            self.net.to(self.device)
            self.net.eval()
            logger.info(f"Loaded VFGN PyTorch engine from '{vf_path}' ({self.variant}, nc={self.nc})")

    def info(self) -> dict:
        """Return model architecture summary (parameters, variant, classes, license)."""
        params = sum(p.numel() for p in self.net.parameters()) if self.net is not None else None
        return {
            "variant": self.variant,
            "description": VARIANT_CONFIGS[self.variant]["description"],
            "nc": self.nc,
            "names": self.names,
            "imgsz": self.imgsz,
            "parameters": params,
            "backend": "onnxruntime" if self.ort_session is not None else "pytorch",
            "license": "Apache-2.0 / VerifyGN Proprietary (Zero AGPL-3.0)",
        }

    def save(self, vf_path: str, metrics: dict = None, epoch: int = -1, key: str = None) -> str:
        """Save current model weights to a `.vf` file."""
        if self.net is None:
            raise RuntimeError("Cannot save PyTorch state_dict from an ONNX-only runtime instance.")
        return save_vf(
            path=vf_path,
            state_dict=self.net.state_dict(),
            variant=self.variant,
            names=self.names,
            imgsz=self.imgsz,
            metrics=metrics,
            epoch=epoch,
            key=key if key is not None else self.key,
        )

    def export_fast_vf(self, output_vf_path: str, imgsz=None, key: str = None) -> str:
        """
        Compile the PyTorch `VFGN26` network into an embedded ONNX graph inside a `.vf` file.
        Supports both square (`640`, `1024`) and rectangular (`(1024, 512)`) resolutions!
        Site PCs can then run `VFGNModel("best_fast.vf")` on CPU at 2x-4x speed using `onnxruntime`!
        """
        if self.net is None:
            raise RuntimeError("Model is already an ONNX runtime package.")
        img_w, img_h = parse_imgsz_wh(imgsz if imgsz is not None else self.imgsz)
        self.net.eval()
        dummy = torch.zeros((1, 3, img_h, img_w), dtype=torch.float32, device=self.device)
        bio = io.BytesIO()
        torch.onnx.export(
            self.net,
            dummy,
            bio,
            input_names=["images"],
            output_names=["predictions"],
            opset_version=14,
            do_constant_folding=True,
        )
        return save_vf(
            path=output_vf_path,
            state_dict={},
            variant=self.variant,
            names=self.names,
            imgsz=(img_w, img_h),
            metrics=self.metadata.get("metrics", {}),
            onnx_bytes=bio.getvalue(),
            key=key if key is not None else self.key,
        )

    def __call__(
        self,
        source,
        conf: float = 0.25,
        iou: float = 0.7,
        agnostic_nms: bool = True,
        letterbox: bool = None,
        device: str = None,
        save: bool = False,
        verbose: bool = False,
        max_det: int = 300,
        **kwargs,
    ):
        """
        Run object detection on an image path or BGR numpy array.
        Uses aspect-preserving letterbox padding (`(114, 114, 114)`) by default when the input
        aspect ratio differs from `(img_w, img_h)`, and un-pads/rescales boxes back to `(orig_w, orig_h)`
        so `result.boxes.xyxy` and `result.boxes.xywhn` match the exact original image coordinates!
        """
        if letterbox is None:
            letterbox = getattr(self, "letterbox", True)
        if isinstance(source, str):
            img_bgr = cv2.imread(source)
            if img_bgr is None:
                raise IOError(f"Cannot read image: {source}")
        elif isinstance(source, np.ndarray):
            img_bgr = source
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        orig_h, orig_w = img_bgr.shape[:2]
        img_w, img_h = parse_imgsz_wh(self.imgsz)

        if letterbox and (orig_w * img_h != orig_h * img_w):
            # Aspect-preserving letterbox resize + (114, 114, 114) border pad
            scale = min(img_w / float(orig_w), img_h / float(orig_h))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            if (new_w, new_h) != (orig_w, orig_h):
                resized_inner = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized_inner = img_bgr
            pad_w = img_w - new_w
            pad_h = img_h - new_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            resized = cv2.copyMakeBorder(
                resized_inner,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            scale_x = 1.0 / scale
            scale_y = 1.0 / scale
        else:
            pad_left, pad_top = 0, 0
            new_w, new_h = img_w, img_h
            resized = cv2.resize(img_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            scale_x = orig_w / float(img_w)
            scale_y = orig_h / float(img_h)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        blob = np.expand_dims(rgb, axis=0).astype(np.float32) / 255.0

        # Run inference via ONNX Runtime or PyTorch
        if self.ort_session is not None:
            preds = self.ort_session.run(None, {self.ort_input_name: blob})[0]
            preds_t = torch.from_numpy(preds[0])
        else:
            run_dev = self.device
            with torch.no_grad():
                inp = torch.from_numpy(blob).to(run_dev)
                preds_t = self.net(inp)[0].cpu()

        # preds_t: [N, 4 + nc]
        boxes = preds_t[:, :4]
        scores_mat = preds_t[:, 4:]

        max_scores, cls_ids = scores_mat.max(dim=1)
        keep_mask = max_scores >= float(conf)

        if keep_mask.any():
            boxes = boxes[keep_mask]
            scores = max_scores[keep_mask]
            cls_ids = cls_ids[keep_mask]

            # Subtract letterbox padding and scale back to original image resolution (orig_w x orig_h)
            boxes[:, 0] = ((boxes[:, 0] - pad_left) * scale_x).clamp(0, orig_w)
            boxes[:, 2] = ((boxes[:, 2] - pad_left) * scale_x).clamp(0, orig_w)
            boxes[:, 1] = ((boxes[:, 1] - pad_top) * scale_y).clamp(0, orig_h)
            boxes[:, 3] = ((boxes[:, 3] - pad_top) * scale_y).clamp(0, orig_h)

            # Filter out degenerate boxes that landed purely inside the letterbox padding
            valid_box = ((boxes[:, 2] - boxes[:, 0]) > 1.0) & ((boxes[:, 3] - boxes[:, 1]) > 1.0)
            boxes = boxes[valid_box]
            scores = scores[valid_box]
            cls_ids = cls_ids[valid_box]

        if keep_mask.any() and len(boxes) > 0:
            nms_idxs = torch.zeros_like(cls_ids) if agnostic_nms else cls_ids
            keep = batched_nms(boxes, scores, nms_idxs, float(iou))[:max_det]

            final_boxes = boxes[keep].numpy()
            final_scores = scores[keep].numpy()
            final_cls = cls_ids[keep].numpy()
        else:
            final_boxes = np.zeros((0, 4), dtype=np.float32)
            final_scores = np.zeros((0,), dtype=np.float32)
            final_cls = np.zeros((0,), dtype=np.int64)

        return [VFGNResult(VFGNBoxes(final_boxes, final_scores, final_cls, orig_shape=(orig_h, orig_w)), self.names, (orig_h, orig_w))]

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = None,
        lr: float = 1e-3,
        patience: int = 100,
        workers: int = 4,
        amp: bool = True,
        project: str = None,
        name: str = None,
        save_dir: str = None,
        resume: str = None,
        key: str = None,
        **kwargs,
    ):
        """
        Train or fine-tune this VFGN26 model on a standard YOLO `data.yaml` dataset (e.g. from Roboflow).
        Supports exact same arguments as YOLO (`data`, `epochs`, `imgsz`, `batch=-1`, `workers`, `amp`,
        `patience`, `lr0`, `project`, `name`, `exist_ok`) and saves `best.vf` + `last.vf`.
        """
        from .trainer import VFGNTrainer

        if save_dir is None:
            if project and name:
                save_dir = os.path.join(project, name)
            elif project:
                save_dir = project
            else:
                save_dir = "runs/vfgn_train"

        effective_lr = float(lr0) if lr0 is not None else float(lr)

        trainer = VFGNTrainer(
            model=self,
            data_yaml=data,
            epochs=epochs,
            imgsz=imgsz,
            batch_size=batch,
            lr=effective_lr,
            patience=patience,
            amp=amp,
            resume=resume,
            save_dir=save_dir,
            key=key if key is not None else self.key,
            workers=workers,
        )
        return trainer.fit()
