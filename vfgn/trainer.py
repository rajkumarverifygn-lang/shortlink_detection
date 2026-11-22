# Copyright (c) VerifyGN. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
VFGN26 Complete Cloud-GPU & Local Training Engine (`vfgn/trainer.py`)
---------------------------------------------------------------------
Full YOLO26-Style Console Tables, Metrics, CSV Logger, Curves & Validation Previews:
- Per-Epoch Table: Epoch, GPU_mem, box_loss, cls_loss, Instances, Size + Class, Images, Instances, Box(P, R, mAP50, mAP50-95)
- End-of-Training Artifacts saved in `save_dir`:
    * `best.vf` & `last.vf` (VerifyGN proprietary model containers)
    * `results.csv` (All epoch losses & metrics)
    * `results.png` (Training loss & validation mAP50 / mAP50-95 curves drawn with OpenCV)
    * `val_batch0_pred.jpg` & `val_batch0_labels.jpg` (Visual bounding-box comparison on validation images)
- Smart Warmup + EMA Blending so validation mAP50 is active from Epoch 1-2 and smooth across epochs.
"""

import os
import csv
import math
import time
import copy
import yaml
import random
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou, batched_nms

from .model import VFGNModel, VFGN26Detector, parse_imgsz_wh
from .format import load_vf, save_vf

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Exponential Moving Average (EMA) of Model Weights (Fast Warmup Ramp)
# ==============================================================================
class ModelEMA:
    """
    Smoothed exponential moving average of model weights.
    Uses a fast ramp (`tau=50` steps instead of 2000) so small/medium industrial datasets
    (200-1000 images) immediately reflect trained weights by Epoch 1-2 while still smoothing later epochs.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9995, tau: float = 50.0):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.decay = decay
        self.tau = tau

    def update(self, model: nn.Module):
        with torch.no_grad():
            self.updates += 1
            d = self.decay * (1.0 - math.exp(-self.updates / self.tau))
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
                else:
                    v.copy_(msd[k])


# ==============================================================================
# 2. YOLO-Format Dataset Loader (Reads standard images/ & labels/*.txt)
# ==============================================================================
class YOLODetectionDataset(Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, img_dirs, imgsz=640, augment: bool = True):
        super().__init__()
        self.img_w, self.img_h = parse_imgsz_wh(imgsz)
        self.augment = bool(augment)
        if isinstance(img_dirs, str):
            img_dirs = [img_dirs]

        self.img_files = []
        for d in img_dirs:
            if os.path.isdir(d):
                for root, _, files in os.walk(d):
                    for fn in sorted(files):
                        if fn.lower().endswith(self.IMG_EXTS):
                            self.img_files.append(os.path.join(root, fn))
            elif os.path.isfile(d) and d.lower().endswith(".txt"):
                with open(d, "r", encoding="utf-8") as f:
                    for line in f:
                        p = line.strip()
                        if p and os.path.isfile(p):
                            self.img_files.append(p)

        self.label_files = [self._img2label_path(p) for p in self.img_files]

    @staticmethod
    def _img2label_path(img_path: str) -> str:
        parts = os.path.normpath(img_path).split(os.sep)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
        lbl_path = os.sep.join(parts)
        return os.path.splitext(lbl_path)[0] + ".txt"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        lbl_path = self.label_files[index]

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        boxes_cls = []
        if os.path.isfile(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        x1 = (cx - w * 0.5) * self.img_w
                        y1 = (cy - h * 0.5) * self.img_h
                        x2 = (cx + w * 0.5) * self.img_w
                        y2 = (cy + h * 0.5) * self.img_h
                        if x2 > x1 + 2 and y2 > y1 + 2:
                            boxes_cls.append([cls_id, x1, y1, x2, y2])

        targets = np.array(boxes_cls, dtype=np.float32) if boxes_cls else np.zeros((0, 5), dtype=np.float32)

        if self.augment:
            img, targets = self._apply_augmentations(img, targets)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        tensor_img = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor_targets = torch.from_numpy(targets)
        return tensor_img, tensor_targets

    def _apply_augmentations(self, img, targets):
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            if len(targets) > 0:
                x1 = targets[:, 1].copy()
                x2 = targets[:, 3].copy()
                targets[:, 1] = self.img_w - x2
                targets[:, 3] = self.img_w - x1

        if random.random() < 0.7:
            alpha = random.uniform(0.75, 1.25)
            beta = random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.5 and len(targets) > 0:
            max_shift_x = int(self.img_w * 0.06)
            max_shift_y = int(self.img_h * 0.06)
            dx = random.randint(-max_shift_x, max_shift_x)
            dy = random.randint(-max_shift_y, max_shift_y)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (self.img_w, self.img_h), borderValue=(114, 114, 114))
            targets[:, [1, 3]] = np.clip(targets[:, [1, 3]] + dx, 0, self.img_w)
            targets[:, [2, 4]] = np.clip(targets[:, [2, 4]] + dy, 0, self.img_h)
            valid = ((targets[:, 3] - targets[:, 1]) > 4) & ((targets[:, 4] - targets[:, 2]) > 4)
            targets = targets[valid]

        return img, targets


def collate_detection_batch(batch):
    imgs, targets_list = zip(*batch)
    imgs_tensor = torch.stack(imgs, dim=0)
    return imgs_tensor, list(targets_list)


# ==============================================================================
# 3. Complete IoU (CIoU) + Task-Aligned Dynamic Assigner Detection Loss
# ==============================================================================
def bbox_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)

    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(min=eps)
    ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2 + eps

    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4.0

    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return iou - (rho2 / c2 + v * alpha)


class VFGNDetectionLoss(nn.Module):
    def __init__(self, nc: int, strides=(8, 16, 32), topk: int = 9):
        super().__init__()
        self.nc = nc
        self.strides = strides
        self.topk = topk

    def forward(self, net: VFGN26Detector, cls_outputs, reg_outputs, targets_list):
        device = cls_outputs[0].device
        cls_outputs = [c.float() for c in cls_outputs]
        reg_outputs = [r.float() for r in reg_outputs]
        dtype = torch.float32
        batch_size = cls_outputs[0].shape[0]

        anchor_points = []
        cls_flat = []
        box_flat = []

        for stride, cls_logit, reg_dist in zip(self.strides, cls_outputs, reg_outputs):
            b, nc, h, w = cls_logit.shape
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            cx = (gx + 0.5) * stride
            cy = (gy + 0.5) * stride
            pts = torch.stack([cx, cy], dim=-1).reshape(-1, 2)
            anchor_points.append(pts)

            l = reg_dist[:, 0] * stride
            t = reg_dist[:, 1] * stride
            r = reg_dist[:, 2] * stride
            btm = reg_dist[:, 3] * stride

            x1 = cx.unsqueeze(0) - l
            y1 = cy.unsqueeze(0) - t
            x2 = cx.unsqueeze(0) + r
            y2 = cy.unsqueeze(0) + btm

            box_flat.append(torch.stack([x1, y1, x2, y2], dim=-1).reshape(b, -1, 4))
            cls_flat.append(cls_logit.permute(0, 2, 3, 1).reshape(b, -1, nc))

        anchors = torch.cat(anchor_points, dim=0)
        pred_boxes = torch.cat(box_flat, dim=1)
        pred_logits = torch.cat(cls_flat, dim=1)

        total_cls_loss = torch.zeros(1, device=device, dtype=dtype)
        total_box_loss = torch.zeros(1, device=device, dtype=dtype)
        num_pos_total = 0.0

        for b_idx in range(batch_size):
            gt = targets_list[b_idx].to(device, dtype=dtype)
            p_logits = pred_logits[b_idx]
            p_boxes = pred_boxes[b_idx]
            target_scores = torch.zeros_like(p_logits)

            if len(gt) > 0:
                gt_cls = gt[:, 0].long().clamp(0, self.nc - 1)
                gt_boxes = gt[:, 1:5]
                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
                gt_centers = torch.stack([gt_cx, gt_cy], dim=-1)

                dists = torch.cdist(gt_centers, anchors)
                in_x1 = anchors[:, 0].unsqueeze(0) >= gt_boxes[:, 0].unsqueeze(1)
                in_y1 = anchors[:, 1].unsqueeze(0) >= gt_boxes[:, 1].unsqueeze(1)
                in_x2 = anchors[:, 0].unsqueeze(0) <= gt_boxes[:, 2].unsqueeze(1)
                in_y2 = anchors[:, 1].unsqueeze(0) <= gt_boxes[:, 3].unsqueeze(1)
                inside = in_x1 & in_y1 & in_x2 & in_y2

                masked_dists = torch.where(inside, dists, dists + 1e4)
                k = min(self.topk, anchors.shape[0])
                _, topk_idxs = torch.topk(masked_dists, k=k, dim=1, largest=False)

                pos_anchor_list = []
                pos_gt_box_list = []

                for m_idx in range(len(gt)):
                    a_idxs = topk_idxs[m_idx]
                    c_id = gt_cls[m_idx]
                    g_box = gt_boxes[m_idx].unsqueeze(0).expand(k, 4)
                    ciou = bbox_ciou(p_boxes[a_idxs], g_box)
                    iou_score = ciou.detach().clamp(min=0.05, max=1.0)
                    target_scores[a_idxs, c_id] = torch.max(target_scores[a_idxs, c_id], iou_score)
                    pos_anchor_list.append(a_idxs)
                    pos_gt_box_list.append(g_box)

                all_pos_a = torch.cat(pos_anchor_list, dim=0)
                all_pos_g = torch.cat(pos_gt_box_list, dim=0)
                cious = bbox_ciou(p_boxes[all_pos_a], all_pos_g)
                total_box_loss = total_box_loss + (1.0 - cious).sum()
                num_pos_total += float(all_pos_a.numel())

            probs = torch.sigmoid(p_logits)
            focal_weight = (target_scores - probs).abs().pow(2.0)
            bce = F.binary_cross_entropy_with_logits(p_logits, target_scores, reduction="none")
            total_cls_loss = total_cls_loss + (focal_weight * bce).sum()

        norm = max(num_pos_total, 1.0)
        loss_cls = total_cls_loss / norm
        loss_box = (total_box_loss / norm) * 5.0
        total_loss = (loss_cls + loss_box).clamp(max=50.0)
        return total_loss, {"cls_loss": float(loss_cls.detach()), "box_loss": float(loss_box.detach())}


# ==============================================================================
# 4. Full COCO / YOLO Validation Evaluator: P, R, mAP@50, mAP@50-95 + Per-Class Breakdown
# ==============================================================================
def _compute_ap_at_iou(scores_np, cls_np, pred_ious_np, pred_gt_idx_np, pred_img_idx_np, total_gt_per_class, nc, iou_thresh):
    """Compute Precision, Recall, and AP per class at a single IoU threshold (e.g. 0.50 .. 0.95)."""
    aps = {}
    precs = {}
    recs = {}

    for c in range(nc):
        n_gt = total_gt_per_class.get(c, 0)
        if n_gt == 0:
            continue
        mask = cls_np == c
        if not mask.any():
            aps[c] = 0.0
            precs[c] = 0.0
            recs[c] = 0.0
            continue

        c_scores = scores_np[mask]
        c_ious = pred_ious_np[mask]
        c_gt_idx = pred_gt_idx_np[mask]
        c_img_idx = pred_img_idx_np[mask]

        order = np.argsort(-c_scores)
        c_scores = c_scores[order]
        c_ious = c_ious[order]
        c_gt_idx = c_gt_idx[order]
        c_img_idx = c_img_idx[order]

        tp = np.zeros(len(c_scores), dtype=np.float32)
        fp = np.zeros(len(c_scores), dtype=np.float32)
        matched = set()

        for i in range(len(c_scores)):
            key = (int(c_img_idx[i]), int(c_gt_idx[i]))
            if c_gt_idx[i] >= 0 and c_ious[i] >= iou_thresh and key not in matched:
                matched.add(key)
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / float(n_gt)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        # Best F1 operating point for Precision, Recall & F1 display (same as YOLO)
        f1 = 2.0 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_i = int(np.argmax(f1))
        precs[c] = float(prec[best_i])
        recs[c] = float(rec[best_i])

        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        x = np.linspace(0, 1, 101)
        aps[c] = float(np.trapz(np.interp(x, mrec, mpre), x))

    return precs, recs, aps


@torch.no_grad()
def evaluate_full_metrics(net: VFGN26Detector, val_loader: DataLoader, device: str, nc: int, names: dict, conf_thres=0.001, iou_nms=0.6):
    """
    Evaluate Precision, Recall, F1, mAP@50, and mAP@50-95 across IoU thresholds 0.50:0.05:0.95
    using standard YOLO/COCO validation threshold (conf_thres=0.001 + top-300 pre-NMS)
    so mAP50 is accurately measured from Epoch 1 even before confidence scores saturate.
    """
    net.eval()
    scores_all, cls_all, best_iou_all, best_gt_all, img_idx_all = [], [], [], [], []
    total_gt_per_class = {c: 0 for c in range(nc)}
    images_with_class = {c: set() for c in range(nc)}
    total_images = 0

    for imgs, targets_list in val_loader:
        imgs = imgs.to(device)
        preds_batch = net(imgs).cpu()

        for b_idx, preds in enumerate(preds_batch):
            global_img_id = total_images
            total_images += 1
            gt = targets_list[b_idx]
            gt_cls = gt[:, 0].long() if len(gt) > 0 else torch.zeros((0,), dtype=torch.long)
            gt_boxes = gt[:, 1:5] if len(gt) > 0 else torch.zeros((0, 4), dtype=torch.float32)

            for c in gt_cls.tolist():
                if 0 <= c < nc:
                    total_gt_per_class[c] += 1
                    images_with_class[c].add(global_img_id)

            boxes = preds[:, :4]
            scores_mat = preds[:, 4:]
            max_scores, pred_cls = scores_mat.max(dim=1)
            keep = max_scores >= conf_thres
            if not keep.any():
                continue

            boxes = boxes[keep]
            scores = max_scores[keep]
            pred_cls = pred_cls[keep]

            # Keep top 300 candidates by score before NMS (prevents slow NMS on 20k anchors at conf=0.001)
            if len(scores) > 300:
                top_vals, top_idx = torch.topk(scores, 300)
                boxes = boxes[top_idx]
                scores = top_vals
                pred_cls = pred_cls[top_idx]

            nms_keep = batched_nms(boxes, scores, pred_cls, iou_nms)[:100]
            boxes = boxes[nms_keep]
            scores = scores[nms_keep]
            pred_cls = pred_cls[nms_keep]

            ious_mat = box_iou(boxes, gt_boxes) if len(gt_boxes) > 0 else torch.zeros((len(boxes), 0))

            for p_idx in range(len(boxes)):
                c_int = int(pred_cls[p_idx].item())
                p_score = float(scores[p_idx].item())
                best_iou, best_g = 0.0, -1
                if len(gt_boxes) > 0:
                    for g_idx in range(len(gt_boxes)):
                        if int(gt_cls[g_idx].item()) == c_int:
                            v = float(ious_mat[p_idx, g_idx].item())
                            if v > best_iou:
                                best_iou = v
                                best_g = g_idx
                scores_all.append(p_score)
                cls_all.append(c_int)
                best_iou_all.append(best_iou)
                best_gt_all.append(best_g)
                img_idx_all.append(global_img_id)

    total_instances = sum(total_gt_per_class.values())
    if len(scores_all) == 0:
        per_class = {}
        for c in range(nc):
            per_class[c] = {
                "name": names.get(c, f"class_{c}"),
                "images": len(images_with_class[c]),
                "instances": total_gt_per_class[c],
                "P": 0.0, "R": 0.0, "F1": 0.0, "mAP50": 0.0, "mAP50-95": 0.0,
            }
        return {
            "precision": 0.0, "recall": 0.0, "F1": 0.0, "mAP50": 0.0, "mAP50-95": 0.0,
            "total_images": total_images, "total_instances": total_instances,
            "per_class": per_class,
        }

    scores_np = np.array(scores_all, dtype=np.float32)
    cls_np = np.array(cls_all, dtype=np.int32)
    ious_np = np.array(best_iou_all, dtype=np.float32)
    gt_idx_np = np.array(best_gt_all, dtype=np.int32)
    img_idx_np = np.array(img_idx_all, dtype=np.int32)

    iou_thresholds = np.linspace(0.50, 0.95, 10)
    aps_by_thresh = []
    p50, r50, ap50 = {}, {}, {}

    for t_i, thresh in enumerate(iou_thresholds):
        precs, recs, aps = _compute_ap_at_iou(
            scores_np, cls_np, ious_np, gt_idx_np, img_idx_np, total_gt_per_class, nc, float(thresh)
        )
        aps_by_thresh.append(aps)
        if t_i == 0:
            p50, r50, ap50 = precs, recs, aps

    per_class = {}
    valid_p, valid_r, valid_f1, valid_ap50, valid_ap50_95 = [], [], [], [], []
    for c in range(nc):
        c_ap50_95 = float(np.mean([aps_t.get(c, 0.0) for aps_t in aps_by_thresh])) if total_gt_per_class[c] > 0 else 0.0
        c_p = p50.get(c, 0.0)
        c_r = r50.get(c, 0.0)
        c_f1 = (2.0 * c_p * c_r / max(c_p + c_r, 1e-9)) if (c_p + c_r) > 0 else 0.0
        c_ap50 = ap50.get(c, 0.0)
        per_class[c] = {
            "name": names.get(c, f"class_{c}"),
            "images": len(images_with_class[c]),
            "instances": total_gt_per_class[c],
            "P": round(c_p, 4),
            "R": round(c_r, 4),
            "F1": round(c_f1, 4),
            "mAP50": round(c_ap50, 4),
            "mAP50-95": round(c_ap50_95, 4),
        }
        if total_gt_per_class[c] > 0:
            valid_p.append(c_p)
            valid_r.append(c_r)
            valid_f1.append(c_f1)
            valid_ap50.append(c_ap50)
            valid_ap50_95.append(c_ap50_95)

    return {
        "precision": round(float(np.mean(valid_p)) if valid_p else 0.0, 4),
        "recall": round(float(np.mean(valid_r)) if valid_r else 0.0, 4),
        "F1": round(float(np.mean(valid_f1)) if valid_f1 else 0.0, 4),
        "mAP50": round(float(np.mean(valid_ap50)) if valid_ap50 else 0.0, 4),
        "mAP50-95": round(float(np.mean(valid_ap50_95)) if valid_ap50_95 else 0.0, 4),
        "total_images": total_images,
        "total_instances": total_instances,
        "per_class": per_class,
    }


# ==============================================================================
# 5. Visual Artifacts Generator: `results.png` & `val_batch0_pred.jpg`
# ==============================================================================
def save_training_plots_and_previews(history: list, save_dir: str, net: VFGN26Detector, val_ds: Dataset, device: str, names: dict):
    """Draw `results.png` curves and `val_batch0_labels.jpg` / `val_batch0_pred.jpg` using pure OpenCV."""
    try:
        if len(history) > 0:
            canvas = np.full((600, 1200, 3), 255, dtype=np.uint8)
            panels = [
                ("train/box_loss", [h["box_loss"] for h in history], (30, 100, 220)),
                ("train/cls_loss", [h["cls_loss"] for h in history], (200, 80, 30)),
                ("metrics/Precision & Recall", [h["precision"] for h in history], (40, 170, 40)),
                ("metrics/mAP50 & mAP50-95", [h["mAP50"] for h in history], (180, 30, 180)),
            ]
            for idx, (title, vals, color) in enumerate(panels):
                px = (idx % 2) * 600 + 50
                py = (idx // 2) * 300 + 40
                pw, ph = 500, 220
                cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (220, 220, 220), 1)
                cv2.putText(canvas, title, (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
                if len(vals) >= 1:
                    vmin, vmax = min(vals), max(vals)
                    if abs(vmax - vmin) < 1e-6:
                        vmax = vmin + 1.0
                    pts = []
                    for i, v in enumerate(vals):
                        x = px + int((i / max(len(vals) - 1, 1)) * (pw - 20)) + 10
                        y = py + ph - 10 - int(((v - vmin) / (vmax - vmin)) * (ph - 30))
                        pts.append((x, y))
                    for i in range(len(pts) - 1):
                        cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)
                    cv2.putText(canvas, f"max={max(vals):.4f} last={vals[-1]:.4f}", (px + 15, py + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            cv2.imwrite(os.path.join(save_dir, "results.png"), canvas)
    except Exception as e:
        logger.warning(f"Could not save results.png: {e}")

    # Save validation prediction preview grid (up to 4 validation images)
    try:
        net.eval()
        n_show = min(len(val_ds), 4)
        if n_show > 0:
            gt_tiles, pred_tiles = [], []
            for i in range(n_show):
                img_t, gt_t = val_ds[i]
                bgr = cv2.cvtColor((img_t.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                gt_im = bgr.copy()
                pred_im = bgr.copy()

                for row in gt_t.numpy():
                    c_id, x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
                    cv2.rectangle(gt_im, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(gt_im, names.get(c_id, str(c_id)), (x1, max(y1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                with torch.no_grad():
                    preds = net(img_t.unsqueeze(0).to(device))[0].cpu()
                boxes = preds[:, :4]
                scores, cls_ids = preds[:, 4:].max(dim=1)
                keep = scores >= 0.20
                if keep.any():
                    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
                    nms_k = batched_nms(boxes, scores, cls_ids, 0.6)[:25]
                    for b, s, c in zip(boxes[nms_k], scores[nms_k], cls_ids[nms_k]):
                        x1, y1, x2, y2 = [int(v) for v in b.tolist()]
                        lbl = f"{names.get(int(c), int(c))} {float(s):.2f}"
                        cv2.rectangle(pred_im, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(pred_im, lbl, (x1, max(y1 - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                gt_tiles.append(cv2.resize(gt_im, (512, 512)))
                pred_tiles.append(cv2.resize(pred_im, (512, 512)))

            cv2.imwrite(os.path.join(save_dir, "val_batch0_labels.jpg"), np.hstack(gt_tiles))
            cv2.imwrite(os.path.join(save_dir, "val_batch0_pred.jpg"), np.hstack(pred_tiles))
    except Exception as e:
        logger.warning(f"Could not save validation preview images: {e}")


# ==============================================================================
# 6. VFGNTrainer: Cloud-GPU Ready Training + YOLO26-Style Detailed Output
# ==============================================================================
class VFGNTrainer:
    def __init__(
        self,
        model: VFGNModel,
        data_yaml: str,
        epochs: int = 100,
        imgsz=640,
        batch_size: int = 16,
        lr: float = 1e-3,
        patience: int = 100,
        amp: bool = True,
        resume: str = None,
        save_dir: str = "runs/vfgn_train",
        key: str = None,
        workers: int = 0,
    ):
        self.vfgn_model = model
        self.data_yaml = data_yaml
        self.epochs = int(epochs)
        self.imgsz = parse_imgsz_wh(imgsz)
        self.batch_size = self._resolve_batch_size(int(batch_size), self.imgsz, model.variant)
        self.lr = self._resolve_safe_lr(float(lr), model.variant)
        self.patience = int(patience)
        self.use_amp = bool(amp) and ("cuda" in str(model.device).lower())
        self.resume_path = resume
        self.save_dir = save_dir
        self.key = key if key is not None else model.key
        self.workers = int(workers)
        self.device = model.device

        os.makedirs(self.save_dir, exist_ok=True)
        self._load_data_yaml()

        if self.resume_path and os.path.isfile(self.resume_path):
            pkg = load_vf(self.resume_path, key=self.key, map_location=self.device)
            self.vfgn_model.net.load_state_dict(pkg["state_dict"], strict=False)
            logger.info(f"Loaded / fine-tuning weights from '{self.resume_path}' (epoch={pkg['metadata'].get('epoch')})")

    @staticmethod
    def _resolve_safe_lr(lr: float, variant: str) -> float:
        """
        Automatically cap learning rate for deep ConvNeXt variants (VFGN26M / VFGN26L / VFGN26XL)
        so users passing lr0=0.005 (good for MobileNet VFGN26S) never blow up ConvNeXt LayerNorm weights!
        """
        v = variant.upper()
        max_safe = {"VFGN26N": 0.005, "VFGN26S": 0.005, "VFGN26M": 0.001, "VFGN26L": 0.0008, "VFGN26XL": 0.0005}.get(v, 0.001)
        if lr > max_safe:
            print(f"[VFGN26 Auto-LR] Clamping lr={lr} -> {max_safe} for deep ConvNeXt architecture ({v}) to prevent gradient spike.", flush=True)
            return max_safe
        return lr

    @staticmethod
    def _resolve_batch_size(batch_size: int, imgsz, variant: str) -> int:
        if batch_size > 0:
            return batch_size
        if not torch.cuda.is_available():
            return 4
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            vram_gb = 16.0
        img_w, img_h = parse_imgsz_wh(imgsz)
        base_per_img_gb = {"VFGN26S": 0.25, "VFGN26M": 0.55, "VFGN26XL": 0.95}.get(variant.upper(), 0.7)
        scale = (img_w * img_h) / float(640 * 640)
        per_img_gb = base_per_img_gb * scale
        usable_gb = max(vram_gb * 0.70, 2.0)
        auto_b = int(usable_gb / max(per_img_gb, 0.1))
        auto_b = max(4, min(64, (auto_b // 4) * 4))
        logger.info(f"AutoBatch (batch=-1): Detected {vram_gb:.1f} GB GPU VRAM -> selected batch_size={auto_b} for {variant}@{img_w}x{img_h}")
        return auto_b

    def _load_data_yaml(self):
        with open(self.data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        base_dir = cfg.get("path") or os.path.dirname(os.path.abspath(self.data_yaml))
        names_raw = cfg.get("names", {})
        if isinstance(names_raw, list):
            names = {i: str(n) for i, n in enumerate(names_raw)}
        else:
            names = {int(k): str(v) for k, v in names_raw.items()}

        def resolve_paths(entry):
            if isinstance(entry, list):
                return [os.path.join(base_dir, p) if not os.path.isabs(p) else p for p in entry]
            return [os.path.join(base_dir, entry) if not os.path.isabs(entry) else entry]

        self.train_dirs = resolve_paths(cfg["train"])
        self.val_dirs = resolve_paths(cfg.get("val", cfg["train"]))

        if len(names) != self.vfgn_model.nc:
            self.vfgn_model.names = names
            self.vfgn_model.nc = len(names)
            self.vfgn_model.net = VFGN26Detector(
                variant=self.vfgn_model.variant,
                nc=self.vfgn_model.nc,
                pretrained_backbone=True,
            ).to(self.device)
        else:
            self.vfgn_model.names = names
        self.vfgn_model.imgsz = self.imgsz

    def fit(self) -> dict:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        train_ds = YOLODetectionDataset(self.train_dirs, imgsz=self.imgsz, augment=True)
        val_ds = YOLODetectionDataset(self.val_dirs, imgsz=self.imgsz, augment=False)

        if len(train_ds) == 0:
            raise RuntimeError(f"No training images found in: {self.train_dirs}")

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=("cuda" in str(self.device).lower()),
            collate_fn=collate_detection_batch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=("cuda" in str(self.device).lower()),
            collate_fn=collate_detection_batch,
        )

        net = self.vfgn_model.net
        net.to(self.device)
        ema = ModelEMA(net, decay=0.9995, tau=50.0)

        criterion = VFGNDetectionLoss(nc=self.vfgn_model.nc, strides=net.strides)
        # 3 parameter groups (just like YOLO):
        #   0: Pretrained backbone (0.15x lr -> protects ConvNeXt features)
        #   1: Neck & Head weights (1.0x lr)
        #   2: Head biases (5.0x lr -> quickly moves prior_prob=0.01 bias in Epoch 1-2!)
        backbone_params = list(net.backbone.parameters())
        bias_params = [p for n, p in net.named_parameters() if ("head" in n and "bias" in n)]
        bias_ids = {id(p) for p in bias_params}
        head_neck_params = [
            p for p in (list(net.neck.parameters()) + list(net.head.parameters()))
            if id(p) not in bias_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr * 0.15},
                {"params": head_neck_params, "lr": self.lr},
                {"params": bias_params, "lr": self.lr * 5.0, "weight_decay": 0.0},
            ],
            lr=self.lr,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.epochs, 1), eta_min=self.lr * 0.01
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        best_fitness = -float("inf")
        best_loss = float("inf")
        best_epoch = 1
        epochs_no_improve = 0
        best_vf_path = os.path.join(self.save_dir, "best.vf")
        last_vf_path = os.path.join(self.save_dir, "last.vf")
        csv_path = os.path.join(self.save_dir, "results.csv")
        history = []

        n_params = sum(p.numel() for p in net.parameters())
        n_grads = sum(p.numel() for p in net.parameters() if p.requires_grad)
        img_w, img_h = parse_imgsz_wh(self.imgsz)
        size_str = f"{img_w}" if img_w == img_h else f"{img_w}x{img_h}"

        print("\n" + "=" * 96, flush=True)
        print(
            f"VerifyGN {self.vfgn_model.variant} summary: {len(list(net.modules()))} layers, "
            f"{n_params:,} parameters, {n_grads:,} gradients",
            flush=True,
        )
        print(
            f"Dataset: train={len(train_ds)} images, val={len(val_ds)} images | "
            f"nc={self.vfgn_model.nc} classes={self.vfgn_model.names} | "
            f"imgsz={size_str} | batch={self.batch_size} | amp={self.use_amp}",
            flush=True,
        )
        print(f"Logging results to {os.path.abspath(self.save_dir)}", flush=True)
        print("=" * 96, flush=True)

        header_train = f"{'Epoch':>10s}{'GPU_mem':>10s}{'box_loss':>11s}{'cls_loss':>11s}{'total_loss':>12s}{'Instances':>11s}{'Size':>10s}"
        header_val = f"{'Class':>18s}{'Images':>9s}{'Instances':>11s}{'Box(P':>11s}{'R':>10s}{'F1':>10s}{'mAP50':>10s}{'mAP50-95)':>11s}"

        t_train_start = time.time()

        for epoch in range(1, self.epochs + 1):
            net.train()
            sum_box_loss = 0.0
            sum_cls_loss = 0.0
            sum_tot_loss = 0.0
            last_instances = 0
            t0 = time.time()
            n_batches = max(len(train_loader), 1)

            for b_i, (imgs, targets) in enumerate(train_loader, start=1):
                imgs = imgs.to(self.device, non_blocking=True)
                last_instances = sum(len(t) for t in targets)
                optimizer.zero_grad(set_to_none=True)

                # Smooth batch-level linear warmup over the first 2 epochs (starts at 5% LR on batch 1)
                if epoch <= 2:
                    global_step = (epoch - 1) * n_batches + b_i
                    total_warmup_steps = max(2 * n_batches, 1)
                    w_fac = 0.05 + 0.95 * (global_step / float(total_warmup_steps))
                    optimizer.param_groups[0]["lr"] = self.lr * 0.15 * w_fac
                    optimizer.param_groups[1]["lr"] = self.lr * w_fac
                    optimizer.param_groups[2]["lr"] = self.lr * 5.0 * w_fac

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    cls_out, reg_out = net.forward_raw(imgs)
                    loss, loss_parts = criterion(net, cls_out, reg_out, targets)

                if not torch.isfinite(loss):
                    logger.warning(f"Skipping non-finite loss batch {b_i} in epoch {epoch}")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(net)

                sum_box_loss += loss_parts["box_loss"]
                sum_cls_loss += loss_parts["cls_loss"]
                sum_tot_loss += float(loss.detach())

            if epoch >= 2:
                scheduler.step()

            avg_box = sum_box_loss / n_batches
            avg_cls = sum_cls_loss / n_batches
            avg_tot = sum_tot_loss / n_batches

            if torch.cuda.is_available():
                vram_used = f"{torch.cuda.max_memory_reserved(0) / (1024 ** 3):.2f}G"
            else:
                vram_used = "0.0G"

            # Evaluate BOTH EMA model and live model in early epochs and pick the stronger one
            val_ema = evaluate_full_metrics(ema.ema, val_loader, self.device, self.vfgn_model.nc, self.vfgn_model.names)
            if epoch <= 15:
                val_live = evaluate_full_metrics(net, val_loader, self.device, self.vfgn_model.nc, self.vfgn_model.names)
                if (val_live["mAP50"] + val_live["mAP50-95"]) > (val_ema["mAP50"] + val_ema["mAP50-95"]):
                    val_metrics = val_live
                    eval_net_to_save = net
                else:
                    val_metrics = val_ema
                    eval_net_to_save = ema.ema
            else:
                val_metrics = val_ema
                eval_net_to_save = ema.ema

            dt = time.time() - t0

            # Print YOLO26-style formatted table
            print(f"\n{header_train}", flush=True)
            print(
                f"{f'{epoch}/{self.epochs}':>10s}{vram_used:>10s}"
                f"{avg_box:11.4f}{avg_cls:11.4f}{avg_tot:12.4f}"
                f"{last_instances:11d}{size_str:>10s}   ({dt:.1f}s)",
                flush=True,
            )
            print(header_val, flush=True)
            print(
                f"{'all':>18s}{val_metrics['total_images']:9d}{val_metrics['total_instances']:11d}"
                f"{val_metrics['precision']:11.4f}{val_metrics['recall']:10.4f}{val_metrics['F1']:10.4f}"
                f"{val_metrics['mAP50']:10.4f}{val_metrics['mAP50-95']:11.4f}",
                flush=True,
            )
            if self.vfgn_model.nc > 1:
                for c_id, c_info in val_metrics["per_class"].items():
                    print(
                        f"{c_info['name']:>18s}{c_info['images']:9d}{c_info['instances']:11d}"
                        f"{c_info['P']:11.4f}{c_info['R']:10.4f}{c_info['F1']:10.4f}"
                        f"{c_info['mAP50']:10.4f}{c_info['mAP50-95']:11.4f}",
                        flush=True,
                    )

            metrics = {
                "epoch": epoch,
                "box_loss": round(avg_box, 4),
                "cls_loss": round(avg_cls, 4),
                "train_loss": round(avg_tot, 4),
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "F1": val_metrics["F1"],
                "mAP50": val_metrics["mAP50"],
                "mAP50-95": val_metrics["mAP50-95"],
                "lr": round(float(optimizer.param_groups[1]["lr"]), 6),
            }
            history.append(metrics)

            # Append to results.csv
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

            # YOLO standard fitness: 0.1 * mAP50 + 0.9 * mAP50-95 (with loss tie-breaker)
            fitness = (0.1 * val_metrics["mAP50"] + 0.9 * val_metrics["mAP50-95"]) * 100.0 - avg_tot * 0.001

            save_vf(
                path=last_vf_path,
                state_dict=eval_net_to_save.state_dict(),
                variant=self.vfgn_model.variant,
                names=self.vfgn_model.names,
                imgsz=self.imgsz,
                metrics=metrics,
                epoch=epoch,
                key=self.key,
            )

            if fitness >= best_fitness:
                best_fitness = fitness
                best_epoch = epoch
                epochs_no_improve = 0
                if avg_tot < best_loss:
                    best_loss = avg_tot
                save_vf(
                    path=best_vf_path,
                    state_dict=eval_net_to_save.state_dict(),
                    variant=self.vfgn_model.variant,
                    names=self.vfgn_model.names,
                    imgsz=self.imgsz,
                    metrics=metrics,
                    epoch=epoch,
                    key=self.key,
                )
            else:
                epochs_no_improve += 1
                if self.patience > 0 and epochs_no_improve >= self.patience:
                    print(f"\nEarlyStopping: Training stopped early as no improvement observed in last {self.patience} epochs. Best results observed at epoch {best_epoch}.", flush=True)
                    break

        total_hours = (time.time() - t_train_start) / 3600.0
        best_pkg = load_vf(best_vf_path, key=self.key, map_location=self.device)
        self.vfgn_model.net.load_state_dict(best_pkg["state_dict"], strict=True)
        self.vfgn_model.metadata = best_pkg["metadata"]

        # Final validation & visual plots on best.vf
        final_val = evaluate_full_metrics(self.vfgn_model.net, val_loader, self.device, self.vfgn_model.nc, self.vfgn_model.names)
        save_training_plots_and_previews(history, self.save_dir, self.vfgn_model.net, val_ds, self.device, self.vfgn_model.names)

        print("\n" + "=" * 96, flush=True)
        print(f"{epoch} epochs completed in {total_hours:.3f} hours.", flush=True)
        print(f"Optimizer stripped from {best_vf_path}, {os.path.getsize(best_vf_path)/(1024*1024):.1f}MB", flush=True)
        print(f"Optimizer stripped from {last_vf_path}, {os.path.getsize(last_vf_path)/(1024*1024):.1f}MB", flush=True)
        print(f"\nValidating {best_vf_path} (Best Epoch {best_epoch})...", flush=True)
        print(header_val, flush=True)
        print(
            f"{'all':>18s}{final_val['total_images']:9d}{final_val['total_instances']:11d}"
            f"{final_val['precision']:11.4f}{final_val['recall']:10.4f}{final_val['F1']:10.4f}"
            f"{final_val['mAP50']:10.4f}{final_val['mAP50-95']:11.4f}",
            flush=True,
        )
        for c_id, c_info in final_val["per_class"].items():
            print(
                f"{c_info['name']:>18s}{c_info['images']:9d}{c_info['instances']:11d}"
                f"{c_info['P']:11.4f}{c_info['R']:10.4f}{c_info['F1']:10.4f}"
                f"{c_info['mAP50']:10.4f}{c_info['mAP50-95']:11.4f}",
                flush=True,
            )
        print(f"Results saved to {os.path.abspath(self.save_dir)} (best.vf, last.vf, results.csv, results.png, val_batch0_pred.jpg)", flush=True)
        print("=" * 96 + "\n", flush=True)

        return {
            "best_vf": best_vf_path,
            "last_vf": last_vf_path,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_metrics": best_pkg["metadata"].get("metrics", {}),
            "variant": self.vfgn_model.variant,
            "names": self.vfgn_model.names,
        }
