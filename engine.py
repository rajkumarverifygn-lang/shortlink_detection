# Copyright (c) VerifyGN. All Rights Reserved.
"""V-Count inference engine for the VFGN26 shortlink detector (.vf model)."""

import os
import sys
import time
import threading

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from vfgn import VFGNModel, inspect_vf  # noqa: E402

# Colours (BGR)
GREEN = (60, 200, 40)
RED = (40, 40, 230)
AMBER = (0, 190, 255)
DARK = (30, 23, 15)


def verdict(count: int, set_count: int, enabled: bool = True) -> str:
    """Return 'OK', 'NOK' or '-' (check disabled)."""
    if not enabled:
        return "-"
    return "OK" if int(count) == int(set_count) else "NOK"


class VCountEngine:
    def __init__(self, model_path: str, key: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.meta = inspect_vf(model_path)
        self.model = VFGNModel(model_path, key=key, device=self.device)
        self.lock = threading.Lock()
        # warm-up
        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    # ------------------------------------------------------------------ #
    def detect(self, img_bgr: np.ndarray, conf: float = 0.50, iou: float = 0.60, max_det: int = 300):
        """Returns (boxes[N,4] xyxy, scores[N], infer_ms). Boxes sorted row-wise (top->bottom, left->right)."""
        t0 = time.perf_counter()
        m = self.model
        oh, ow = img_bgr.shape[:2]
        iw, ih = (m.imgsz if isinstance(m.imgsz, (list, tuple)) else (m.imgsz, m.imgsz))
        iw, ih = int(iw), int(ih)

        img = cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_LINEAR) if (iw, ih) != (ow, oh) else img_bgr
        sx, sy = ow / float(iw), oh / float(ih)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))

        with self.lock:
            if m.ort_session is not None:
                blob = rgb[None].astype(np.float32) / 255.0
                preds = torch.from_numpy(m.ort_session.run(None, {m.ort_input_name: blob})[0][0])
            else:
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=(str(m.device) == "cuda")):
                    inp = torch.from_numpy(rgb).unsqueeze(0).to(m.device).float().div_(255.0)
                    preds = m.net(inp)[0]

        boxes, scores_mat = preds[:, :4].float(), preds[:, 4:].float()
        scores, cls_ids = scores_mat.max(dim=1)
        keep = scores >= conf
        empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
        if not keep.any():
            return (*empty, (time.perf_counter() - t0) * 1000)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * sx).clamp(0, ow)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * sy).clamp(0, oh)
        valid = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
        boxes, scores = boxes[valid], scores[valid]
        if len(boxes) == 0:
            return (*empty, (time.perf_counter() - t0) * 1000)

        b_np, s_np = boxes.detach().cpu().numpy(), scores.detach().cpu().numpy()
        order_s = np.argsort(-s_np)
        keep_idx = []
        while len(order_s) > 0:
            i = order_s[0]
            keep_idx.append(i)
            if len(order_s) == 1 or len(keep_idx) >= max_det:
                break
            rest = order_s[1:]
            bi, br = b_np[i], b_np[rest]
            ix1, iy1 = np.maximum(bi[0], br[:, 0]), np.maximum(bi[1], br[:, 1])
            ix2, iy2 = np.minimum(bi[2], br[:, 2]), np.minimum(bi[3], br[:, 3])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            box_iou = inter / ((bi[2] - bi[0]) * (bi[3] - bi[1]) + (br[:, 2] - br[:, 0]) * (br[:, 3] - br[:, 1]) - inter + 1e-6)
            cdist = np.hypot((br[:, 0] + br[:, 2]) * 0.5 - (bi[0] + bi[2]) * 0.5, (br[:, 1] + br[:, 3]) * 0.5 - (bi[1] + bi[3]) * 0.5)
            min_dim = np.minimum(np.minimum(bi[2] - bi[0], bi[3] - bi[1]), np.minimum(br[:, 2] - br[:, 0], br[:, 3] - br[:, 1]))
            order_s = rest[~((box_iou > 0.72) | ((box_iou > iou) & (cdist < min_dim * 0.26)))]
        b, s = b_np[keep_idx], s_np[keep_idx]

        cy, cx = (b[:, 1] + b[:, 3]) / 2, (b[:, 0] + b[:, 2]) / 2
        order = np.lexsort((cx, np.round(cy / max(16.0, oh * 0.05))))
        return b[order], s[order], (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ #
    @staticmethod
    def annotate(img_bgr, boxes, scores, set_count=None, check_enabled=True,
                 show_numbers=True, show_conf=False, display_count=None, fps=None):
        vis = img_bgr.copy()
        h, w = vis.shape[:2]
        th = max(2, round(min(w, h) / 360))
        fs = max(0.45, min(w, h) / 1200)

        for i, (bx, sc) in enumerate(zip(boxes, scores), start=1):
            x1, y1, x2, y2 = [int(round(v)) for v in bx]
            cv2.rectangle(vis, (x1, y1), (x2, y2), GREEN, th, cv2.LINE_AA)
            cv2.circle(vis, ((x1 + x2) // 2, (y1 + y2) // 2), th + 1, AMBER, -1, cv2.LINE_AA)
            parts = ([f"{i}"] if show_numbers else []) + ([f"{int(sc * 100)}%"] if show_conf else [])
            if parts:
                txt = " ".join(parts)
                (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th - 1))
                ty = max(0, y1 - tht - 6)
                cv2.rectangle(vis, (x1, ty), (x1 + tw + 8, ty + tht + 6), (35, 140, 20), -1)
                cv2.putText(vis, txt, (x1 + 4, ty + tht + 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (255, 255, 255), max(1, th - 1), cv2.LINE_AA)

        count = len(boxes) if display_count is None else display_count
        v = verdict(count, set_count, check_enabled) if set_count is not None else "-"

        # HUD bar
        bh = max(46, int(h * 0.08))
        ov = vis.copy()
        cv2.rectangle(ov, (0, 0), (w, bh), DARK, -1)
        cv2.addWeighted(ov, 0.85, vis, 0.15, 0, vis)
        hfs = max(0.6, bh / 60)
        txt = f"COUNT: {count}"
        if set_count is not None and check_enabled:
            txt += f"  /  SET: {set_count}"
        if fps:
            txt += f"   {fps:.0f} FPS"
        cv2.putText(vis, txt, (14, int(bh * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, hfs, (255, 255, 255), 2, cv2.LINE_AA)
        if v in ("OK", "NOK"):
            col = GREEN if v == "OK" else RED
            (tw, _), _ = cv2.getTextSize(v, cv2.FONT_HERSHEY_DUPLEX, hfs * 1.2, 3)
            cv2.rectangle(vis, (w - tw - 34, 6), (w - 8, bh - 6), col, -1)
            cv2.putText(vis, v, (w - tw - 21, int(bh * 0.72)), cv2.FONT_HERSHEY_DUPLEX, hfs * 1.2,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), col, max(4, th * 2))
        return vis
