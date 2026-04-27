
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    conf: float
    xyxy: Tuple[int, int, int, int]


def load_image_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image. Please upload a valid PNG/JPG.')
    return img


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1c = max(0, min(x1, w - 1))
    y1c = max(0, min(y1, h - 1))
    x2c = max(0, min(x2, w - 1))
    y2c = max(0, min(y2, h - 1))
    if x2c <= x1c:
        x2c = min(w - 1, x1c + 1)
    if y2c <= y1c:
        y2c = min(h - 1, y1c + 1)
    return x1c, y1c, x2c, y2c


def draw_detections(img_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 2)
        txt = d.label + ' ' + str(round(d.conf, 2))
        cv2.putText(out, txt, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    return out


def compute_defect_score(roi_bgr: np.ndarray) -> Dict[str, float]:
    """Explainable heuristic defect scoring.

    Returns
    spot_area_ratio: fraction of ROI flagged as dark spots
    edge_density: proxy for surface irregularities
    defect_score_0_100: combined score

    This is a course-project-friendly approximation.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return {
            'spot_area_ratio': 0.0,
            'edge_density': 0.0,
            'defect_score_0_100': 0.0,
        }

    roi = roi_bgr.copy()
    h, w = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    thr = int(max(30, np.percentile(v, 20)))
    spot_mask = (v < thr).astype(np.uint8) * 255

    k = max(3, (min(h, w) // 50) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)

    spot_area_ratio = float(np.mean(spot_mask > 0))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)
    edge_density = float(np.mean(edges > 0))

    defect_score = 100.0 * (0.75 * min(1.0, spot_area_ratio * 4.0) + 0.1 * min(1.0, edge_density * 3.0))

    return {
        'spot_area_ratio': spot_area_ratio,
        'edge_density': edge_density,
        'defect_score_0_100': float(defect_score),
    }


def quality_score(defect_score_0_100: float, cls_probs: Dict[str, float]) -> Dict[str, object]:
    p_fresh = float(cls_probs.get('fresh', 0.0))
    p_rotten = float(cls_probs.get('rotten', 0.0))

    # 🚨 HARD RULE (very important)
    if defect_score_0_100 > 40:
        return {
            'quality_score_0_100': float(30.0),
            'grade': 'C',
            'decision': 'REJECT',
        }

    # Normal scoring
    base = 100.0 * (0.7 * p_fresh + 0.3 * (1.0 - p_rotten))
    score = base - 0.6 * defect_score_0_100
    score = float(max(0.0, min(100.0, score)))

    if score >= 75:
        grade = 'A'
        decision = 'ACCEPT'
    elif score >= 50:
        grade = 'B'
        decision = 'CHECK'
    else:
        grade = 'C'
        decision = 'REJECT'

    return {
        'quality_score_0_100': score,
        'grade': grade,
        'decision': decision,
    }