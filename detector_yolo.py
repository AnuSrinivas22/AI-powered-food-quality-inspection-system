
from __future__ import annotations

from typing import List

import numpy as np
from ultralytics import YOLO

from .utils import Detection, clip_box


class YoloDetector:
    def __init__(self, model_name: str = 'yolov8n.pt', conf: float = 0.25):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        h, w = img_bgr.shape[:2]
        results = self.model.predict(img_bgr, conf=self.conf, verbose=False)
        dets: List[Detection] = []

        if len(results) == 0:
            return dets

        r0 = results[0]
        if r0.boxes is None:
            return dets

        names = r0.names
        boxes = r0.boxes

        for i in range(len(boxes)):
            b = boxes[i]
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            x1i, y1i, x2i, y2i = clip_box(x1i, y1i, x2i, y2i, w, h)
            label = str(names.get(cls_id, cls_id))
            dets.append(Detection(label=label, conf=conf, xyxy=(x1i, y1i, x2i, y2i)))

        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets
