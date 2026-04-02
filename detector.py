"""
detector.py  —  YOLO wrapper + zone-intrusion logic
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import config


def _poly_from_fractions(zone_fracs, frame_h, frame_w):
    """Convert fractional (0-1) zone coordinates to pixel coordinates."""
    pts = [(int(x * frame_w), int(y * frame_h)) for x, y in zone_fracs]
    return np.array(pts, dtype=np.int32)


def _point_in_poly(cx, cy, poly):
    """Return True if (cx, cy) is inside the polygon."""
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


class Detector:
    def __init__(self):
        self.model = YOLO(config.MODEL)
        self.names = self.model.names          # {id: class_name}
        self.alert_classes = set(config.ALERT_CLASSES)

    # ------------------------------------------------------------------ #
    def run(self, frame):
        """
        Run inference on one frame.
        Returns:
            annotated  — frame with drawings
            counts     — dict {class_name: count}
            alert      — True if any object entered the alert zone
        """
        h, w = frame.shape[:2]
        zone_poly = (
            _poly_from_fractions(config.ALERT_ZONE, h, w)
            if config.ALERT_ZONE else None
        )

        results = self.model(
            frame,
            conf=config.CONFIDENCE,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMGSZ,
            verbose=False,
        )[0]

        counts = defaultdict(int)
        alert = False
        annotated = frame.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.names[cls_id]
            counts[label] += 1

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Zone intrusion check
            in_zone = (
                zone_poly is not None
                and _point_in_poly(cx, cy, zone_poly)
                and (not self.alert_classes or label in self.alert_classes)
            )
            if in_zone:
                alert = True

            # ── Draw bounding box ──
            color = (0, 60, 220) if not in_zone else (0, 30, 200)
            if in_zone:
                # thicker red box for intrusions
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 230), 3)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 200, 50), 2)

            # ── Label chip ──
            tag = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            chip_color = (0, 0, 200) if in_zone else (30, 160, 30)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), chip_color, -1)
            cv2.putText(annotated, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Centre dot
            cv2.circle(annotated, (cx, cy), 3, (255, 255, 0), -1)

        # ── Draw alert zone ──
        if zone_poly is not None and config.SHOW_ZONE:
            zone_color = (0, 0, 255) if alert else (0, 255, 255)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [zone_poly], (*zone_color[::-1], 40))
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            cv2.polylines(annotated, [zone_poly], True, zone_color, 2, cv2.LINE_AA)
            label_txt = "ZONE: ALERT" if alert else "ZONE: CLEAR"
            cv2.putText(annotated, label_txt,
                        (zone_poly[:, 0].min(), zone_poly[:, 1].min() - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone_color, 2, cv2.LINE_AA)

        return annotated, dict(counts), alert
