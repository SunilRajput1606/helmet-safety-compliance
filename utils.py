# ============================================================
#  utils.py — Helper functions
# ============================================================

import cv2
import numpy as np


def resize_frame(frame: np.ndarray,
                 width: int = 640, height: int = 480) -> np.ndarray:
    """Frame resize karo performance ke liye"""
    return cv2.resize(frame, (width, height))


def get_centroid(box: list) -> tuple:
    """Bounding box ka center point"""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def is_in_zone(box: list, zone: tuple) -> bool:
    """Kya box ka center zone ke andar hai?"""
    if zone is None:
        return False
    cx, cy = get_centroid(box)
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def iou(box1: list, box2: list) -> float:
    """
    Two boxes ka Intersection over Union calculate karo.
    Helmet association ke liye use hota hai.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def draw_compliance_bar(frame: np.ndarray,
                        rate: float) -> np.ndarray:
    """
    Bottom-right mein compliance rate bar draw karo.
    rate = 0-100
    """
    H, W = frame.shape[:2]
    bar_w = 150
    bar_h = 18
    x     = W - bar_w - 10
    y     = H - 30

    # Background
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                  (50, 50, 50), -1)

    # Fill
    fill_w = int(bar_w * rate / 100)
    color  = (0, 255, 0) if rate >= 80 else \
             (0, 165, 255) if rate >= 50 else \
             (0, 0, 255)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y),
                      (x + fill_w, y + bar_h),
                      color, -1)

    # Text
    cv2.putText(frame, f"Compliance: {rate:.0f}%",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    return frame
