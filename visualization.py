# ============================================================
#  visualization.py
#  Bounding boxes, compliance labels, HUD draw karta hai
# ============================================================

import cv2
import numpy as np


# Colors
COLOR_COMPLIANT     = (0,   255,  0)    # Green  ✅
COLOR_VIOLATION     = (0,   0,   255)   # Red    ❌
COLOR_UNKNOWN       = (0,   165, 255)   # Orange ❓
COLOR_ZONE          = (0,   255, 255)   # Yellow


class Visualizer:
    """
    Frame pe safety compliance visualization draw karta hai.
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Heatmap accumulator
        self.heatmap = np.zeros(
            (frame_h, frame_w), dtype=np.float32)

    # ----------------------------------------------------------
    def draw_worker(self, frame: np.ndarray,
                    obj: dict, helmet: dict) -> np.ndarray:
        """
        Worker ka box + compliance status draw karo.

        Label format:
        ✅ Worker #3 | Compliant
        ❌ Worker #5 | NO HELMET
        """
        x1, y1, x2, y2 = map(int, obj["box"])
        tid    = obj["track_id"]
        status = helmet.get("status", "Unknown")

        if status == "Compliant":
            color = COLOR_COMPLIANT
            icon  = "✓"
            label = f"{icon} Worker #{tid} | Compliant"
        else:
            color = COLOR_VIOLATION
            icon  = "✗"
            label = f"{icon} Worker #{tid} | NO HELMET"

        # Main box
        thickness = 3 if status == "Non-Compliant" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Head area highlight
        hx1, hy1, hx2, hy2 = map(int, obj["head_box"])
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2),
                      color, 1)

        # Label background
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        pad = 4
        cv2.rectangle(frame,
                      (x1, y1 - th - 2*pad),
                      (x1 + tw + 2*pad, y1),
                      color, -1)

        cv2.putText(frame, label,
                    (x1 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Violation pe flashing border
        if status == "Non-Compliant":
            cv2.rectangle(frame,
                          (x1-3, y1-3), (x2+3, y2+3),
                          (0, 0, 200), 1)

        # Heatmap update
        self.heatmap[y1:y2, x1:x2] += 1.0

        return frame

    # ----------------------------------------------------------
    def draw_hud(self, frame: np.ndarray,
                 summary: dict, frame_num: int) -> np.ndarray:
        """Top-left mein live safety stats dikhao"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 190),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        total    = summary.get("total_workers", 0)
        comp     = summary.get("compliant", 0)
        viol     = summary.get("violations", 0)
        rate     = summary.get("compliance_rate", 0)
        snaps    = summary.get("snapshots_saved", 0)

        # Rate color
        rate_color = (0, 255, 0) if rate >= 80 else \
                     (0, 165, 255) if rate >= 50 else \
                     (0, 0, 255)

        lines = [
            ("HELMET SAFETY SYSTEM",  (0, 220, 220)),
            (f"Frame: {frame_num}",   (180, 180, 180)),
            ("",                       (255, 255, 255)),
            (f"Workers   : {total}",  (255, 255, 255)),
            (f"Compliant : {comp}",   COLOR_COMPLIANT),
            (f"Violations: {viol}",   COLOR_VIOLATION),
            (f"Rate      : {rate}%",  rate_color),
            (f"Snapshots : {snaps}",  (200, 200, 0)),
        ]

        y = 20
        for text, color in lines:
            cv2.putText(frame, text, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, color, 1, cv2.LINE_AA)
            y += 23

        return frame

    # ----------------------------------------------------------
    def draw_alert(self, frame: np.ndarray,
                   violations: int) -> np.ndarray:
        """Agar violations hain toh alert banner dikhao"""
        if violations == 0:
            return frame

        H, W = frame.shape[:2]
        msg  = f"⚠  SAFETY ALERT: {violations} WORKER(S) WITHOUT HELMET"

        # Bottom pe red banner
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (0, H - 40), (W, H),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, msg,
                    (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    # ----------------------------------------------------------
    def draw_restricted_zone(self, frame: np.ndarray,
                              zone: tuple,
                              violators: set) -> np.ndarray:
        """Restricted zone draw karo"""
        if zone is None:
            return frame
        x1, y1, x2, y2 = zone
        color = (0, 0, 255) if violators else COLOR_ZONE

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "HELMET REQUIRED ZONE",
                    (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
        return frame

    # ----------------------------------------------------------
    def get_heatmap(self) -> np.ndarray:
        norm = cv2.normalize(
            self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(
            norm.astype(np.uint8), cv2.COLORMAP_JET)
