# ============================================================
#  helmet_checker.py
#  Har worker ke liye helmet compliance check karta hai
# ============================================================

import cv2
import numpy as np


class HelmetChecker:
    """
    Person ke sir ke area mein helmet detect karta hai.

    2 methods use karta hai:
    1. Color-based detection (hardhat colors: yellow, orange, white, blue)
    2. Brightness/texture analysis

    Note: Best results ke liye custom YOLO helmet model use karo.
    Roboflow pe "helmet detection" search karo — free model milega.
    """

    # Helmet ke common colors (HSV format)
    HELMET_COLORS = {
        "yellow": ([20,  100, 100], [30,  255, 255]),
        "orange": ([5,   100, 100], [20,  255, 255]),
        "white" : ([0,   0,   180], [180, 30,  255]),
        "blue"  : ([100, 100, 100], [130, 255, 255]),
        "red1"  : ([0,   100, 100], [10,  255, 255]),
        "red2"  : ([170, 100, 100], [180, 255, 255]),
    }

    def __init__(self, min_helmet_ratio: float = 0.08):
        """
        min_helmet_ratio : head area ka kitna % helmet color hona chahiye
        """
        self.min_ratio = min_helmet_ratio
        print("[HelmetChecker] ✅ Ready!")

    # ----------------------------------------------------------
    def check(self, frame: np.ndarray,
              head_box: list) -> dict:
        """
        Head area mein helmet hai ya nahi check karo.

        Returns:
        {
          "has_helmet"  : True/False,
          "confidence"  : 0.0 - 1.0,
          "color"       : "yellow" / "none",
          "status"      : "Compliant" / "Non-Compliant"
        }
        """
        x1, y1, x2, y2 = map(int, head_box)
        H, W = frame.shape[:2]

        # Bounds check
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return self._result(False, 0.0, "none")

        # Head region crop karo
        head_region = frame[y1:y2, x1:x2]
        if head_region.size == 0:
            return self._result(False, 0.0, "none")

        # HSV convert karo
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        total_pixels = head_region.shape[0] * head_region.shape[1]

        if total_pixels == 0:
            return self._result(False, 0.0, "none")

        # Har helmet color check karo
        best_ratio = 0.0
        best_color = "none"

        for color_name, (lower, upper) in self.HELMET_COLORS.items():
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask     = cv2.inRange(hsv, lower_np, upper_np)
            ratio    = cv2.countNonZero(mask) / total_pixels

            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color_name

        # Red 2 ranges combine karo
        if best_color in ("red1", "red2"):
            best_color = "red"

        has_helmet = best_ratio >= self.min_ratio
        confidence = min(best_ratio / self.min_ratio, 1.0)

        return self._result(has_helmet, confidence, best_color)

    # ----------------------------------------------------------
    def _result(self, has_helmet: bool,
                confidence: float, color: str) -> dict:
        return {
            "has_helmet" : has_helmet,
            "confidence" : round(confidence, 2),
            "color"      : color,
            "status"     : "Compliant" if has_helmet else "Non-Compliant",
        }
