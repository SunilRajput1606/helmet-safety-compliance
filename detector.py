# ============================================================
#  detector.py
#  YOLOv8 se Person + Helmet detect karta hai
# ============================================================

from ultralytics import YOLO
import numpy as np


# COCO class IDs
# 0 = person
# Helmet ke liye custom model chahiye
# Abhi ke liye hardhat proxy classes use karenge

PERSON_CLASS_ID = 0

# Agar custom helmet model use kar rahe ho toh:
# HELMET_CLASS_ID = 1  (apne model ke hisaab se change karo)


class Detector:
    """
    YOLOv8 wrapper.
    Person detect karta hai aur helmet association karta hai.
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 conf: float = 0.30):
        print(f"[Detector] Model load ho raha hai: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("[Detector] ✅ Model ready!")
        except Exception as e:
            raise RuntimeError(f"[Detector] Model load fail: {e}")

        self.conf = conf

    # ----------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Frame mein persons detect karo.

        Returns:
        [
          {
            "box"       : [x1, y1, x2, y2],
            "class_id"  : 0,
            "class_name": "Person",
            "conf"      : 0.87
          }
        ]
        """
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []
        r = results[0]

        for box, cls_id, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy(),
        ):
            if cls_id == PERSON_CLASS_ID:
                detections.append({
                    "box"       : box.tolist(),
                    "class_id"  : int(cls_id),
                    "class_name": "Person",
                    "conf"      : float(conf),
                })

        return detections
