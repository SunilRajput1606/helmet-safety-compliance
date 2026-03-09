# ============================================================
#  analytics.py
#  Compliance analytics + CSV logging + Violation snapshots
# ============================================================

import csv
import os
import cv2
import numpy as np
from collections import defaultdict


class SafetyAnalytics:
    """
    Compliance tracking:
    - Total workers count
    - Compliant count
    - Non-Compliant (violation) count
    - Violation snapshots save karta hai
    - CSV log generate karta hai
    """

    def __init__(self, csv_path: str, snapshots_dir: str, fps: float):
        self.csv_path      = csv_path
        self.snapshots_dir = snapshots_dir
        self.fps           = fps

        os.makedirs(snapshots_dir, exist_ok=True)

        # Unique worker IDs
        self.all_workers     = set()
        self.violators       = set()   # No helmet IDs
        self.compliant       = set()   # Helmet IDs

        # Frame-wise stats
        self.frame_stats = []

        # Violation snapshot counter
        self.snapshot_count  = 0
        self.last_snapshot   = {}   # {track_id: last_snapshot_frame}

        # CSV banao
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "timestamp", "track_id",
                "status", "helmet_color", "confidence",
                "x", "y", "width", "height"
            ])

        print(f"[Analytics] CSV: {csv_path}")
        print(f"[Analytics] Snapshots: {snapshots_dir}")

    # ----------------------------------------------------------
    def update(self, frame: np.ndarray, tracked: list[dict],
               helmet_results: dict, frame_num: int):
        """
        Har frame mein call karo.

        tracked        = tracker ka output
        helmet_results = { track_id: helmet_checker ka result }
        """
        timestamp = round(frame_num / self.fps, 3)

        for obj in tracked:
            tid    = obj["track_id"]
            box    = obj["box"]
            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1

            helmet = helmet_results.get(tid, {
                "has_helmet": False,
                "confidence": 0.0,
                "color"     : "none",
                "status"    : "Non-Compliant",
            })

            # Unique workers track karo
            self.all_workers.add(tid)

            if helmet["has_helmet"]:
                self.compliant.add(tid)
                # Agar pehle violator tha toh hata do
                self.violators.discard(tid)
            else:
                # Sirf tab violator mark karo jab
                # kabhi compliant nahi tha
                if tid not in self.compliant:
                    self.violators.add(tid)

            # CSV row
            self.frame_stats.append([
                frame_num,
                timestamp,
                tid,
                helmet["status"],
                helmet["color"],
                helmet["confidence"],
                x1, y1, w, h,
            ])

            # Violation snapshot save karo
            # Har violator ka har 90 frames mein ek snapshot
            if not helmet["has_helmet"]:
                last = self.last_snapshot.get(tid, -999)
                if frame_num - last >= 90:
                    self._save_snapshot(
                        frame, box, tid, frame_num)
                    self.last_snapshot[tid] = frame_num

    # ----------------------------------------------------------
    def _save_snapshot(self, frame: np.ndarray,
                       box: list, tid: int, frame_num: int):
        """Violation ka snapshot save karo"""
        x1, y1, x2, y2 = map(int, box)
        H, W = frame.shape[:2]
        pad  = 20

        # Thoda padding ke saath crop karo
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(W, x2 + pad)
        cy2 = min(H, y2 + pad)

        crop = frame[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            return

        # Red border lagao
        cv2.rectangle(crop, (0, 0),
                      (crop.shape[1]-1, crop.shape[0]-1),
                      (0, 0, 255), 3)

        # Label
        cv2.putText(crop, f"NO HELMET ID:{tid}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        path = os.path.join(
            self.snapshots_dir,
            f"violation_id{tid}_f{frame_num}.jpg"
        )
        cv2.imwrite(path, crop)
        self.snapshot_count += 1
        print(f"  📸 Snapshot: {path}")

    # ----------------------------------------------------------
    def flush_csv(self):
        """Buffered rows CSV mein likho"""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.frame_stats)
        self.frame_stats = []

    # ----------------------------------------------------------
    def get_summary(self) -> dict:
        total      = len(self.all_workers)
        violations = len(self.violators)
        compliant  = len(self.compliant)
        rate       = round(
            compliant / total * 100 if total else 0, 1)

        return {
            "total_workers"    : total,
            "compliant"        : compliant,
            "violations"       : violations,
            "compliance_rate"  : rate,
            "snapshots_saved"  : self.snapshot_count,
        }
