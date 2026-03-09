# ============================================================
#  main.py — Main Pipeline
#  Run: python main.py
# ============================================================

import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from  tracker        import Tracker
from  helmet_checker import HelmetChecker
from  analytics      import SafetyAnalytics
from  visualization  import Visualizer
from  utils          import resize_frame, is_in_zone, \
                           draw_compliance_bar

# ---- Config ------
CONFIG = {
    # Paths
    "model_path"    : "yolov8n.pt",
    "input_video"   : "data/sample_video.mp4",
    "output_video"  : "output/output_video.mp4",
    "csv_path"      : "output/compliance_logs.csv",
    "snapshots_dir" : "output/violations",
    "heatmap_path"  : "output/heatmap.jpg",

    # Detection
    "confidence"    : 0.30,
    "iou"           : 0.30,

    # Helmet checker
    "helmet_ratio"  : 0.08,   # head area ka min % helmet color

    # Display
    "show_preview"  : True,
    "resize_w"      : 640,
    "resize_h"      : 480,

    # Restricted zone: (x1,y1,x2,y2) ya None
    "restricted_zone": None,
}


# ---- Main ----
def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/violations", exist_ok=True)

    # ---- Video open ----
    video_path = CONFIG["input_video"]
    if not os.path.exists(video_path):
        print(f"❌ Video is not found: {video_path}")
        print(f"   '{video_path}' video path   !")
        sys.exit(1)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = CONFIG["resize_w"]
    H     = CONFIG["resize_h"]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n🎬 Video: {W}x{H} | {fps:.1f} FPS | {total} frames\n")

    # ----- Modules -----
    tracker = Tracker(
        model_path = CONFIG["model_path"],
        conf       = CONFIG["confidence"],
        iou        = CONFIG["iou"],
    )

    helmet_checker = HelmetChecker(
        min_helmet_ratio = CONFIG["helmet_ratio"]
    )

    analytics = SafetyAnalytics(
        csv_path      = CONFIG["csv_path"],
        snapshots_dir = CONFIG["snapshots_dir"],
        fps           = fps,
    )

    visualizer = Visualizer(frame_w=W, frame_h=H)

    # ---- Output video ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(
        CONFIG["output_video"], fourcc, fps, (W, H))

    # ---- Main Loop ----
    frame_num = 0
    print("▶️  Processing on going...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        frame = resize_frame(frame, W, H)

        # ---- Track workers ----
        tracked = tracker.track(frame)

        # ---- Helmet check for  workers ----
        helmet_results = {}
        for obj in tracked:
            tid    = obj["track_id"]
            result = helmet_checker.check(
                frame, obj["head_box"])
            helmet_results[tid] = result

        # ---- Analytics update ---
        analytics.update(
            frame, tracked, helmet_results, frame_num)

        # ---- Restricted zone ----
        zone      = CONFIG.get("restricted_zone")
        violators = set()
        for obj in tracked:
            if is_in_zone(obj["box"], zone):
                violators.add(obj["track_id"])

        frame = visualizer.draw_restricted_zone(
            frame, zone, violators)

        # ---- Draw workers -----
        for obj in tracked:
            tid    = obj["track_id"]
            helmet = helmet_results.get(tid, {})
            frame  = visualizer.draw_worker(
                frame, obj, helmet)

        # ---- Summary + HUD ----
        summary = analytics.get_summary()
        frame   = visualizer.draw_hud(
            frame, summary, frame_num)

        # Alert banner
        frame = visualizer.draw_alert(
            frame, summary["violations"])

        # Compliance bar
        frame = draw_compliance_bar(
            frame, summary["compliance_rate"])

        # ---- CSV flush ----
        if frame_num % 300 == 0:
            analytics.flush_csv()

        # ---- Write + Preview ----
        out.write(frame)

        if CONFIG["show_preview"]:
            cv2.imshow("Helmet Safety System  [Q=quit]", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⏹️  User ne quit kiya.")
                break

        # Progress
        if frame_num % 100 == 0:
            pct = frame_num / total * 100 if total else 0
            s   = summary
            print(f"   {frame_num}/{total} ({pct:.0f}%)  "
                  f"Workers:{s['total_workers']}  "
                  f"✅{s['compliant']}  ❌{s['violations']}")

    # ---- Cleanup ----
    analytics.flush_csv()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ---- Heatmap ----
    heatmap_img = visualizer.get_heatmap()
    cv2.imwrite(CONFIG["heatmap_path"], heatmap_img)
    print(f"\n🗺️  Heatmap: {CONFIG['heatmap_path']}")

    # ---- Final Summary ----
    summary = analytics.get_summary()
    print("\n" + "="*52)
    print("📊  FINAL SAFETY REPORT")
    print("="*52)
    print(f"  Total Workers    : {summary['total_workers']}")
    print(f"  ✅ Compliant     : {summary['compliant']}")
    print(f"  ❌ Violations    : {summary['violations']}")
    print(f"  📈 Compliance    : {summary['compliance_rate']}%")
    print(f"  📸 Snapshots     : {summary['snapshots_saved']}")
    print(f"\n✅  Output video  : {CONFIG['output_video']}")
    print(f"✅  CSV log        : {CONFIG['csv_path']}")
    print(f"✅  Violations     : {CONFIG['snapshots_dir']}")
    print(f"✅  Heatmap        : {CONFIG['heatmap_path']}")
    print("="*52 + "\n")


if __name__ == "__main__":

    main()
