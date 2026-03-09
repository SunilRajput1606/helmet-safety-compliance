# ⛑️ Helmet Safety Compliance Detection System

## Overview
Real-time AI system jo warehouse workers ki helmet compliance detect karta hai.
Har worker ko track karta hai aur violations ka snapshot save karta hai.

---

## System Architecture

```
Video Input
    ↓
[tracker.py]         YOLOv8 + ByteTrack — workers track karo
    ↓
[helmet_checker.py]  Head area mein helmet detect karo
    ↓
[analytics.py]       Compliance count + CSV + Snapshots
    ↓
[visualization.py]   Boxes + Labels + Alert banner
    ↓
Output Video + CSV + Violation Images + Heatmap
```

---

## Project Structure

```
helmet-safety-system/
├── data/
│   └── sample_video.mp4      ← Apna video yahan rakho
├── src/
│   ├── tracker.py            ← ByteTrack tracking
│   ├── helmet_checker.py     ← Helmet detection logic
│   ├── analytics.py          ← Compliance analytics
│   ├── visualization.py      ← Drawing functions
│   └── utils.py              ← Helper functions
├── output/
│   ├── output_video.mp4      ← Annotated video
│   ├── compliance_logs.csv   ← Compliance data
│   ├── heatmap.jpg           ← Movement heatmap
│   └── violations/           ← Violation snapshots
├── main.py
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard.py
```

---

## CSV Format

```
frame, timestamp, track_id, status, helmet_color, confidence, x, y, width, height
12,    0.400,     4,        Compliant, yellow,    0.85, 120, 230, 50, 60
13,    0.433,     5,        Non-Compliant, none,  0.0,  200, 150, 45, 55
```

---

## Compliance Logic

```
✅ Compliant     = Head area mein helmet color detect hua
❌ Non-Compliant = Head area mein koi helmet color nahi mila
📸 Snapshot      = Har violator ka har 90 frames mein save hota hai
```

---

## Limitations
- Color-based detection — complex lighting mein accuracy kam hoti hai
- Custom helmet YOLO model se accuracy bohot badhegi
- Roboflow pe "helmet detection" search karo free model ke liye
- Occlusion mein tracking ID change ho sakti hai

## Future Improvements
- Custom YOLOv8 helmet model train karna
- REST API endpoint banana
- Real-time camera (RTSP) support
- SMS/Email alerts for violations
