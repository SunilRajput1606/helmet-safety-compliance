# ⛑️ Helmet Safety System — System Design

---

## 1. Detection Pipeline

```
Frame (640x480)
      ↓
  YOLOv8n — Person detect karo
      ↓
  Head Box = Person box ka upar 35%
      ↓
  HelmetChecker — Head area mein
  helmet color check karo (HSV)
      ↓
  Output:
  - Compliant / Non-Compliant
  - Helmet color
  - Confidence score
```

---

## 2. Helmet Association Logic

```
Person detect hua → [x1, y1, x2, y2]
      ↓
Head Box = [x1, y1, x2, y1 + height×0.35]
      ↓
HSV color analysis:
  Yellow, Orange, White, Blue, Red → Helmet ✅
  No matching color              → No Helmet ❌
      ↓
min_ratio = 8% pixels match → Compliant
```

---

## 3. Tracking Pipeline

```
ByteTrack (persist=True)
      ↓
Har worker ko unique ID
      ↓
ID frames ke across consistent rehta hai
      ↓
Compliance history per ID maintain hoti hai
```

---

## 4. Analytics Pipeline

```
Tracked Workers + Helmet Results
      ↓
Unique IDs set maintain karo
      ↓
Compliant set / Violator set update karo
      ↓
Violation snapshot save karo (90 frame interval)
      ↓
CSV row append karo
      ↓
Compliance rate calculate karo:
  rate = compliant / total × 100
```

---

## 5. Data Flow

```
┌──────────┐   frames   ┌──────────────┐
│  OpenCV  │──────────→ │   Tracker    │
└──────────┘            └──────┬───────┘
                               │ tracked workers
              ┌────────────────┼────────────────┐
              ↓                ↓                ↓
     ┌──────────────┐ ┌──────────────┐ ┌───────────┐
     │HelmetChecker │ │  Analytics   │ │Visualizer │
     │Color detect  │ │CSV+Snapshots │ │Boxes+HUD  │
     └──────────────┘ └──────────────┘ └─────┬─────┘
                                             ↓
                                      Output Video
```

---

## 6. Bonus Features

| Feature | Implementation |
|---------|---------------|
| Restricted Zone | Centroid zone check |
| Alert Banner | Red banner jab violations > 0 |
| Violation Snapshots | Auto-save every 90 frames |
| Compliance Bar | Bottom-right progress bar |
| Heatmap | NumPy + OpenCV COLORMAP_JET |
| Dashboard | Streamlit + Plotly |
