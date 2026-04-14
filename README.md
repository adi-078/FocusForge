# ⚡ FocusForge AI v2
> **Real-time Cognitive Intelligence System** — context-switch detection, procrastination prevention, and LSTM-based behaviour prediction using computer vision.

---

## 🧩 Problem Statement

Knowledge workers lose an average of **23 minutes of focus after every context switch** (Gloria Mark, UC Irvine). Procrastination and distraction are not character flaws — they are measurable, predictable physiological and behavioural signals.

FocusForge captures these signals in real time from a standard webcam, models them with both rule-based heuristics and a machine-learning layer, and gives the user live feedback — without any cloud API, wearable, or special hardware.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Camera Feed                             │
│                    (standard webcam, 30fps)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    MediaPipe + OpenCV
               (FaceMesh · Pose · Hands)
                           │
                    Kalman Filter +
                  Baseline Calibration
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │  OpenCV HUD │  │  Backend    │  │   SQLite DB  │
   │  (Live)     │  │  Modules    │  │  (Feature    │
   │  real-time  │  │  (7 orig +  │  │   Log + Main)│
   │  overlay    │  │   4 new)    │  └─────────────┘
   └─────────────┘  └──────┬──────┘
                           │
                    FastAPI + WebSocket
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐           ┌──────▼──────┐
       │  OpenCV HUD │           │    React     │
       │  Live       │           │  Dashboard   │
       │  Overlay    │           │  Analytics   │
       └─────────────┘           └─────────────┘
```

**Dual-interface architecture** — OpenCV HUD for real-time frame-level feedback, React Dashboard for analytics, ML outputs, and historical trends.

---

## 📦 Project Structure

```
FocusForge/
├── run.py                              ← Entry point (python run.py)
├── main.py                             ← FastAPI server + 8-stage pipeline
├── requirements.txt
├── config/
│   └── config.yaml                     ← All thresholds + model config
│
├── backend/
│   ├── camera/
│   │   ├── mediapipe_analyzer.py       ← FaceMesh + Pose + Hands CV engine
│   │   └── webcam_capture.py           ← Threaded capture loop
│   │
│   ├── modules/                        ← Original 7 cognitive modules
│   │   ├── signal_processor.py         ← Kalman filter + calibration
│   │   ├── cognitive_state.py          ← Module 2: Focus/Load/Fatigue/Confusion
│   │   ├── context_switch.py           ← Module 1: FSM switch detector
│   │   ├── procrastination.py          ← Module 3: 5-layer procrastination engine
│   │   ├── cognitive_signature.py      ← Module 4: Personal behavioural profile
│   │   ├── temporal_impact.py          ← Module 5+6: Ripple + Meta-cognition
│   │   └── recovery_optimizer.py       ← Module 7: Recovery action planner
│   │
│   ├── models/                         ← NEW v2: ML Model Layer
│   │   ├── temporal_model.py           ← LSTM (pure NumPy, no PyTorch needed)
│   │   └── feature_logger.py           ← Feature log + pseudo-label pipeline
│   │
│   ├── events/                         ← NEW v2: Event Bus
│   │   └── event_bus.py                ← Async pub/sub (decoupled architecture)
│   │
│   ├── api/                            ← NEW v2: Analytics API
│   │   └── metrics.py                  ← Evaluation metrics dashboard
│   │
│   └── database/
│       └── db.py                       ← Async SQLite (extended for v2)
│
├── frontend/
│   └── index.html                      ← OpenCV-style HTML dashboard (v1)
│
├── dashboard/                          ← NEW v2: React Dashboard
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                     ← 5-tab layout
│       ├── index.css
│       ├── hooks/
│       │   └── useWebSocket.js         ← WS connection + history buffer
│       └── components/
│           ├── LivePanel.jsx           ← Real-time gauges + live chart
│           ├── TimelinePanel.jsx       ← Prediction timeline strip
│           ├── MLPanel.jsx             ← LSTM output + radar chart
│           ├── MetricsPanel.jsx        ← Analytics cards + DB charts
│           └── EventPanel.jsx          ← Event bus live feed
│
└── data/                               ← Auto-created on first run
    ├── cognitive_data.db               ← Main session DB
    ├── feature_log.db                  ← ML training data
    └── lstm_weights.json               ← Saved LSTM weights (after training)
```

---

## 🚀 Setup & Run

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the backend server
```bash
python run.py
```

### 3. Start the React dashboard (development)
```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

### 4. Or build the dashboard (production)
```bash
cd dashboard && npm run build
# Served automatically at http://localhost:8765/dashboard
```

---

## 🔌 API Reference

### Original endpoints
| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/ws` | Real-time data stream |
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/session/summary` | Current session summary |
| `GET` | `/api/sessions` | Recent session history |
| `GET` | `/api/signature` | Personal cognitive signature |
| `POST`| `/api/recalibrate` | Force new calibration |

### New v2 endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/metrics` | Full session metrics |
| `GET` | `/api/metrics/timeline` | Prediction timeline events |
| `GET` | `/api/metrics/graph` | Recharts-ready time series |
| `GET` | `/api/metrics/cards` | Summary card data |
| `GET` | `/api/model/stats` | LSTM model statistics |
| `GET` | `/api/model/predictions` | Last 50 LSTM predictions |
| `POST`| `/api/model/save-weights` | Persist current LSTM weights |
| `GET` | `/api/events` | Recent event bus events |
| `GET` | `/api/events/stats` | Event bus subscriber counts |
| `GET` | `/api/feature-log/stats` | Feature logging pipeline stats |
| `GET` | `/api/db/dashboard` | Full DB snapshot for React |
| `GET` | `/api/db/cognitive-history` | Cognitive state time series |
| `GET` | `/api/db/switch-history` | Switch event history |

---

## 🧠 Module Details

### v1 Modules (unchanged)

| # | Module | Description |
|---|--------|-------------|
| 1 | Context Switch Intelligence | FSM: FOCUSED→PRE_SWITCH→DISENGAGING→SWITCHED→RETURNING→RECOVERING |
| 2 | Cognitive State Model | Estimates Focus / Confusion / Load / Fatigue / Distraction (EMA smoothed) |
| 3 | Procrastination Analysis | 5 layers: trigger detection, probability engine, behaviour learning, intervention, coaching |
| 4 | Cognitive Signature | Personal profile: switch style, recovery speed, primary trigger (7-day rolling) |
| 5 | Temporal Impact Tracker | Measures 6-minute ripple effect after each switch |
| 6 | Meta-Cognition Module | Detects behavioural spirals, delivers self-awareness insights |
| 7 | Recovery Optimizer | Builds personalised 2-4 step recovery plans; learns what works |

### v2 New Features

#### 🤖 LSTM Model Layer (`temporal_model.py`)
- Pure NumPy LSTM — no PyTorch / TensorFlow dependency
- Input: 30-frame rolling window × 12 normalised features
- Output: `switch_probability`, `procrastination_score`, `cognitive_state` (5 classes)
- Confidence scores via entropy-based uncertainty estimation
- Weights save/load via JSON (`POST /api/model/save-weights`)

#### 📦 Feature Logging Pipeline (`feature_logger.py`)
- Async write-ahead queue — never blocks the inference loop
- Logs: raw feature vectors, model predictions, named events
- **Weak supervision**: generates pseudo-labels from behavioural rules
  - `keyboard_inactivity > 8s` → `distracted`
  - `≥3 rapid switches` → `procrastinating`
  - `low motion + no typing > 5s` → `confused`
  - `focus > 0.70 + stable gaze` → `focused`
- Stored in `data/feature_log.db` — ready for offline LSTM training

#### 🔔 Event Bus (`event_bus.py`)
- Async pub/sub architecture — fully decouples modules
- 11 typed events: `SWITCH_DETECTED`, `HIGH_PROCRASTINATION`, `RECOVERY_COMPLETE`, etc.
- Any module can subscribe without changing the publisher

#### 📊 Metrics Dashboard (`metrics.py`)
- `avg_switch_cost` — seconds of productivity lost per switch
- `procrastination_rate` — fraction of time at high risk
- `avg_recovery_time` — mean seconds to regain focus
- `deep_work_duration` — longest single uninterrupted focus block
- `focus_percentage` — % of session classified as focus

---

## 🎯 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| No ground-truth labels for ML | Weak supervision via behavioural pseudo-labels |
| ML without heavy dependencies | Pure NumPy LSTM — runs anywhere Python does |
| Modules becoming tightly coupled | Event bus pub/sub architecture |
| DB writes blocking inference loop | Async write-ahead queue in FeatureLogger |
| One interface not enough | Dual: OpenCV HUD (real-time) + React Dashboard (analytics) |
| Calibration varies per person | Per-user Kalman-filtered baseline + z-score normalisation |

---

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
model:
  weights_path: "data/lstm_weights.json"   # auto-saved after training

thresholds:
  switch_dwell_seconds: 1.5    # lower = more sensitive switch detection
  risk_high: 0.70              # procrastination alert threshold

camera:
  device_id: 0                 # change to 1/2 for external camera
```

---

## 🔬 Research Background

- Context switch cost: **Gloria Mark (UC Irvine)** — avg 23 min to fully recover
- Eye Aspect Ratio (EAR): **Soukupová & Čech (2016)** — real-time blink detection
- Head pose estimation: PnP algorithm with 3D facial landmark model
- Procrastination signals: **Pychyl & Flett (2012)** — emotional regulation model
- Weak supervision: **Ratner et al. (2016)** — Snorkel / data programming paradigm

---

## 🔒 Privacy

**100% local processing.** No video frames leave your device. Only derived numerical metrics are stored in a local SQLite database. No cloud API calls are made at any point.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera won't open | Try `device_id: 1` or `2` in config |
| Face not detected | Improve lighting; face camera directly |
| Too many false switches | Increase `switch_dwell_seconds` to `2.5` |
| Calibration won't complete | Sit still, face camera; takes ~7 seconds |
| React dashboard blank | Run `npm install` then `npm run dev` in `dashboard/` |
| LSTM outputs all similar | Expected — model is randomly initialised until trained |
