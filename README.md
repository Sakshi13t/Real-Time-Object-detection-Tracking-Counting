# Real-Time Object Detection, Tracking & Counting
### Production-grade bag counting system deployed on NVIDIA Jetson Orin — 24/7 industrial operation

![System Status](https://img.shields.io/badge/status-production-brightgreen)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson%20Orin-76b900)
![Framework](https://img.shields.io/badge/model-YOLOv8-00BFFF)
![Optimization](https://img.shields.io/badge/inference-TensorRT-orange)

---

## Overview

A real-time computer vision system that detects, tracks, and counts cement bags on production lines using YOLOv8 + SORT tracking, deployed on NVIDIA Jetson AGX Orin / Orin Nano edge devices. The system runs 24/7 across 4+ cement manufacturing plants with 99% counting accuracy.

The inference pipeline was optimised with **TensorRT + GStreamer**, reducing latency from ~200ms to ~100ms at 15–20 FPS — without any loss in accuracy.

---

## Demo

<!-- Add your screenshots here once captured -->
> Screenshots coming soon — system is deployed on-site. Output includes live bounding boxes, track IDs, and a running bag count overlaid on the video feed.

**Suggested screenshots to add:**
- Live feed with bounding boxes + track IDs
- Console output showing FPS and count
- Jetson device running the system (optional)

---

## Key Results

| Metric | Value |
|---|---|
| Counting accuracy | 99% |
| Inference latency (before optimisation) | ~200ms |
| Inference latency (after TensorRT) | ~100ms |
| Throughput | 15–20 FPS |
| Deployment | 24/7, 4+ cement plants |
| Hardware | NVIDIA Jetson AGX Orin / Orin Nano |

---

## System Architecture

```
Camera Feed (GStreamer pipeline)
        │
        ▼
  YOLOv8 Detection
  (TensorRT engine)
        │
        ▼
  SORT Tracker
  (Hungarian algorithm + Kalman filter)
        │
        ▼
  Counting Logic
  (line-crossing / zone-based)
        │
        ▼
  Annotated Output + Count Display
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Object tracking | SORT (Simple Online Realtime Tracking) |
| Inference optimisation | TensorRT |
| Video pipeline | GStreamer |
| Edge hardware | NVIDIA Jetson AGX Orin, Orin Nano |
| Language | Python |
| Libraries | OpenCV, NumPy, PyTorch |

---

## How It Works

**1. Detection**
Each frame is passed through a YOLOv8 model exported to TensorRT engine format. TensorRT optimises the model for the Jetson's GPU using FP16 precision, significantly reducing inference time without accuracy loss.

**2. Tracking**
Detections are passed to a SORT tracker, which assigns persistent IDs to each detected bag across frames using a Kalman filter for motion prediction and the Hungarian algorithm for ID assignment.

**3. Counting**
A virtual counting line or zone is defined in the frame. When a tracked object crosses the line (based on centroid position), the count increments. Each track ID is counted only once to avoid duplicates.

**4. GStreamer Pipeline**
Video input is handled via GStreamer for efficient hardware-accelerated decoding on Jetson, reducing CPU overhead and maintaining smooth throughput at 15–20 FPS.

---

## Project Structure

```
├── detect_track_count.py     # Main inference + tracking + counting script
├── sort.py                   # SORT tracker implementation
├── utils/
│   ├── line_counter.py       # Line-crossing counting logic
│   └── visualiser.py         # Bounding box + annotation drawing
├── models/
│   └── best.engine           # TensorRT engine (not included — generate from best.pt)
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
# Hardware
NVIDIA Jetson AGX Orin or Orin Nano
JetPack 6.x or later

# Python dependencies
pip install ultralytics opencv-python numpy scipy filterpy
```

### Export YOLOv8 model to TensorRT

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="engine", half=True, device=0)
```

### Run the system

```bash
python detect_track_count.py \
  --model models/best.engine \
  --source 0 \          # camera index or RTSP stream
  --count-line 400      # y-coordinate of counting line
```

---

## Deployment Context

This system is part of a broader Industrial AI platform built for cement manufacturing. It operates continuously in dusty, high-vibration industrial environments — edge deployment on Jetson was chosen over cloud inference to eliminate network latency and ensure operation even during connectivity loss.

---

## Related Projects

Other production AI systems built as part of the same platform:

- **Packer Efficiency Monitoring** — YOLOv8 + Flask REST API + React dashboard for shift-wise production reporting
- **Predictive Maintenance Platform** — XGBoost pipeline predicting equipment failure 30 minutes ahead with 94%+ accuracy
- **Firlobot** — LLM Text-to-SQL chatbot (LLaMA 3.3-70B) for natural language data access

---

## Author

**Sakshi Tandon** — Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/sakshi-tandon-865371249/) · [GitHub](https://github.com/Sakshi13t)
