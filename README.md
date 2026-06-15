# Real-Time Bag Counting System
### Production-deployed multi-camera bag counting on NVIDIA Jetson Orin — 24/7 industrial operation

![Status](https://img.shields.io/badge/status-production-brightgreen)
![Platform](https://img.shields.io/badge/platform-NVIDIA%20Jetson%20Orin-76b900)
![Model](https://img.shields.io/badge/model-YOLOv8-00BFFF)
![Inference](https://img.shields.io/badge/inference-TensorRT%20%7C%20FP16-orange)
![Backend](https://img.shields.io/badge/api-Flask%20REST-blue)

---

## Overview

A production-grade computer vision system that detects, tracks, and counts cement bags on conveyor belts in real time. Deployed 24/7 across 4+ industrial cement plants, the system integrates:

- **YOLOv8** object detection (TensorRT FP16 on Jetson GPU)
- **SORT tracking** with Kalman filter + Hungarian algorithm
- **Line-crossing counting** logic per camera/loader
- **Flask REST API** for integration with a Java plant management backend
- **Relay control** for automated conveyor belt start/stop
- **GStreamer NVENC pipeline** for hardware-accelerated MP4 recording
- **Watchdog thread** for automatic crash recovery without human intervention

Inference latency was reduced from ~200ms to ~100ms through TensorRT optimisation + GStreamer hardware-accelerated decoding at 15–20 FPS.

---

## Demo

<img width="1912" height="893" alt="image" src="https://github.com/user-attachments/assets/f8c6f442-571e-4229-8770-5bd493924962" />
<img width="1911" height="885" alt="image" src="https://github.com/user-attachments/assets/d4e88584-6019-4de8-ac67-ca84e595c99f" />

**Planned screenshots:**
- Live feed with bounding boxes, track IDs, and bag count overlay
- Multi-camera display grid
- `/health` API response showing live counts

---

## Key Metrics

| Metric | Value |
|---|---|
| Counting accuracy | 99% |
| Inference latency (before) | ~200ms |
| Inference latency (after TensorRT) | ~100ms |
| Throughput | 15–20 FPS |
| Deployment | 24/7, 4+ cement plants |
| Cameras supported | Multi-camera (RTSP) |
| Hardware | NVIDIA Jetson AGX Orin / Orin Nano |

---

## System Architecture

```
RTSP Camera Streams (per loader/belt)
        │
        ▼
┌──────────────────────────────────┐
│  ThreadedCamera (per camera)     │  ← auto-reconnect on disconnect
│  cv2.VideoCapture (TCP RTSP)     │
└──────────────┬───────────────────┘
               │ frames
               ▼
┌──────────────────────────────────┐
│  RTSPBagCounter (Thread)         │
│  ┌─────────────────────────┐     │
│  │ YOLOv8 Detection        │     │  ← CUDA FP16 (TensorRT engine)
│  │ confidence threshold     │     │
│  └──────────┬──────────────┘     │
│             │ detections         │
│  ┌──────────▼──────────────┐     │
│  │ SORT Tracker            │     │  ← Kalman filter + Hungarian algo
│  │ max_age, min_hits, IOU  │     │
│  └──────────┬──────────────┘     │
│             │ tracked IDs        │
│  ┌──────────▼──────────────┐     │
│  │ Line Crossing Counter   │     │  ← configurable Y-line per camera
│  │ direction-aware         │     │
│  └──────────┬──────────────┘     │
│             │ count events       │
│  ┌──────────▼──────────────┐     │
│  │ GStreamer NVENC Writer  │     │  ← async write queue, hardware MP4
│  └─────────────────────────┘     │
└──────────────────────────────────┘
        │ live/final counts
        ▼
┌──────────────────────────────────┐
│  Flask REST API (port 8888)      │
│  POST /api/getTargetForAI        │  ← receives job from Java backend
│  GET  /health                    │
│  GET  /videos                    │
└──────────────┬───────────────────┘
               │ count updates
               ▼
┌──────────────────────────────────┐
│  Java Plant Management Backend   │
│  POST /api/send-count            │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Relay Controller (TCP socket)   │  ← start/stop conveyor belt
│  start_belt() / stop_belt()      │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│  Watchdog Thread (every 30s)     │  ← restarts dead workers automatically
│  + Global Exception Handler      │
│  + SIGSEGV handler               │
└──────────────────────────────────┘
```

---

## Key Components

### `RTSPBagCounter` (core worker thread)
Each camera runs its own `RTSPBagCounter` thread. It handles detection, tracking, counting, relay signalling, and video recording independently. Key methods:

| Method | Purpose |
|---|---|
| `process_frame()` | YOLOv8 inference → SORT tracking → line crossing check → count update |
| `has_crossed_line()` | Direction-aware line crossing using track position history |
| `handle_target_reached()` | Fires when count == target: stops belt, sends final count, queues next job |
| `reset_counter()` | Clears counted IDs and resets state for a new vehicle/job |
| `init_video_writer()` | Opens GStreamer NVENC pipeline; falls back to XVID if unavailable |

### `ThreadedCamera`
Dedicated thread per camera for frame capture with auto-reconnect on RTSP disconnect. Prevents the main processing thread from blocking on network I/O.

### Flask REST API
Receives job targets from the Java backend and reports counts back:

```
POST /api/getTargetForAI    — set target bag count + vehicle info for a loader
GET  /health                — uptime, live counts, active loaders
GET  /videos                — list recorded MP4 files
GET  /videos/<filename>     — serve a recording
```

Job queueing is built in: if a loader is mid-count when a new job arrives, the new target is held in `pending_target` and applied automatically on completion.

### Watchdog Thread
Runs every 30 seconds. Checks each worker thread's `is_alive()` status. Dead workers are restarted automatically — this is what keeps the system running for weeks without manual intervention.

### GStreamer NVENC Pipeline
Video is recorded using hardware H.264 encoding on the Jetson GPU:
```
appsrc → videoconvert → nvvidconv (720p GPU downscale) → nvv4l2h264enc → mp4mux → filesink
```
Falls back to XVID software encoding if GStreamer is unavailable.

---

## Configuration

All parameters are loaded from `config.properties` — no hardcoding required:

```properties
# API ports
PYTHON_API_PORT=8888
JAVA_COUNT_UPDATE_API_URL=http://<java-server>:5050/api/send-count

# Jetson device
JETSON_IP=<jetson-ip>
JETSON_VIDEO_PORT=1234

# Camera / Loader config (supports 2 cameras)
camera_1=rtsp://user:pass@<camera-ip>:554/cam/realmonitor?channel=1&subtype=0
loader_name_1=Loader-BC03
relay_ip_1=<relay-ip>
relay_start_cmd_1=*R1#0#$#
relay_stop_cmd_1=*R1#1#10#$#

# Per-loader model and detection tuning
model_path_1=best.pt
confidence_1=0.25
max_age_1=5
min_hits_1=1
iou_threshold_1=0.3

# Counting line (as fraction of frame height)
counting_line_y_1=0.55
counting_direction_1=down
center_dot_position_1=0.5
```

---

## Setup & Usage

### Requirements

```
Hardware:   NVIDIA Jetson AGX Orin or Orin Nano (JetPack 5.x+)
Python:     3.8+
```

```bash
pip install ultralytics opencv-python numpy scipy requests flask torch
```

SORT tracker (`sort.py`) must be present in the same directory.  
Source: https://github.com/abewley/sort

### Export YOLOv8 model to TensorRT

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="engine", half=True, device=0)  # FP16 TensorRT engine
```

### Configure

Edit `config.properties` with your camera RTSP URLs, relay IPs, and loader names.

### Run

```bash
python production_bag_counting.py
```

The system will:
1. Load config and models
2. Start `ThreadedCamera` + `RTSPBagCounter` for each configured loader
3. Start the Flask API server on port 8888
4. Start the watchdog thread
5. Begin processing frames — waiting for a job target from the Java backend

---

## File Structure

```
├── production_bag_counting.py   # Main system — all logic
├── sort.py                      # SORT tracker (Kalman filter + Hungarian)
├── config.properties            # All runtime config (cameras, models, relays)
├── best.pt                      # YOLOv8 trained weights (not included)
├── best.engine                  # TensorRT engine — generated from best.pt
└── production_bag_counting.log  # Auto-generated runtime log
```

---

## Deployment Context

Built for industrial cement manufacturing environments — dusty, high-vibration, with intermittent network connectivity. Edge deployment on Jetson was chosen over cloud inference to:
- Eliminate network latency in the counting loop
- Maintain operation during connectivity loss
- Keep sensitive plant camera feeds on-premise

The system has run continuously for months across multiple plants with zero manual restarts, enabled by the watchdog thread and global exception handling.

---

## Related Projects

Other production AI systems from the same industrial platform:

- **Packer Efficiency Monitoring** — YOLOv8 + Flask REST API + React dashboard for shift-wise production reporting
- **Predictive Maintenance Platform** — 2-stage XGBoost pipeline predicting equipment failure 30 minutes ahead with 94%+ accuracy
- **Firlobot** — LLM Text-to-SQL chatbot (LLaMA 3.3-70B) enabling natural language data queries for non-technical operators

---

## Author

**Sakshi Tandon** — Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/sakshi-tandon-865371249) · [GitHub](https://github.com/Sakshi13t)
