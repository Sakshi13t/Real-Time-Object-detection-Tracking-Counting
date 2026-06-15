"""
RTSP Multi-Camera Bag Counting System - Production Integrated
Features: 
- Robust Multi-Camera Display (No blank screens)
- Auto-Reconnection for Cameras
- YOLOv8 + SORT Tracking + Line Counting
- Relay Control (Start/Stop Belt)
- Java API Integration
- Video Recording on Target
- DEBUG: Full payload logging on every API call
- FIX: Video path sent as raw file path (not HTTP URL)
- FIX: Crash prevention — global exception handler + watchdog thread
"""

import cv2
import numpy as np
import time
from datetime import datetime
import logging
import os
import sys
import socket
from threading import Thread, Lock
from ultralytics import YOLO
from sort import Sort
import torch
from collections import defaultdict
import requests
from flask import Flask, request, jsonify, send_from_directory
import signal
import traceback

# --- Force RTSP to use TCP ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ==============================================================================
# GSTREAMER DETECTION & NVENC WRITER PIPELINE
# ==============================================================================
def _cv2_has_gstreamer() -> bool:
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    for line in info.splitlines():
        low = line.lower()
        if "gstreamer" in low and "yes" in low:
            return True
    return False

CV2_GSTREAMER_AVAILABLE = _cv2_has_gstreamer()

def _build_writer_pipeline(path: str, width: int, height: int,
                           fps: int, bitrate_bps: int = 4_000_000) -> str:
    """GStreamer NVENC pipeline: appsrc → nvv4l2h264enc → mp4mux → filesink.
    Downscales to 720p on the GPU before encode to halve NVENC workload."""
    OUT_W, OUT_H = 1280, 720
    if width <= OUT_W and height <= OUT_H:
        OUT_W, OUT_H = width, height
    return (
        f'appsrc ! '
        f'video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! '
        f'queue max-size-buffers=4 leaky=downstream ! '
        f'videoconvert ! '
        f'video/x-raw,format=I420,colorimetry=bt709,chroma-site=mpeg2 ! '
        f'nvvidconv ! video/x-raw(memory:NVMM),format=I420,width={OUT_W},height={OUT_H} ! '
        f'nvv4l2h264enc bitrate={bitrate_bps} preset-level=1 '
        f'insert-sps-pps=true iframeinterval={int(fps)*2} '
        f'EnableTwopassCBR=0 ! '
        f'h264parse ! '
        f'video/x-h264,stream-format=avc,alignment=au,profile=main ! '
        f'mp4mux faststart=true ! '
        f'filesink location="{path}" sync=false'
    )

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("production_bag_counting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# GLOBAL EXCEPTION HANDLER — catches any unhandled exception in main thread
# Prevents silent crash; logs full traceback so you know exactly what failed.
# ==============================================================================
def global_exception_handler(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.critical("💥 UNHANDLED EXCEPTION — Application crash prevented:")
    logger.critical("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

sys.excepthook = global_exception_handler

# --- Handle Segmentation Faults ---
def handle_segfault(sig, frame):
    logger.error("❌ Segmentation fault detected!")
    sys.exit(1)

signal.signal(signal.SIGSEGV, handle_segfault)

# === GLOBALS ===
models = {}
lock = Lock()
targets = {}
relays_tripped = {}
relay_configs = {}
live_counts = {}
vehicle_info = {}
last_command_time = {}
video_threads = {}
loader_params = {}

PYTHON_API_PORT = 8888
JAVA_COUNT_UPDATE_API_URL = ""
JETSON_IP = "10.236.237.27" 
JETSON_VIDEO_PORT = 1234   
OUTPUT_VIDEO_DIR = "/media/amazin/store/output_videos"

# === FLASK SERVER SETUP ===
flask_app = Flask(__name__)

# ==============================================================================
# UTILITY FUNCTIONS (Relays & Config)
# ==============================================================================

def start_belt(loader_name):
    """Start conveyor belt via relay (Non-blocking)"""
    global last_command_time
    COOLDOWN_SECONDS = 2

    with lock:
        last_time = last_command_time.get(loader_name, 0)
        current_time = time.time()
        if current_time - last_time < COOLDOWN_SECONDS:
            return False

    if loader_name not in relay_configs:
        return False

    config = relay_configs[loader_name]
    ip = config.get('ip')
    port = relay_configs.get('port', 5000)
    command = config.get('relay_start_cmd', '*R4#0#$#')

    if not all([ip, port, command]):
        return False

    def _send():
        for attempt in range(3):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    s.connect((ip, port))
                    s.sendall(command.encode('utf-8'))
                    with lock:
                        last_command_time[loader_name] = time.time()
                    logger.info(f"✅ Belt started: {loader_name}")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Belt start attempt {attempt+1} failed: {e}")
                time.sleep(0.5)

    Thread(target=_send, daemon=True).start()
    return True

def stop_belt_immediate(loader_name):
    """Stop conveyor belt via relay IMMEDIATELY (Blocking, no cooldown)."""
    if loader_name not in relay_configs:
        return False

    config = relay_configs[loader_name]
    ip = config.get('ip')
    port = relay_configs.get('port', 5000)
    command = config.get('relay_stop_cmd', '*R4#1#$#')

    if not all([ip, port, command]):
        return False

    for attempt in range(3):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(0.3)
                s.connect((ip, port))
                s.sendall(command.encode('utf-8'))
                with lock:
                    last_command_time[loader_name] = time.time()
                logger.info(f"🛑 Belt IMMEDIATELY stopped: {loader_name}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Immediate belt stop attempt {attempt+1} failed: {e}")
    return False

def load_config_properties(file_path="config.properties"):
    """Load configuration from properties file"""
    global PYTHON_API_PORT, JAVA_COUNT_UPDATE_API_URL, loader_params, relay_configs

    camera_urls = [None, None]
    loader_names_list = [None, None]
    model_paths = [None, None]

    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "=" not in line: continue

                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()

                if key == "PYTHON_API_PORT": PYTHON_API_PORT = int(value)
                elif key == "JAVA_COUNT_UPDATE_API_URL": JAVA_COUNT_UPDATE_API_URL = value
                elif key == "JETSON_IP": JETSON_IP = value 
                elif key == "JETSON_VIDEO_PORT": JETSON_VIDEO_PORT = int(value)  
                elif key == "relay_port": relay_configs['port'] = int(value)

                for i in range(1, 3):
                    idx = i - 1
                    suffix = f"_{i}"

                    if key == f"camera{suffix}": camera_urls[idx] = value
                    elif key == f"loader_name{suffix}":
                        loader_names_list[idx] = value
                        relay_configs[value] = relay_configs.get(value, {})
                        loader_params[value] = loader_params.get(value, {})

                    elif loader_names_list[idx]:
                        lname = loader_names_list[idx]
                        if key == f"model_path{suffix}":
                            model_paths[idx] = value
                            loader_params[lname]['model_path'] = value
                        elif key == f"relay_ip{suffix}": relay_configs[lname]['ip'] = value
                        elif key == f"relay_start_cmd{suffix}": relay_configs[lname]['relay_start_cmd'] = value
                        elif key == f"relay_stop_cmd{suffix}": relay_configs[lname]['relay_stop_cmd'] = value
                        elif key == f"confidence{suffix}": loader_params[lname]['confidence'] = float(value)
                        elif key == f"max_age{suffix}": loader_params[lname]['max_age'] = int(value)
                        elif key == f"min_hits{suffix}": loader_params[lname]['min_hits'] = int(value)
                        elif key == f"iou_threshold{suffix}": loader_params[lname]['iou_threshold'] = float(value)
                        elif key == f"counting_line_y{suffix}": loader_params[lname]['counting_line_y'] = float(value)
                        elif key == f"counting_direction{suffix}": loader_params[lname]['counting_direction'] = value
                        elif key == f"center_dot_position{suffix}": loader_params[lname]['center_dot_position'] = float(value)
                        elif key == f"pre_trigger_offset{suffix}": loader_params[lname]['pre_trigger_offset'] = int(value)
                        elif key == f"recording_fps{suffix}": loader_params[lname]['recording_fps'] = float(value)

    except Exception as e:
        logger.error(f"❌ Failed to read config file: {e}")
        raise

    if 'port' not in relay_configs: relay_configs['port'] = 5000
    if not os.path.exists(OUTPUT_VIDEO_DIR): os.makedirs(OUTPUT_VIDEO_DIR)

    final_cameras = []
    final_loaders = []
    final_models = []

    for i in range(2):
        if camera_urls[i] and loader_names_list[i]:
            final_cameras.append(camera_urls[i])
            final_loaders.append(loader_names_list[i])
            final_models.append(model_paths[i])

    return final_cameras, final_loaders, final_models

def load_models(model_paths, loader_names):
    global models
    for idx, model_path in enumerate(model_paths):
        loader_name = loader_names[idx]
        logger.info(f"Loading YOLO model for {loader_name}: {model_path}")
        try:
            model = YOLO(model_path, task="detect")
            if torch.cuda.is_available():
                model.to('cuda')
                try: model.half()
                except: pass
                logger.info(f"✅ {loader_name} model loaded on GPU (FP16)")
            models[loader_name] = model
        except Exception as e:
            logger.error(f"❌ Failed to load model for {loader_name}: {e}")
            raise

def resize_image_keeping_aspect_ratio(image, target_height):
    h, w = image.shape[:2]
    if h == 0: return image
    aspect = w / h
    new_w = int(target_height * aspect)
    return cv2.resize(image, (new_w, target_height))

# ==============================================================================
# FLASK API
# ==============================================================================
@flask_app.route('/api/getTargetForAI', methods=['POST'])
def get_target_from_java():
    try:
        raw_data = request.get_data(as_text=True)
        content_type = request.content_type
        logger.info(f"📥 [API] Incoming request | Content-Type: {content_type} | Raw body: {raw_data}")

        data = request.get_json(silent=True)

        if data is None:
            logger.error(f"❌ [API] get_json() returned None — Content-Type: '{content_type}' | Body: '{raw_data}'")
            return jsonify({"status": "error", "message": "No JSON or invalid Content-Type"}), 400

        logger.info(f"📋 [API] Parsed JSON payload: {data}")

        loader_name = data.get("loader")
        target_bags_str = data.get("total_bags")
        vehicle_number = data.get("vehicle_number", "NA")
        material_type = data.get("materialType", "NA")
        start_time = data.get("startTime", "NA")

        logger.info(f"🔍 [API] Fields — loader='{loader_name}' | total_bags='{target_bags_str}' | "
                    f"vehicle_number='{vehicle_number}' | materialType='{material_type}' | startTime='{start_time}'")

        if not loader_name:
            logger.error(f"❌ [API] Missing 'loader' field in payload: {data}")
            return jsonify({"status": "error", "message": "Missing 'loader' field"}), 400

        if not target_bags_str:
            logger.error(f"❌ [API] Missing 'total_bags' field in payload: {data}")
            return jsonify({"status": "error", "message": "Missing 'total_bags' field"}), 400

        try:
            target_bags = int(target_bags_str)
        except (ValueError, TypeError) as e:
            logger.error(f"❌ [API] 'total_bags' value '{target_bags_str}' cannot be converted to int: {e}")
            return jsonify({"status": "error", "message": f"total_bags must be int, got: '{target_bags_str}'"}), 400

        new_vehicle_info = {
            "vehicleNumber": vehicle_number,
            "materialType": material_type,
            "startTime": start_time,
            "stopTime": "NA"
        }

        worker = video_threads.get(loader_name)

        # ── KEY LOGIC: Check if a job is currently running ───────────────────
        with lock:
            current_target = targets.get(loader_name, 0)
            current_count = live_counts.get(loader_name, 0)
            is_currently_counting = (
                current_target > 0 and
                current_count > 0 and
                not relays_tripped.get(loader_name, False)
            )

        if is_currently_counting:
            # Job in progress — queue this target for after completion
            if worker:
                worker.pending_target = target_bags
                worker.pending_vehicle_info = new_vehicle_info
            logger.warning(f"⏳ [API] Loader {loader_name} is currently counting "
                           f"({current_count}/{current_target}). "
                           f"New target {target_bags} for {vehicle_number} QUEUED — "
                           f"will apply after current job completes.")
            return jsonify({
                "status": "queued",
                "message": f"Loader busy ({current_count}/{current_target}). Target queued.",
                "queued_target": target_bags,
                "queued_vehicle": vehicle_number
            }), 200  # 200 not 400 — Java should treat this as accepted

        # ── No active job — apply immediately ────────────────────────────────
        with lock:
            targets[loader_name] = target_bags
            relays_tripped[loader_name] = False
            vehicle_info[loader_name] = new_vehicle_info

        video_path = "N/A"
        if worker:
            worker.reset_counter()
            video_path = worker.output_path or "N/A"

        start_belt(loader_name)
        logger.info(f"🎯 [API] Target SET — loader={loader_name} | target={target_bags} | video_path={video_path}")
        return jsonify({"status": "success", "video_path": video_path}), 200

    except Exception as e:
        logger.error(f"❌ [API] Unhandled exception: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
    
# @flask_app.route('/api/getTargetForAI', methods=['POST'])
# def get_target_from_java():
#     try:
#         # ── DEBUG: Log raw incoming payload every time ────────────────────────
#         raw_data = request.get_data(as_text=True)
#         content_type = request.content_type
#         logger.info(f"📥 [API] Incoming request | Content-Type: {content_type} | Raw body: {raw_data}")

#         data = request.get_json(silent=True)

#         # ── DEBUG: Log parsed result ──────────────────────────────────────────
#         if data is None:
#             logger.error(f"❌ [API] get_json() returned None — JSON parsing failed. "
#                          f"Content-Type was: '{content_type}'. Raw body was: '{raw_data}'")
#             return jsonify({"status": "error", "message": "No JSON or invalid Content-Type"}), 400

#         logger.info(f"📋 [API] Parsed JSON payload: {data}")

#         loader_name = data.get("loader")
#         target_bags_str = data.get("total_bags")
#         vehicle_number = data.get("vehicle_number", "NA")
#         material_type = data.get("materialType", "NA")
#         start_time = data.get("startTime", "NA")

#         # ── DEBUG: Log each extracted field ──────────────────────────────────
#         logger.info(f"🔍 [API] Fields — loader='{loader_name}' | total_bags='{target_bags_str}' | "
#                     f"vehicle_number='{vehicle_number}' | materialType='{material_type}' | startTime='{start_time}'")

#         if not loader_name:
#             logger.error(f"❌ [API] Missing 'loader' field in payload: {data}")
#             return jsonify({"status": "error", "message": "Missing 'loader' field"}), 400

#         if not target_bags_str:
#             logger.error(f"❌ [API] Missing 'total_bags' field in payload: {data}")
#             return jsonify({"status": "error", "message": "Missing 'total_bags' field"}), 400

#         try:
#             target_bags = int(target_bags_str)
#         except (ValueError, TypeError) as e:
#             logger.error(f"❌ [API] 'total_bags' value '{target_bags_str}' cannot be converted to int: {e}")
#             return jsonify({"status": "error", "message": f"total_bags must be int, got: '{target_bags_str}'"}), 400

#         with lock:
#             targets[loader_name] = target_bags
#             relays_tripped[loader_name] = False
#             vehicle_info[loader_name] = {
#                 "vehicleNumber": vehicle_number,
#                 "materialType": material_type,
#                 "startTime": start_time,
#                 "stopTime": "NA"
#             }

#         video_path = "N/A"
#         if loader_name in video_threads:
#             video_threads[loader_name].reset_counter()
#             video_path = video_threads[loader_name].output_path or "N/A"

#         start_belt(loader_name)
#         logger.info(f"🎯 [API] Target SET — loader={loader_name} | target={target_bags} bags | video_path={video_path}")
#         return jsonify({"status": "success", "video_path": video_path}), 200

#     except Exception as e:
#         logger.error(f"❌ [API] Unhandled exception in get_target_from_java: {traceback.format_exc()}")
#         return jsonify({"status": "error", "message": str(e)}), 500

@flask_app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory(OUTPUT_VIDEO_DIR, filename)

@flask_app.route('/videos')
def list_videos():
    files = sorted(os.listdir(OUTPUT_VIDEO_DIR))
    videos = [f for f in files if f.endswith(('.avi', '.mp4'))]
    links = "".join(f'<li><a href="/videos/{f}">{f}</a></li>' for f in videos)
    return f"<ul>{links}</ul>"

@flask_app.route('/health')
def health_check():
    """Simple health endpoint — useful for monitoring if the server is alive"""
    with lock:
        status = {
            "status": "ok",
            "uptime_seconds": int(time.time() - APP_START_TIME),
            "loaders": list(video_threads.keys()),
            "live_counts": dict(live_counts),
            "targets": dict(targets),
        }
    return jsonify(status), 200

def start_api_server():
    t = Thread(target=lambda: flask_app.run(
        host='0.0.0.0', port=PYTHON_API_PORT, debug=False, use_reloader=False
    ))
    t.daemon = True
    t.start()
    logger.info(f"✅ Flask API server started on port {PYTHON_API_PORT}")

# ==============================================================================
# WATCHDOG THREAD
# Monitors worker threads every 30s and restarts any that have silently died.
# This is what prevents the application from going dark after a few days.
# ==============================================================================

def watchdog(loader_names, camera_urls):
    """
    Runs forever in background. Every 30 seconds checks if each worker thread
    is still alive. If a worker has died (e.g. due to a GPU crash, memory error,
    or unhandled exception inside the thread), it restarts it automatically.
    Also logs a periodic heartbeat so you can confirm the app is still running
    by tailing the log.
    """
    logger.info("🐕 Watchdog started — monitoring worker threads every 30s")
    url_map = dict(zip(loader_names, camera_urls))

    while True:
        try:
            time.sleep(30)
            logger.info(f"💓 Heartbeat | uptime={int(time.time()-APP_START_TIME)}s | "
                        f"live_counts={dict(live_counts)} | targets={dict(targets)}")

            for name in loader_names:
                worker = video_threads.get(name)
                if worker is None or not worker.is_alive():
                    logger.error(f"💀 Worker '{name}' is DEAD — restarting now...")
                    try:
                        if worker:
                            worker.stop()
                    except Exception:
                        pass
                    url = url_map.get(name)
                    if url:
                        new_worker = RTSPBagCounter(url, name)
                        new_worker.daemon = True
                        new_worker.start()
                        video_threads[name] = new_worker
                        logger.info(f"✅ Worker '{name}' restarted successfully")
                    else:
                        logger.error(f"❌ Cannot restart '{name}' — no URL found in url_map")

        except Exception as e:
            # Watchdog itself must never crash
            logger.error(f"❌ Watchdog error (continuing): {e}\n{traceback.format_exc()}")

# ==============================================================================
# ROBUST THREADED CAMERA
# ==============================================================================

class ThreadedCamera:
    """Robust Camera Reader with Auto-Reconnect"""
    def __init__(self, src):
        self.src = src
        self.lock = Lock()
        self.frame = None
        self.status = False
        self.stopped = False
        self.capture = None
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture is None or not self.capture.isOpened():
                with self.lock: self.status = False
                try:
                    self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                    self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    if not self.capture.isOpened():
                        time.sleep(2.0)
                        continue
                except:
                    time.sleep(2.0)
                    continue

            status, frame = self.capture.read()
            if status and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.status = True
                time.sleep(0.01)
            else:
                with self.lock: self.status = False
                if self.capture: self.capture.release()
                time.sleep(0.5)

    def get_frame(self):
        with self.lock:
            if self.status and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive(): self.thread.join(timeout=1)
        if self.capture: self.capture.release()

# ==============================================================================
# WORKER LOGIC (Detection, Tracking, Recording)
# ==============================================================================

class RTSPBagCounter(Thread):
    def __init__(self, url, loader_name):
        super().__init__()
        self.url = url
        self.loader_name = loader_name
        self.pending_target = None  # Holds next job while current is running
        self.pending_vehicle_info = None
        self.stopped = False
        self.processed_frame = None
        self.frame_lock = Lock()

        params = loader_params.get(loader_name, {})
        self.confidence_threshold = params.get('confidence', 0.28)
        self.counting_line_y = params.get('counting_line_y', 0.45)
        self.counting_direction = params.get('counting_direction', 'down')
        self.center_dot_position = params.get('center_dot_position', 0.25)
        self.pre_trigger_offset = params.get('pre_trigger_offset', 1)
        self.recording_fps = params.get('recording_fps', 15.0)

        self.model = models.get(loader_name)
        self.tracker = Sort(
            max_age=params.get('max_age', 5),
            min_hits=params.get('min_hits', 2),
            iou_threshold=0.3
        )

        self.counted_ids = set()
        self.total_count = 0
        self.tracker_positions = defaultdict(list)
        self.frame_count = 0
        self.start_time = datetime.now()
        self.last_sent_count = None
        self.fps_history = []
        self.target_reached = False

        self.video_writer = None
        self.output_path = None        # final path (what Java sees)
        self._tmp_output_path = None   # .tmp.mp4 written by GStreamer
        self.is_recording = False
        self.height = 0
        self.width = 0

        # Async writer: a dedicated thread drains _write_q so video I/O
        # never blocks the main processing loop.
        from queue import Queue as _Queue
        self._writer_lock = Lock()
        self._write_q = _Queue(maxsize=10)
        self._writer_thread = Thread(target=self._writer_loop, daemon=True,
                                     name=f"writer-{loader_name}")
        self._writer_thread.start()

        logger.info(f"✅ Worker initialized: {loader_name}")

    def _remux_to_final(self, tmp_path: str, final_path: str):
        """Rename .tmp.mp4 to final .mp4 (mp4mux already wrote faststart moov)."""
        try:
            os.rename(tmp_path, final_path)
            logger.info(f"[{self.loader_name}] Recording finalized → {final_path}")
        except Exception as e:
            logger.error(f"[{self.loader_name}] Rename failed: {e}")

    def init_video_writer(self):
        if self.width == 0 or self.height == 0: return
        try:
            with lock:
                v_num = vehicle_info.get(self.loader_name, {}).get("vehicleNumber", "NA")
            v_num = "".join(c for c in v_num if c.isalnum() or c in ('-', '_'))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            writer = None
            backend_used = "none"

            # --- Try NVENC GStreamer pipeline first (produces .mp4) ---
            if CV2_GSTREAMER_AVAILABLE:
                final_path = os.path.join(
                    OUTPUT_VIDEO_DIR,
                    f"{self.loader_name}_{v_num}_{timestamp}.mp4"
                )
                tmp_path = os.path.join(
                    OUTPUT_VIDEO_DIR,
                    f"{self.loader_name}_{v_num}_{timestamp}.tmp.mp4"
                )
                gst_pipeline = _build_writer_pipeline(
                    tmp_path,
                    self.width, self.height,
                    int(self.recording_fps),
                    bitrate_bps=4_000_000,
                )
                try:
                    candidate = cv2.VideoWriter(
                        gst_pipeline, cv2.CAP_GSTREAMER, 0,
                        float(self.recording_fps),
                        (self.width, self.height), True
                    )
                    if candidate.isOpened():
                        writer = candidate
                        self.output_path = final_path      # Java always sees final path
                        self._tmp_output_path = tmp_path   # GStreamer writes here
                        backend_used = "nvenc/h264"
                    else:
                        try:
                            candidate.release()
                        except Exception:
                            pass
                        logger.warning(
                            f"[{self.loader_name}] NVENC writer did not open; "
                            f"falling back to XVID."
                        )
                except Exception as e:
                    logger.warning(
                        f"[{self.loader_name}] NVENC writer exception: {e}; "
                        f"falling back to XVID."
                    )

            # --- Fallback: original XVID software encoder (.avi) ---
            if writer is None:
                avi_path = os.path.join(
                    OUTPUT_VIDEO_DIR,
                    f"{self.loader_name}_{v_num}_{timestamp}.avi"
                )
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(
                    avi_path, fourcc, self.recording_fps,
                    (self.width, self.height)
                )
                if writer.isOpened():
                    self.output_path = avi_path
                    backend_used = "xvid (CPU fallback)"
                else:
                    logger.error(f"[{self.loader_name}] Both NVENC and XVID writers failed to open.")
                    self.is_recording = False
                    return

            with self._writer_lock:
                self.video_writer = writer
                self.is_recording = True
            logger.info(f"📹 Recording [{backend_used}]: {self.output_path}")
        except Exception as e:
            logger.error(f"❌ Video writer error: {e}")
            self.is_recording = False

    def _writer_loop(self):
        """Background thread: drains _write_q and writes frames to disk."""
        from queue import Empty
        while not self.stopped:
            try:
                frame = self._write_q.get(timeout=0.5)
            except Empty:
                continue
            if frame is None:
                continue
            with self._writer_lock:
                w = self.video_writer
                if not self.is_recording or w is None:
                    continue
                try:
                    w.write(frame)
                except Exception as e:
                    logger.warning(f"[{self.loader_name}] writer error: {e}")

    def _release_writer(self):
        """Thread-safe: stop recording, flush queue, release writer, finalize file."""
        from queue import Empty
        with self._writer_lock:
            self.is_recording = False
            w = self.video_writer
            self.video_writer = None

        tmp_path = self._tmp_output_path
        final_path = self.output_path
        self._tmp_output_path = None

        if w is not None:
            # Drain any frames still queued so they make it into the file
            try:
                while True:
                    frame = self._write_q.get_nowait()
                    if frame is not None:
                        w.write(frame)
            except Empty:
                pass
            try:
                w.release()   # sends EOS → flushes GStreamer muxer
            except Exception as e:
                logger.warning(f"[{self.loader_name}] writer release error: {e}")

            # Rename .tmp.mp4 → final .mp4 only for NVENC path
            if tmp_path and final_path and tmp_path != final_path:
                self._remux_to_final(tmp_path, final_path)
        else:
            try:
                from queue import Empty as _E
                while True:
                    self._write_q.get_nowait()
            except _E:
                pass

    def reset_counter(self):
        """Reset triggered by API"""
        self._release_writer()

        self.target_reached = False
        with lock:
            self.counted_ids.clear()
            self.tracker_positions.clear()
            self.total_count = 0
            self.last_sent_count = None
            live_counts[self.loader_name] = 0
            vehicle_info[self.loader_name]["stopTime"] = "NA"

        self.send_live_count(0)
        self.init_video_writer()

    def get_tracking_point(self, x1, y1, x2, y2):
        center_x = (x1 + x2) // 2
        center_y = int(y1 + (y2 - y1) * self.center_dot_position)
        return center_x, center_y

    def has_crossed_line(self, track_id, center_x, center_y, line_y):
        self.tracker_positions[track_id].append((center_x, center_y))
        if len(self.tracker_positions[track_id]) > 10:
            self.tracker_positions[track_id] = self.tracker_positions[track_id][-10:]
        if len(self.tracker_positions[track_id]) < 2: return False

        history = self.tracker_positions[track_id]
        window = history[-4:] if len(history) >= 4 else history
        earliest_y = window[0][1]
        curr_y = history[-1][1]

        if self.counting_direction == 'down':
            return earliest_y < line_y and curr_y >= line_y
        else:
            return earliest_y > line_y and curr_y <= line_y

    def draw_label(self, frame, text, x, y, font_scale, bg_color, text_color=(0, 0, 0), thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 6
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1)
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def process_frame(self, frame):
        frame_start = time.time()
        line_y = int(self.height * self.counting_line_y)

        try:
            results = self.model.predict(frame, imgsz=640, verbose=False, conf=self.confidence_threshold)[0]
        except Exception as e:
            logger.error(f"❌ [{self.loader_name}] YOLO predict failed: {e}")
            return frame

        detections = np.empty((0, 5))
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        try:
            tracked_objects = self.tracker.update(detections)
        except Exception as e:
            logger.error(f"❌ [{self.loader_name}] SORT tracker failed: {e}")
            return frame

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)

            cx, cy = self.get_tracking_point(x1, y1, x2, y2)

            if track_id not in self.counted_ids:
                if self.has_crossed_line(track_id, cx, cy, line_y):
                    self.counted_ids.add(track_id)
                    self.total_count += 1
                    logger.info(f"[{self.loader_name}] Bag Counted: {self.total_count}")

            color = (0, 0, 255) if track_id in self.counted_ids else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.line(frame, (0, line_y), (self.width, line_y), (0, 0, 255), 2)

        with lock:
            target = targets.get(self.loader_name, 0)
            is_tripped = relays_tripped.get(self.loader_name, False)
            live_counts[self.loader_name] = self.total_count

        if self.last_sent_count is None or self.total_count != self.last_sent_count:
            self.send_live_count(self.total_count)
            self.last_sent_count = self.total_count

        if target > 0 and self.total_count >= (target - self.pre_trigger_offset) and not is_tripped:
            self.handle_target_reached(self.total_count)

        elapsed = time.time() - frame_start
        fps = 1 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > 30: self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        CYAN   = (200, 200, 0)
        GREEN  = (0, 230, 100)
        YELLOW = (0, 230, 230)
        DARK   = (10, 10, 10)

        self.draw_label(frame, f"Cam: {self.loader_name}",              10, 30,  0.7,  CYAN,   DARK, 2)
        self.draw_label(frame, f"Count: {self.total_count} / {target}", 10, 68,  1.0,  GREEN,  DARK, 2)
        self.draw_label(frame, f"FPS: {avg_fps:.1f}",                   10, 105, 0.65, YELLOW, DARK, 2)

        if self.is_recording:
            cv2.circle(frame, (self.width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 43, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def run(self):
        """
        Main worker loop wrapped in a broad try/except so that even if something
        unexpected crashes the loop, the exception is logged and the watchdog
        can detect the dead thread and restart it.
        """
        try:
            self.cam = ThreadedCamera(self.url)
            time.sleep(1.0)

            while not self.stopped:
                try:
                    ret, frame = self.cam.get_frame()
                    if not ret:
                        with self.frame_lock: self.processed_frame = None
                        time.sleep(0.1)
                        continue

                    self.height, self.width = frame.shape[:2]

                    if self.target_reached:
                        processed = frame.copy()
                        self.draw_label(processed, f"Cam: {self.loader_name}", 10, 30, 0.7, (200,200,0), (10,10,10), 2)
                        self.draw_label(processed, f"Count: {self.total_count} / {self.total_count} DONE", 10, 68, 1.0, (0,200,200), (10,10,10), 2)
                        with self.frame_lock: self.processed_frame = processed
                        continue

                    with lock:
                        should_process = targets.get(self.loader_name, 0) > 0 or self.is_recording

                    if should_process:
                        processed = self.process_frame(frame)
                        if self.is_recording:
                            try:
                                self._write_q.put_nowait(processed)
                            except Exception:
                                try: self._write_q.get_nowait()
                                except Exception: pass
                                try: self._write_q.put_nowait(processed)
                                except Exception: pass
                    else:
                        processed = frame.copy()
                        cv2.putText(processed, f"{self.loader_name}: IDLE (Waiting for Target)",
                                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    with self.frame_lock: self.processed_frame = processed

                except Exception as e:
                    # Inner loop exception — log and keep running, don't die
                    logger.error(f"❌ [{self.loader_name}] Frame loop error: {e}\n{traceback.format_exc()}")
                    time.sleep(0.5)

        except Exception as e:
            logger.critical(f"💥 [{self.loader_name}] Worker thread CRASHED: {e}\n{traceback.format_exc()}")
            # Do NOT re-raise — let the watchdog detect is_alive()==False and restart

    def stop(self):
        self.stopped = True
        self._release_writer()
        if hasattr(self, 'cam'): self.cam.stop()

    def send_live_count(self, count):
        if not JAVA_COUNT_UPDATE_API_URL: return
        try:
            
            with lock:
                info = vehicle_info.get(self.loader_name, {})
                ip = relay_configs.get(self.loader_name, {}).get('ip', "NA") 
                if self.output_path:
                    filename = os.path.basename(self.output_path)
                    current_video_path = f"http://{JETSON_IP}:{JETSON_VIDEO_PORT}/output_videos/{filename}"
                else:
                    current_video_path = "NA"
                    
                target = targets.get(self.loader_name, 0)

            payload = {
                "loader": self.loader_name, "ip": ip, "actualBags": count, "target": target,
                "vehicleNumber": info.get("vehicleNumber", "NA"),
                "materialType": info.get("materialType", "NA"),
                "status": "counting",
                "startTime": info.get("startTime", "NA"),
                "stopTime": "NA"
            }
            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=0.1)
        except: pass

    # def handle_target_reached(self, final_count):
    #     self.target_reached = True
    #     stop_belt_immediate(self.loader_name)

    #     with lock:
    #         reported_count = targets.get(self.loader_name, final_count)
    #         self.total_count = reported_count
    #         live_counts[self.loader_name] = reported_count
    #         relays_tripped[self.loader_name] = True
    #         vehicle_info[self.loader_name]["stopTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         targets[self.loader_name] = 0

    #     if self.is_recording and self.video_writer:
    #         self.video_writer.release()
    #         self.is_recording = False

    #     self.send_final_count(reported_count)

    def handle_target_reached(self, final_count):
        self.target_reached = True
        stop_belt_immediate(self.loader_name)

        with lock:
            reported_count = targets.get(self.loader_name, final_count)
            self.total_count = reported_count
            live_counts[self.loader_name] = reported_count
            relays_tripped[self.loader_name] = True
            vehicle_info[self.loader_name]["stopTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            targets[self.loader_name] = 0

        self._release_writer()
        self.send_final_count(reported_count)

        # ── Apply queued target if one was held ──────────────────────────────────
        if self.pending_target is not None:
            logger.info(f"▶️ [{self.loader_name}] Applying QUEUED target: "
                        f"{self.pending_target} bags for "
                        f"{self.pending_vehicle_info.get('vehicleNumber', 'NA')}")

            with lock:
                targets[self.loader_name] = self.pending_target
                relays_tripped[self.loader_name] = False
                vehicle_info[self.loader_name] = self.pending_vehicle_info

            # Clear the queue
            self.pending_target = None
            self.pending_vehicle_info = None

            # Reset and start next job
            self.reset_counter()
            start_belt(self.loader_name)
            logger.info(f"🎯 [{self.loader_name}] Queued job started automatically.")
        
    def send_final_count(self, count):
        """
        FIX: sends self.output_path directly as a raw file system path.
        e.g. /media/amazin/store/output_videos/Loader-BC03_hp12g7091_20260506_153247.avi
        """
        if not JAVA_COUNT_UPDATE_API_URL: return
        try:
            with lock:
                info = vehicle_info.get(self.loader_name, {})
                ip = relay_configs.get(self.loader_name, {}).get('ip', "NA")
                # ── FIX: raw file path, not HTTP URL ─────────────────────────
                current_video_path = self.output_path if self.output_path else "NA"

            payload = {
                "loader": self.loader_name,
                "ip": ip,
                "actualBags": count,
                "target": targets.get(self.loader_name, 0),
                "vehicleNumber": info.get("vehicleNumber", "NA"),
                "materialType": info.get("materialType", "NA"),
                "status": "completed",
                "startTime": info.get("startTime", "NA"),
                "stopTime": info.get("stopTime", "NA"),
                "videoPath": current_video_path   # raw path: /media/amazin/store/...
            }

            logger.info(f"📤 [FINAL] Sending to Java | loader={self.loader_name} | "
                        f"actualBags={count} | videoPath={current_video_path} | payload={payload}")

            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=5)
            logger.info(f"✅ Final sent for {self.loader_name} with video path: {current_video_path}")

        except Exception as e:
            logger.error(f"❌ Failed final send for {self.loader_name}: {e}\n{traceback.format_exc()}")

# ==============================================================================
# MAIN DISPLAY LOOP
# ==============================================================================

APP_START_TIME = time.time()

def main():
    logger.info("=" * 60)
    logger.info("RTSP Multi-Camera System - Production Integrated")
    logger.info(f"🚀 Application started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    camera_urls, loader_names, model_paths = load_config_properties()
    if not camera_urls:
        logger.error("❌ No cameras configured.")
        return

    load_models(model_paths, loader_names)
    start_api_server()

    for i, url in enumerate(camera_urls):
        name = loader_names[i]
        worker = RTSPBagCounter(url, name)
        worker.daemon = True
        worker.start()
        video_threads[name] = worker

    # ── Start watchdog AFTER workers are registered ───────────────────────────
    wd = Thread(target=watchdog, args=(loader_names, camera_urls), daemon=True)
    wd.start()

    logger.info("✅ Workers + Watchdog started. Opening Display...")

    window_name = "Multi-Camera Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 720)

    try:
        while True:
            display_images = []

            for name in loader_names:
                worker = video_threads.get(name)
                frame = None

                if worker:
                    with worker.frame_lock:
                        if worker.processed_frame is not None:
                            frame = worker.processed_frame.copy()

                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{name}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, "CONNECTING...", (20, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(frame, (0, 0), (639, 479), (50, 50, 50), 4)

                display_images.append(frame)

            if display_images:
                target_h = display_images[0].shape[0]
                resized = [resize_image_keeping_aspect_ratio(img, target_h) for img in display_images]

                try:
                    combined = np.hstack(resized)
                    if combined.shape[1] > 1920:
                        combined = resize_image_keeping_aspect_ratio(combined, 600)
                    cv2.imshow(window_name, combined)
                except Exception as e:
                    logger.error(f"Display error: {e}")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("🛑 Shutting down via KeyboardInterrupt...")
    except Exception as e:
        logger.critical(f"💥 Main display loop crashed: {e}\n{traceback.format_exc()}")
    finally:
        logger.info("🧹 Cleaning up workers...")
        for w in video_threads.values():
            try: w.stop()
            except: pass
        cv2.destroyAllWindows()
        logger.info("👋 Application exited cleanly.")
        sys.exit(0)

if __name__ == "__main__":
    main()
