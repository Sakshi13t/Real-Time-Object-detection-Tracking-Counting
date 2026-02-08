"""
RTSP Camera Bag Counting System - Production Integrated
Real-time bag counting with Java API integration and relay control
"""

import cv2
import numpy as np
import time
from datetime import datetime
import logging
from pathlib import Path
import json
import os
import sys
import socket
from threading import Thread, Lock
from ultralytics import YOLO
from sort import Sort
import torch
from collections import defaultdict
import requests
from flask import Flask, request, jsonify
import signal
import traceback

# --- Force RTSP to use TCP ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("production_bag_counting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Handle Segmentation Faults ---
def handle_segfault(sig, frame):
    logger.error("❌ Segmentation fault detected!")
    sys.exit(1)

signal.signal(signal.SIGSEGV, handle_segfault)

# === GLOBALS ===
models = {}  # Per-loader models
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
OUTPUT_VIDEO_DIR = "/media/nvidia/My Passport/output_videos"

# === FLASK SERVER SETUP ===
flask_app = Flask(__name__)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def start_belt(loader_name):
    """Start conveyor belt via relay"""
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
   
    for attempt in range(5):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((ip, port))
                s.sendall(command.encode('utf-8'))
                try:
                    s.recv(1024)
                except socket.timeout:
                    pass
                with lock:
                    last_command_time[loader_name] = time.time()
                logger.info(f"✅ Belt started: {loader_name}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Belt start attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return False


def stop_belt(loader_name):
    """Stop conveyor belt via relay"""
    global last_command_time
    COOLDOWN_SECONDS = 1
    
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
    command = config.get('relay_stop_cmd', '*R4#1#$#')

    if not all([ip, port, command]):
        return False
   
    for attempt in range(5):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((ip, port))
                s.sendall(command.encode('utf-8'))
                try:
                    s.recv(1024)
                except socket.timeout:
                    pass
                with lock:
                    last_command_time[loader_name] = time.time()
                logger.info(f"🛑 Belt stopped: {loader_name}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Belt stop attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return False


def load_config_properties(file_path="config.properties"):
    """Load configuration from properties file"""
    global PYTHON_API_PORT, JAVA_COUNT_UPDATE_API_URL, loader_params, relay_configs
    
    loaded_urls = [None, None]
    loader_names = [None, None]
    model_paths = [None, None]  # Separate model paths
   
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                    
                try:
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip()
                    
                    if key == "PYTHON_API_PORT":
                        PYTHON_API_PORT = int(value)
                    elif key == "JAVA_COUNT_UPDATE_API_URL":
                        JAVA_COUNT_UPDATE_API_URL = value
                    elif key == "relay_port":
                        relay_configs['port'] = int(value)
                    elif key == "camera_1":
                        loaded_urls[0] = value
                    elif key == "camera_2":
                        loaded_urls[1] = value
                   
                    # Loader Configs
                    for l_idx in ["1", "2"]:
                        idx_num = int(l_idx) - 1
                        
                        if key == f"loader_name_{l_idx}":
                            loader_names[idx_num] = value
                            relay_configs[value] = relay_configs.get(value, {})
                            loader_params[value] = loader_params.get(value, {})
                            
                        elif key == f"model_path_{l_idx}" and loader_names[idx_num]:
                            model_paths[idx_num] = value
                            loader_params[loader_names[idx_num]]['model_path'] = value
                            
                        elif key == f"relay_ip_{l_idx}" and loader_names[idx_num]:
                            relay_configs[loader_names[idx_num]]['ip'] = value
                            
                        elif key == f"relay_start_cmd_{l_idx}" and loader_names[idx_num]:
                            relay_configs[loader_names[idx_num]]['relay_start_cmd'] = value
                            
                        elif key == f"relay_stop_cmd_{l_idx}" and loader_names[idx_num]:
                            relay_configs[loader_names[idx_num]]['relay_stop_cmd'] = value
                        
                        # Per-loader parameters
                        elif key == f"confidence_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['confidence'] = float(value)
                            
                        elif key == f"max_age_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['max_age'] = int(value)
                            
                        elif key == f"min_hits_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['min_hits'] = int(value)
                            
                        elif key == f"iou_threshold_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['iou_threshold'] = float(value)
                            
                        elif key == f"counting_line_y_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['counting_line_y'] = float(value)
                            
                        elif key == f"counting_direction_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['counting_direction'] = value
                            
                        elif key == f"center_dot_position_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['center_dot_position'] = float(value)

                except Exception as e:
                    logger.error(f"❌ Config parse error for line: {line} - {e}")
                    
    except Exception as e:
        logger.error(f"❌ Failed to read config file: {e}")
        raise

    camera_urls = [url for url in loaded_urls if url is not None]
   
    if 'port' not in relay_configs:
        relay_configs['port'] = 5000
        
    if not os.path.exists(OUTPUT_VIDEO_DIR):
        os.makedirs(OUTPUT_VIDEO_DIR)
    
    for loader_name, params in loader_params.items():
        logger.info(f"📋 {loader_name} parameters: {params}")
    
    return camera_urls, model_paths


# ==============================================================================
# FLASK API
# ==============================================================================

@flask_app.route('/api/getTargetForAI', methods=['POST'])
def get_target_from_java():
    """Receive target from Java backend"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data"}), 400
       
        loader_name = data.get("loader")
        target_bags_str = data.get("total_bags")
        vehicle_number = data.get("vehicle_number", "NA")
        material_type = data.get("materialType", "NA")
        start_time = data.get("startTime", "NA")
       
        if not loader_name or not target_bags_str:
            return jsonify({"status": "error", "message": "Missing loader or total_bags"}), 400

        try:
            target_bags = int(target_bags_str)
        except ValueError:
            return jsonify({"status": "error", "message": "total_bags must be int"}), 400

        with lock:
            targets[loader_name] = target_bags
            relays_tripped[loader_name] = False
            vehicle_info[loader_name] = {
                "vehicleNumber": vehicle_number,
                "materialType": material_type,
                "startTime": start_time,
                "stopTime": "NA"
            }
       
        video_path = "N/A"
        if loader_name in video_threads:
            video_threads[loader_name].reset_counter()
            video_path = video_threads[loader_name].output_path
            logger.info(f"🎬 New video path for {loader_name}: {video_path}")

        start_belt(loader_name)
        
        logger.info(f"🎯 Target set for {loader_name}: {target_bags} bags")
        return jsonify({"status": "success", "video_path": video_path}), 200
       
    except Exception as e:
        logger.error(f"❌ API Error in getTargetForAI: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ==============================================================================
# THREADED CAMERA CLASS
# ==============================================================================

class ThreadedCamera:
    """Thread-safe camera reader"""
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        self.lock = Lock()
        self.frame = None
        self.status = False
        self.stopped = False
        self.src = src
       
        if self.capture.isOpened():
            self.status, self.frame = self.capture.read()
       
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.lock:
                    self.status = status
                    self.frame = frame
               
                if not status:
                    time.sleep(0.1)
            else:
                time.sleep(1)

    def get_frame(self):
        with self.lock:
            if self.status and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        self.capture.release()


# ==============================================================================
# BAG COUNTER WITH LINE CROSSING
# ==============================================================================

class RTSPBagCounter(Thread):
    """Production bag counter with line-crossing logic"""
    
    def __init__(self, url, loader_name):
        super().__init__()
        self.url = url
        self.loader_name = loader_name
        self.stopped = False
        
        # Get loader-specific parameters
        params = loader_params.get(loader_name, {})
        self.confidence_threshold = params.get('confidence', 0.28)
        self.counting_line_y = params.get('counting_line_y', 0.45)
        self.counting_direction = params.get('counting_direction', 'down')
        self.center_dot_position = params.get('center_dot_position', 0.25)
        
        max_age = params.get('max_age', 5)
        min_hits = params.get('min_hits', 2)
        iou_threshold = params.get('iou_threshold', 0.3)
        
        # Get loader-specific model
        self.model = models.get(loader_name)
        if not self.model:
            logger.error(f"❌ No model loaded for {loader_name}")
            raise ValueError(f"Model not found for {loader_name}")
        
        # Initialize tracker
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )
        
        # Counting data
        self.counted_ids = set()
        self.total_count = 0
        self.tracker_positions = defaultdict(list)
        
        # Statistics
        self.frame_count = 0
        self.start_time = datetime.now()
        self.fps_history = []
        self.last_sent_count = None
        
        # Camera
        self.cam_thread = None
        self.width = 0
        self.height = 0
        
        # Video writer
        self.video_writer = None
        self.output_path = None
        self.is_recording = False
        
        self.window_name = f"Bag Counter - {self.loader_name}"
        
        logger.info(f"✅ Counter initialized: {loader_name} (model={params.get('model_path', 'default')}, line={self.counting_line_y}, dir={self.counting_direction}, dot={self.center_dot_position})")
        # Don't send live count or start recording yet - wait for target
        # self.send_live_count(0)
    
    def init_video_writer(self, frame_width, frame_height):
        """Initialize video writer"""
        try:
            with lock:
                vehicle_number = vehicle_info.get(self.loader_name, {}).get("vehicleNumber", "NA")
            vehicle_number = "".join(c for c in vehicle_number if c.isalnum() or c in ('-', '_'))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(OUTPUT_VIDEO_DIR, f"{self.loader_name}_{vehicle_number}_{timestamp}.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 10.0, (frame_width, frame_height))
            self.is_recording = True
            logger.info(f"📹 Video recording started: {self.output_path}")
        except Exception as e:
            logger.error(f"❌ Video writer error: {e}")
            self.is_recording = False
    
    def release_video_writer(self):
        """Release video writer"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            logger.info(f"⏹️ Video recording stopped: {self.output_path}")
    
    def start_recording(self):
        """Start video recording"""
        if self.cam_thread and not self.is_recording:
            ret, frame = self.cam_thread.get_frame()
            if ret:
                h, w = frame.shape[:2]
                self.init_video_writer(w, h)
    
    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording:
            self.release_video_writer()
    
    def get_line_coordinates(self):
        """Get counting line coordinates"""
        y = int(self.height * self.counting_line_y)
        return (0, y), (self.width, y)
    
    def get_tracking_point(self, x1, y1, x2, y2):
        """Get tracking point based on center_dot_position"""
        center_x = (x1 + x2) // 2
        # Adjust Y based on center_dot_position (0.0=top, 0.5=middle, 1.0=bottom)
        center_y = int(y1 + (y2 - y1) * self.center_dot_position)
        return center_x, center_y
    
    def has_crossed_line(self, track_id, center_x, center_y, line_y):
        """Check if tracker crossed the counting line"""
        current_time = time.time()
        self.tracker_positions[track_id].append((center_x, center_y, current_time))
        
        # Keep only last 10 positions
        if len(self.tracker_positions[track_id]) > 10:
            self.tracker_positions[track_id] = self.tracker_positions[track_id][-10:]
        
        # Need at least 2 positions
        if len(self.tracker_positions[track_id]) < 2:
            return False
        
        prev_y = self.tracker_positions[track_id][-2][1]
        curr_y = self.tracker_positions[track_id][-1][1]
        
        if self.counting_direction == 'down':
            return prev_y < line_y and curr_y >= line_y
        else:  # up
            return prev_y > line_y and curr_y <= line_y
    
    def process_frame(self, frame):
        """Process a single frame with line-crossing logic"""
        frame_start = time.time()
        
        # Get counting line
        line_p1, line_p2 = self.get_line_coordinates()
        line_y = line_p1[1]
        
        # YOLO Detection
        try:
            results = self.model.predict(
                frame,
                imgsz=640,
                verbose=False,
                conf=self.confidence_threshold
            )[0]
        except Exception as e:
            logger.error(f"❌ YOLO error: {e}")
            return frame
        
        # Prepare detections for tracker
        detections = np.empty((0, 5))
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Process tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Get tracking point
            center_x, center_y = self.get_tracking_point(x1, y1, x2, y2)
            
            # Check line crossing
            if track_id not in self.counted_ids:
                if self.has_crossed_line(track_id, center_x, center_y, line_y):
                    self.counted_ids.add(track_id)
                    self.total_count += 1
                    logger.info(f"[{self.loader_name}] Bag #{self.total_count} counted (ID: {track_id})")
            
            # Draw bounding box
            color = (255, 0, 0) if track_id in self.counted_ids else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw tracking point
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw ID
            label = f"ID:{track_id}"
            if track_id in self.counted_ids:
                label += " [COUNTED]"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw counting line
        cv2.line(frame, line_p1, line_p2, (0, 0, 255), 3)
        label_y = line_y - 10 if line_y > 30 else line_y + 25
        cv2.putText(frame, "COUNTING LINE", (line_p1[0] + 10, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Update live count
        with lock:
            live_counts[self.loader_name] = self.total_count
            target = targets.get(self.loader_name, 0)
            is_tripped = relays_tripped.get(self.loader_name, False)
        
        # Send live count update
        if self.last_sent_count is None or self.total_count != self.last_sent_count:
            self.send_live_count(self.total_count)
            self.last_sent_count = self.total_count
        
        # Check if target reached
        if target > 0 and self.total_count >= target and not is_tripped:
            logger.info(f"🎯 Target {target} reached for {self.loader_name}")
            with lock:
                vehicle_info[self.loader_name]["stopTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                relays_tripped[self.loader_name] = True
            self.handle_target_reached(self.total_count)
        
        # Draw statistics (dark red, top-right)
        DARK_RED = (0, 0, 150)
        MARGIN = 10
        
        # Camera name
        camera_text = f"Camera: {self.loader_name}"
        (text_w, text_h), _ = cv2.getTextSize(camera_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x_camera = self.width - text_w - MARGIN
        cv2.putText(frame, camera_text, (x_camera, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_RED, 2)
        
        # Count
        count_text = f"Count: {self.total_count} / {target}"
        (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x_count = self.width - text_w - MARGIN
        cv2.putText(frame, count_text, (x_count, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, DARK_RED, 3)
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 30:
            self.fps_history = self.fps_history[-30:]
        avg_fps = np.mean(self.fps_history)
        
        fps_text = f"FPS: {avg_fps:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x_fps = self.width - text_w - MARGIN
        cv2.putText(frame, fps_text, (x_fps, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DARK_RED, 2)
        
        # Timestamp (bottom-left)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main processing loop"""
        self.cam_thread = ThreadedCamera(self.url)
        time.sleep(1.0)
        
        # Get frame dimensions
        ret, frame = self.cam_thread.get_frame()
        if ret:
            self.height, self.width = frame.shape[:2]
            # Don't start recording yet - wait for target
        else:
            logger.error(f"❌ Failed to get initial frame for {self.loader_name}")
            return
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        logger.info(f"▶️ Starting processing: {self.loader_name} (recording starts when target is set)")
        
        try:
            while not self.stopped:
                ret, frame = self.cam_thread.get_frame()
                
                if not ret:
                    logger.warning(f"⚠️ No frame from {self.loader_name}. Reconnecting...")
                    self.cam_thread.stop()
                    time.sleep(2)
                    self.cam_thread = ThreadedCamera(self.url)
                    time.sleep(1)
                    continue
                
                self.frame_count += 1
                
                # Process frame only if we have a target or are recording
                # Always process for display, but only count when target is set
                with lock:
                    has_target = targets.get(self.loader_name, 0) > 0 or self.is_recording
                
                if has_target or self.frame_count == 1:  # Always process first frame for display
                    processed_frame = self.process_frame(frame)
                else:
                    # Just show live view without processing when no target
                    processed_frame = frame.copy()
                    # Add status text
                    cv2.putText(processed_frame, f"{self.loader_name}: WAITING FOR TARGET", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Save to video ONLY if recording is active
                if self.is_recording and self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # Display
                cv2.imshow(self.window_name, processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info(f"🛑 Quit key pressed for {self.loader_name}")
                    break
                
                # Log statistics
                if self.frame_count % 100 == 0:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    recording_status = "RECORDING" if self.is_recording else "IDLE (Waiting for target)"
                    logger.info(f"[{self.loader_name}] Frames: {self.frame_count}, Count: {self.total_count}, Avg FPS: {avg_fps:.1f}, Status: {recording_status}")
        
        except KeyboardInterrupt:
            logger.info(f"⏹️ Interrupted: {self.loader_name}")
        except Exception as e:
            logger.error(f"❌ Error in {self.loader_name}: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"🧹 Cleaning up: {self.loader_name}")
        
        if self.cam_thread:
            self.cam_thread.stop()
        
        # Stop recording if active
        if self.is_recording:
            self.release_video_writer()
        
        cv2.destroyWindow(self.window_name)
        
        # Final statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"[{self.loader_name}] Final Stats - Frames: {self.frame_count}, Count: {self.total_count}, Duration: {elapsed:.1f}s")
    
    def stop(self):
        """Stop the counter"""
        self.stopped = True
        self.send_live_count(0)
    
    def send_live_count(self, current_count):
        """Send live count to Java backend"""
        if not JAVA_COUNT_UPDATE_API_URL:
            return
            
        with lock:
            info = vehicle_info.get(self.loader_name, {
                "vehicleNumber": "NA",
                "materialType": "NA",
                "startTime": "NA",
                "stopTime": "NA"
            })
            ip = relay_configs.get(self.loader_name, {}).get('ip', "NA")
            target = targets.get(self.loader_name, 0)
            
            payload = {
                "loader": self.loader_name,
                "ip": ip,
                "actualBags": current_count,
                "target": target,
                "vehicleNumber": info["vehicleNumber"],
                "materialType": info["materialType"],
                "status": "counting",
                "startTime": info["startTime"],
                "stopTime": info["stopTime"]
            }
        
        try:
            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=0.5)
        except:
            pass
    
    def handle_target_reached(self, final_count):
        """Handle target reached event"""
        # Stop belt
        stop_belt(self.loader_name)
        
        # Stop recording
        self.stop_recording()
        
        # Send final count
        self.send_final_count(final_count)
        
        # Clear target
        with lock:
            targets[self.loader_name] = 0
    
    def send_final_count(self, final_count):
        """Send final count to Java backend"""
        if not JAVA_COUNT_UPDATE_API_URL:
            return
            
        try:
            with lock:
                info = vehicle_info.get(self.loader_name, {})
                ip = relay_configs.get(self.loader_name, {}).get('ip', "NA")
                payload = {
                    "loader": self.loader_name,
                    "ip": ip,
                    "actualBags": final_count,
                    "target": targets.get(self.loader_name, 0),
                    "vehicleNumber": info.get("vehicleNumber", "NA"),
                    "materialType": info.get("materialType", "NA"),
                    "status": "completed",
                    "startTime": info.get("startTime", "NA"),
                    "stopTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=5)
            logger.info(f"✅ Final count sent for {self.loader_name}: {final_count}")
        except Exception as e:
            logger.error(f"❌ Failed to send final count: {e}")
    
    def reset_counter(self):
        """Reset counter for new vehicle and start recording"""
        # Stop any existing recording first
        if self.is_recording:
            self.stop_recording()
        
        with lock:
            self.counted_ids.clear()
            self.tracker_positions.clear()
            self.total_count = 0
            self.last_sent_count = None
            live_counts[self.loader_name] = 0
            vehicle_info[self.loader_name]["stopTime"] = "NA"
        
        # Start new recording
        self.start_recording()
        
        # Now send initial count
        self.send_live_count(0)
        
        logger.info(f"🔄 Counter reset for {self.loader_name} - Recording started")


# ==============================================================================
# MAIN
# ==============================================================================

def load_models(model_paths):
    """Load YOLO models for each loader"""
    global models
    
    loader_mapping = {
        0: "Loader-BC03",
        1: "Loader-BC02"
    }
    
    for idx, model_path in enumerate(model_paths):
        if model_path is None:
            continue
            
        loader_name = loader_mapping.get(idx)
        if not loader_name:
            continue
        
        logger.info(f"Loading YOLO model for {loader_name}: {model_path}")
        
        try:
            model = YOLO(model_path, task="detect")
            model.fuse()
            
            if torch.cuda.is_available():
                model.to('cuda')
                logger.info(f"✅ {loader_name} model loaded on GPU")
                try:
                    model.half()
                    logger.info(f"✅ {loader_name} model converted to FP16")
                except:
                    logger.warning(f"⚠️ {loader_name} FP16 conversion failed, using FP32")
            else:
                logger.info(f"⚠️ {loader_name} model loaded on CPU")
            
            models[loader_name] = model
            logger.info(f"✅ Model loaded for {loader_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model for {loader_name}: {e}")
            raise


def start_api_server():
    """Start Flask API server"""
    logger.info(f"🌐 Starting API server on port {PYTHON_API_PORT}")
    t = Thread(target=lambda: flask_app.run(host='0.0.0.0', port=PYTHON_API_PORT, debug=False))
    t.daemon = True
    t.start()


def start_tracking(camera_urls):
    """Start tracking threads for all cameras"""
    # Stop existing threads
    for _, worker in video_threads.items():
        worker.stop()
    video_threads.clear()
    
    # Map cameras to loaders
    loader_mapping = {
        0: "Loader-BC03",  # Camera 1
        1: "Loader-BC02"   # Camera 2
    }
    
    for idx in range(min(2, len(camera_urls))):
        loader = loader_mapping.get(idx)
        if not loader:
            continue
            
        worker = RTSPBagCounter(camera_urls[idx], loader)
        worker.daemon = True
        worker.start()
        video_threads[loader] = worker
        
        logger.info(f"✅ Started worker for {loader}")
    
    logger.info(f"✅ All {len(video_threads)} camera workers started")


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("RTSP Production Bag Counting System")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        camera_urls, model_paths = load_config_properties()
        
        if not camera_urls:
            logger.error("❌ No cameras configured. Exiting.")
            return
        
        logger.info(f"✅ Loaded {len(camera_urls)} camera(s)")
        
        # Load YOLO models (separate for each loader)
        load_models(model_paths)
        
        # Start API server
        start_api_server()
        
        # Start tracking
        start_tracking(camera_urls)
        
        logger.info("✅ System running. Press Ctrl+C to exit.")
        logger.info("=" * 60)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        
        # Stop all workers
        for worker in video_threads.values():
            worker.stop()
        
        logger.info("=" * 60)
        logger.info("System stopped")
        logger.info("=" * 60)
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
