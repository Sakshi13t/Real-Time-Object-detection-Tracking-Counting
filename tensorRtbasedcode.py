"""
RTSP Camera Bag Counting System - TensorRT Optimized for Jetson Orin Nano
Enhanced with Multi-Line Fallback + Performance Optimizations
FIXED VERSION - NO DOUBLE COUNTING + SMOOTH REALTIME VIDEO
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
from threading import Thread, Lock, Timer, Event
from collections import defaultdict, deque
import requests
from flask import Flask, request, jsonify
import signal
import traceback

# TensorRT support
try:
    from ultralytics import YOLO
    import torch
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT/YOLO not available")

from sort import Sort

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
delayed_stop_timers = {}

PYTHON_API_PORT = 8888
JAVA_COUNT_UPDATE_API_URL = ""
OUTPUT_VIDEO_DIR = "/media/nvidia/My Passport/output_videos"
RECORDING_DELAY_SECONDS = 10

# Performance optimization settings
JETSON_OPTIMIZATION = {
    'frame_skip': 1,  # Process every Nth frame (1 = no skip, 2 = skip 1, etc.)
    'display_downsample': 1.0,  # Downsample display (1.0 = full res, 0.5 = half)
    'inference_size': 416,  # Smaller input size for faster inference (416 or 640)
    'max_fps': 15,  # Target FPS to reduce CPU load
    'use_tensorrt': True,  # Use TensorRT engine
    'async_display': True,  # Async display updates
}

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
   
    for attempt in range(3):  # Reduced retries for speed
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)  # Reduced timeout
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
            if attempt == 2:  # Only log last attempt
                logger.warning(f"⚠️ Belt start failed: {e}")
            time.sleep(0.5)
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
   
    for attempt in range(3):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
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
            if attempt == 2:
                logger.warning(f"⚠️ Belt stop failed: {e}")
            time.sleep(0.5)
    return False


def load_config_properties(file_path="config.properties"):
    """Load configuration from properties file"""
    global PYTHON_API_PORT, JAVA_COUNT_UPDATE_API_URL, loader_params, relay_configs
    
    loaded_urls = [None, None]
    loader_names = [None, None]
    model_paths = [None, None]
   
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
                        
                        # Enhanced parameters
                        elif key == f"fallback_line_y_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['fallback_line_y'] = float(value)
                            
                        elif key == f"early_detection_line_y_{l_idx}" and loader_names[idx_num]:
                            loader_params[loader_names[idx_num]]['early_detection_line_y'] = float(value)

                except Exception as e:
                    logger.error(f"❌ Config parse error for line: {line} - {e}")
                    
    except Exception as e:
        logger.error(f"❌ Failed to read config file: {e}")
        raise

    camera_urls = [url for url in loaded_urls if url is not None]
   
    if 'port' not in relay_configs:
        relay_configs['port'] = 5000
    
    # Set default values for enhanced parameters if not specified
    for loader_name, params in loader_params.items():
        if 'fallback_line_y' not in params:
            primary_y = params.get('counting_line_y', 0.45)
            params['fallback_line_y'] = primary_y + 0.10
        
        if 'early_detection_line_y' not in params:
            primary_y = params.get('counting_line_y', 0.45)
            params['early_detection_line_y'] = primary_y - 0.10
        
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

        if loader_name in delayed_stop_timers:
            delayed_stop_timers[loader_name].cancel()
            del delayed_stop_timers[loader_name]

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
# OPTIMIZED THREADED CAMERA CLASS
# ==============================================================================

class OptimizedThreadedCamera:
    """Optimized thread-safe camera reader for Jetson"""
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        
        # Optimized settings for Jetson
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        
        # Try to set lower resolution if possible (reduces bandwidth)
        # Uncomment if you want to force lower resolution from camera
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.lock = Lock()
        self.frame = None
        self.status = False
        self.stopped = False
        self.src = src
        self.frame_count = 0
       
        if self.capture.isOpened():
            self.status, self.frame = self.capture.read()
       
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Optimized frame reading with skip logic"""
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                
                if status:
                    with self.lock:
                        self.status = status
                        self.frame = frame
                        self.frame_count += 1
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.5)

    def get_frame(self):
        """Get latest frame"""
        with self.lock:
            if self.status and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        """Stop camera thread"""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.capture.release()


# ==============================================================================
# ENHANCED BAG COUNTER (Same as before - no changes needed)
# ==============================================================================

class EnhancedBagCounter:
    """Multi-line fallback counting with trajectory prediction"""
    
    def __init__(self, loader_name, params):
        self.loader_name = loader_name
        
        self.primary_line_y = params.get('counting_line_y', 0.45)
        self.fallback_line_y = params.get('fallback_line_y', 0.55)
        self.early_detection_line_y = params.get('early_detection_line_y', 0.35)
        
        self.counting_direction = params.get('counting_direction', 'down')
        self.center_dot_position = params.get('center_dot_position', 0.5)
        
        self.counted_ids = set()
        self.fallback_counted_ids = set()
        self.early_detected_ids = set()
        self.between_lines_ids = set()
        
        self.tracker_positions = defaultdict(lambda: deque(maxlen=20))
        self.tracker_velocities = defaultdict(list)
        self.tracker_last_seen = defaultdict(float)
        self.tracker_missed_frames = defaultdict(int)
        self.tracker_crossed_primary = defaultdict(lambda: False)
        
        self.primary_count = 0
        self.fallback_count = 0
        self.predicted_count = 0
        self.total_count = 0
        self.double_count_warnings = 0
        
        self.width = 0
        self.height = 0
        
        logger.info(f"✅ Enhanced counter initialized: {loader_name}")
    
    def set_frame_dimensions(self, width, height):
        self.width = width
        self.height = height
    
    def get_tracking_point(self, x1, y1, x2, y2):
        center_x = (x1 + x2) // 2
        center_y = int(y1 + (y2 - y1) * self.center_dot_position)
        return center_x, center_y
    
    def calculate_velocity(self, track_id):
        positions = self.tracker_positions[track_id]
        if len(positions) < 2:
            return 0, 0
        
        velocities_x = []
        velocities_y = []
        
        for i in range(len(positions) - 1):
            x1, y1, t1 = positions[i]
            x2, y2, t2 = positions[i + 1]
            dt = t2 - t1
            if dt > 0:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                velocities_x.append(vx)
                velocities_y.append(vy)
        
        if velocities_x and velocities_y:
            return np.median(velocities_x), np.median(velocities_y)
        return 0, 0
    
    def predict_trajectory(self, track_id, time_ahead=0.5):
        positions = self.tracker_positions[track_id]
        if len(positions) < 2:
            return None
        
        last_x, last_y, last_t = positions[-1]
        vx, vy = self.calculate_velocity(track_id)
        
        pred_x = last_x + vx * time_ahead
        pred_y = last_y + vy * time_ahead
        
        return int(pred_x), int(pred_y)
    
    def is_in_fallback_zone(self, center_y):
        primary_y = int(self.height * self.primary_line_y)
        fallback_y = int(self.height * self.fallback_line_y)
        
        if self.counting_direction == 'down':
            return primary_y < center_y < fallback_y
        else:
            return fallback_y < center_y < primary_y
    
    def check_fallback_eligibility(self, track_id):
        if track_id in self.counted_ids:
            return False
        
        if len(self.tracker_positions[track_id]) < 2:
            return False
        
        if track_id not in self.between_lines_ids:
            return False
        
        positions = list(self.tracker_positions[track_id])
        if len(positions) < 2:
            return False
        
        recent_positions = positions[-min(5, len(positions)):]
        direction_consistent = True
        
        for i in range(len(recent_positions) - 1):
            y1 = recent_positions[i][1]
            y2 = recent_positions[i + 1][1]
            
            if self.counting_direction == 'down':
                if y2 < y1:
                    direction_consistent = False
                    break
            else:
                if y2 > y1:
                    direction_consistent = False
                    break
        
        return direction_consistent
    
    def handle_missing_trackers(self, current_frame_time):
        MISSING_THRESHOLD = 0.3
        
        primary_y = int(self.height * self.primary_line_y)
        fallback_y = int(self.height * self.fallback_line_y)
        
        for track_id, last_seen in list(self.tracker_last_seen.items()):
            time_since_seen = current_frame_time - last_seen
            
            if time_since_seen < MISSING_THRESHOLD:
                continue
            
            if track_id in self.counted_ids or track_id in self.fallback_counted_ids:
                continue
            
            if len(self.tracker_positions[track_id]) < 3:
                continue
            
            last_x, last_y, _ = self.tracker_positions[track_id][-1]
            
            was_in_zone = False
            if self.counting_direction == 'down':
                was_in_zone = primary_y < last_y < fallback_y
            else:
                was_in_zone = fallback_y < last_y < primary_y
            
            if not was_in_zone:
                continue
            
            predicted_pos = self.predict_trajectory(track_id, time_since_seen)
            
            if predicted_pos:
                pred_x, pred_y = predicted_pos
                
                should_count = False
                if self.counting_direction == 'down':
                    should_count = last_y < fallback_y and pred_y >= fallback_y
                else:
                    should_count = last_y > fallback_y and pred_y <= fallback_y
                
                if should_count and track_id in self.between_lines_ids:
                    if track_id not in self.counted_ids and track_id not in self.fallback_counted_ids:
                        self.fallback_counted_ids.add(track_id)
                        self.predicted_count += 1
                        logger.warning(f"[{self.loader_name}] ⚠️ PREDICTED count for missing ID {track_id}")
    
    def process_tracked_objects(self, tracked_objects, current_frame_time):
        """Process tracked objects - FIXED multi-line logic"""
        
        primary_y = int(self.height * self.primary_line_y)
        fallback_y = int(self.height * self.fallback_line_y)
        
        visible_ids = set()
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            visible_ids.add(track_id)
            
            center_x, center_y = self.get_tracking_point(x1, y1, x2, y2)
            
            already_counted = (track_id in self.counted_ids or 
                              track_id in self.fallback_counted_ids)
            
            self.tracker_positions[track_id].append((center_x, center_y, current_frame_time))
            self.tracker_last_seen[track_id] = current_frame_time
            self.tracker_missed_frames[track_id] = 0
            
            if self.counting_direction == 'down':
                if center_y < primary_y and track_id not in self.early_detected_ids:
                    self.early_detected_ids.add(track_id)
            else:
                if center_y > primary_y and track_id not in self.early_detected_ids:
                    self.early_detected_ids.add(track_id)
            
            if not already_counted and self.is_in_fallback_zone(center_y):
                if track_id not in self.between_lines_ids:
                    self.between_lines_ids.add(track_id)
            
            if already_counted:
                continue
            
            if len(self.tracker_positions[track_id]) >= 2:
                prev_y = self.tracker_positions[track_id][-2][1]
                curr_y = center_y
                
                primary_crossed = False
                if self.counting_direction == 'down':
                    primary_crossed = prev_y < primary_y and curr_y >= primary_y
                else:
                    primary_crossed = prev_y > primary_y and curr_y <= primary_y
                
                if primary_crossed:
                    if track_id in self.counted_ids:
                        logger.error(f"[{self.loader_name}] ❌ ID {track_id} already counted!")
                        self.double_count_warnings += 1
                        continue
                    
                    self.counted_ids.add(track_id)
                    self.primary_count += 1
                    
                    if track_id in self.between_lines_ids:
                        self.between_lines_ids.discard(track_id)
                    
                    continue
                
                if self.check_fallback_eligibility(track_id):
                    fallback_crossed = False
                    if self.counting_direction == 'down':
                        fallback_crossed = prev_y < fallback_y and curr_y >= fallback_y
                    else:
                        fallback_crossed = prev_y > fallback_y and curr_y <= fallback_y
                    
                    if fallback_crossed:
                        if track_id in self.counted_ids:
                            logger.error(f"[{self.loader_name}] ❌ PREVENTED double count for ID {track_id}")
                            self.double_count_warnings += 1
                            continue
                        
                        if track_id in self.fallback_counted_ids:
                            logger.error(f"[{self.loader_name}] ❌ PREVENTED duplicate fallback!")
                            self.double_count_warnings += 1
                            continue
                        
                        self.fallback_counted_ids.add(track_id)
                        self.fallback_count += 1
        
        self.handle_missing_trackers(current_frame_time)
        self.total_count = self.primary_count + self.fallback_count + self.predicted_count
        
        for track_id in self.tracker_last_seen:
            if track_id not in visible_ids:
                self.tracker_missed_frames[track_id] += 1
        
        double_counted = self.counted_ids.intersection(self.fallback_counted_ids)
        if double_counted:
            logger.error(f"[{self.loader_name}] ❌❌❌ DOUBLE COUNT DETECTED! IDs: {double_counted}")
            for dup_id in double_counted:
                self.fallback_counted_ids.discard(dup_id)
                self.fallback_count -= 1
                self.double_count_warnings += 1
            self.total_count = self.primary_count + self.fallback_count + self.predicted_count
    
    def draw_visualization(self, frame, tracked_objects, hud=None, lightweight=False):
        """Optimized visualization with lightweight mode"""
        hud = hud or {}

        primary_y = int(self.height * self.primary_line_y)
        fallback_y = int(self.height * self.fallback_line_y)

        # Draw primary line (simplified in lightweight mode)
        cv2.line(frame, (0, primary_y), (self.width, primary_y), (0, 0, 255), 2)
        cv2.line(frame, (0, fallback_y), (self.width, fallback_y), (0, 165, 255), 2)

        # Draw tracked objects (simplified bounding boxes)
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)

            if track_id in self.counted_ids:
                color = (255, 0, 0)
            elif track_id in self.fallback_counted_ids:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Skip trajectories and labels in lightweight mode for speed
            if not lightweight:
                center_x, center_y = self.get_tracking_point(x1, y1, x2, y2)
                cv2.circle(frame, (center_x, center_y), 4, color, -1)

        # Compact HUD
        stats_x = self.width - 280
        stats_y = 30
        line_height = 25

        loader = hud.get("loader", self.loader_name)
        target = hud.get("target", 0)
        fps = hud.get("fps", 0.0)

        lines = [
            (f"{loader}", (255, 255, 255), 0.6, 2),
            (f"Count: {self.total_count}/{target}", (0, 255, 0), 0.8, 2),
            (f"FPS: {fps:.1f}", (200, 200, 200), 0.55, 1),
        ]

        bg_pad = 10
        bg_h = line_height * len(lines) + 2 * bg_pad
        bg_top = stats_y - 20
        bg_left = stats_x - 10
        bg_right = self.width - 5
        bg_bottom = bg_top + bg_h

        cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1)
        cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), (255, 255, 255), 1)

        y = stats_y
        for text, color, scale, thick in lines:
            cv2.putText(frame, text, (stats_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            y += line_height

        return frame
    
    def reset(self):
        """Reset counter"""
        self.counted_ids.clear()
        self.fallback_counted_ids.clear()
        self.early_detected_ids.clear()
        self.between_lines_ids.clear()
        self.tracker_positions.clear()
        self.tracker_velocities.clear()
        self.tracker_last_seen.clear()
        self.tracker_missed_frames.clear()
        self.tracker_crossed_primary.clear()
        
        self.primary_count = 0
        self.fallback_count = 0
        self.predicted_count = 0
        self.total_count = 0
        self.double_count_warnings = 0
    
    def get_statistics(self):
        primary_accuracy = (self.primary_count / self.total_count * 100) if self.total_count > 0 else 0
        fallback_rate = (self.fallback_count / self.total_count * 100) if self.total_count > 0 else 0
        
        return {
            "loader": self.loader_name,
            "primary_count": self.primary_count,
            "fallback_count": self.fallback_count,
            "predicted_count": self.predicted_count,
            "total_count": self.total_count,
            "primary_accuracy": round(primary_accuracy, 2),
            "fallback_rate": round(fallback_rate, 2),
            "double_count_warnings": self.double_count_warnings
        }


# ==============================================================================
# OPTIMIZED BAG COUNTER FOR JETSON ORIN NANO
# ==============================================================================

class OptimizedRTSPBagCounter(Thread):
    """TensorRT-optimized bag counter with frame skipping and async display"""
    
    def __init__(self, url, loader_name):
        super().__init__()
        self.url = url
        self.loader_name = loader_name
        self.stopped = False
        
        params = loader_params.get(loader_name, {})
        self.confidence_threshold = params.get('confidence', 0.28)
        
        max_age = params.get('max_age', 5)
        min_hits = params.get('min_hits', 2)
        iou_threshold = params.get('iou_threshold', 0.3)
        
        self.model = models.get(loader_name)
        if not self.model:
            logger.error(f"❌ No model loaded for {loader_name}")
            raise ValueError(f"Model not found for {loader_name}")
        
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )
        
        self.counter = EnhancedBagCounter(loader_name, params)
        
        # Performance tracking
        self.frame_count = 0
        self.processed_frame_count = 0
        self.start_time = datetime.now()
        self.fps_history = deque(maxlen=30)
        self.last_sent_count = None
        
        # Frame skipping
        self.frame_skip = JETSON_OPTIMIZATION['frame_skip']
        self.inference_size = JETSON_OPTIMIZATION['inference_size']
        self.max_fps = JETSON_OPTIMIZATION['max_fps']
        self.min_frame_time = 1.0 / self.max_fps
        
        # Camera
        self.cam_thread = None
        self.width = 0
        self.height = 0
        
        # Video writer
        self.video_writer = None
        self.output_path = None
        self.is_recording = False
        self.target_reached_time = None
        
        # Display optimization
        self.display_frame = None
        self.display_lock = Lock()
        self.display_updated = Event()
        
        self.window_name = f"Optimized Counter - {self.loader_name}"
        
        logger.info(f"✅ Optimized counter initialized: {loader_name}")
        logger.info(f"   Frame skip: {self.frame_skip}, Inference size: {self.inference_size}")
    
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
            self.target_reached_time = None
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
            self.target_reached_time = None
            logger.info(f"⏹️ Video recording stopped")
    
    def schedule_delayed_stop(self):
        """Schedule delayed stop"""
        def delayed_stop_callback():
            logger.info(f"⏱️ {RECORDING_DELAY_SECONDS}s delay - stopping recording for {self.loader_name}")
            self.release_video_writer()
            if self.loader_name in delayed_stop_timers:
                del delayed_stop_timers[self.loader_name]
        
        if self.loader_name in delayed_stop_timers:
            delayed_stop_timers[self.loader_name].cancel()
        
        timer = Timer(RECORDING_DELAY_SECONDS, delayed_stop_callback)
        timer.daemon = True
        timer.start()
        delayed_stop_timers[self.loader_name] = timer
    
    def process_frame(self, frame, lightweight=False):
        """Optimized frame processing with TensorRT"""
        frame_start = time.time()
        current_frame_time = time.time()

        # YOLO Detection with TensorRT
        try:
            results = self.model.predict(
                frame,
                imgsz=self.inference_size,
                verbose=False,
                conf=self.confidence_threshold,
                device=0,  # GPU
                half=True  # FP16 for speed
            )[0]
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
            return frame

        # Prepare detections
        detections = np.empty((0, 5))
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # Update tracker
        tracked_objects = self.tracker.update(detections)

        # Process with counter
        self.counter.process_tracked_objects(tracked_objects, current_frame_time)

        # Update counts
        with lock:
            live_counts[self.loader_name] = self.counter.total_count
            target = targets.get(self.loader_name, 0)
            is_tripped = relays_tripped.get(self.loader_name, False)

        # Send live count
        if self.last_sent_count is None or self.counter.total_count != self.last_sent_count:
            self.send_live_count(self.counter.total_count)
            self.last_sent_count = self.counter.total_count

        # Check target
        if target > 0 and self.counter.total_count >= target and not is_tripped:
            with lock:
                vehicle_info[self.loader_name]["stopTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                relays_tripped[self.loader_name] = True
            
            if self.target_reached_time is None:
                self.target_reached_time = time.time()
            
            self.handle_target_reached(self.counter.total_count)

        # Stop recording after delay
        if self.is_recording and self.target_reached_time is not None:
            if time.time() - self.target_reached_time >= RECORDING_DELAY_SECONDS:
                self.release_video_writer()

        # FPS calculation
        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        self.fps_history.append(current_fps)
        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0

        # Draw visualization (lightweight for display)
        hud = {
            "loader": self.loader_name,
            "target": target,
            "fps": avg_fps,
        }
        frame = self.counter.draw_visualization(frame, tracked_objects, hud=hud, lightweight=lightweight)

        return frame
    
    def display_thread_func(self):
        """Separate thread for display updates"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)  # Smaller window for performance
        
        while not self.stopped:
            self.display_updated.wait(timeout=0.1)
            
            with self.display_lock:
                if self.display_frame is not None:
                    display_frame = self.display_frame.copy()
                else:
                    continue
            
            # Downsample for display if needed
            downsample = JETSON_OPTIMIZATION['display_downsample']
            if downsample < 1.0:
                h, w = display_frame.shape[:2]
                display_frame = cv2.resize(display_frame, 
                                          (int(w * downsample), int(h * downsample)))
            
            cv2.imshow(self.window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break
            
            self.display_updated.clear()
    
    def run(self):
        """Main processing loop - OPTIMIZED"""
        self.cam_thread = OptimizedThreadedCamera(self.url)
        time.sleep(1.0)
        
        # Get frame dimensions
        ret, frame = self.cam_thread.get_frame()
        if ret:
            self.height, self.width = frame.shape[:2]
            self.counter.set_frame_dimensions(self.width, self.height)
        else:
            logger.error(f"❌ Failed to get initial frame for {self.loader_name}")
            return
        
        # Start async display thread
        if JETSON_OPTIMIZATION['async_display']:
            display_thread = Thread(target=self.display_thread_func)
            display_thread.daemon = True
            display_thread.start()
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
        
        logger.info(f"▶️ Starting optimized processing: {self.loader_name}")
        
        last_process_time = 0
        
        try:
            while not self.stopped:
                ret, frame = self.cam_thread.get_frame()
                
                if not ret:
                    logger.warning(f"⚠️ No frame from {self.loader_name}")
                    time.sleep(0.5)
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # Frame rate limiting
                time_since_last = current_time - last_process_time
                if time_since_last < self.min_frame_time:
                    continue
                
                # Frame skipping logic
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                with lock:
                    has_target = targets.get(self.loader_name, 0) > 0 or self.is_recording
                
                if has_target or self.processed_frame_count == 0:
                    # Process frame (lightweight visualization for display)
                    processed_frame = self.process_frame(frame, lightweight=True)
                    self.processed_frame_count += 1
                    last_process_time = current_time
                else:
                    processed_frame = frame.copy()
                    cv2.putText(processed_frame, f"{self.loader_name}: WAITING", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Save to video (full quality)
                if self.is_recording and self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # Update display (async or sync)
                if JETSON_OPTIMIZATION['async_display']:
                    with self.display_lock:
                        self.display_frame = processed_frame
                    self.display_updated.set()
                else:
                    downsample = JETSON_OPTIMIZATION['display_downsample']
                    if downsample < 1.0:
                        h, w = processed_frame.shape[:2]
                        display_frame = cv2.resize(processed_frame,
                                                   (int(w * downsample), int(h * downsample)))
                    else:
                        display_frame = processed_frame
                    
                    cv2.imshow(self.window_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Periodic stats
                if self.processed_frame_count % 50 == 0:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    avg_fps = self.processed_frame_count / elapsed if elapsed > 0 else 0
                    stats = self.counter.get_statistics()
                    logger.info(f"[{self.loader_name}] Processed: {self.processed_frame_count}, "
                              f"Total: {stats['total_count']}, FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            logger.info(f"⏹️ Interrupted: {self.loader_name}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"🧹 Cleaning up: {self.loader_name}")
        
        if self.loader_name in delayed_stop_timers:
            delayed_stop_timers[self.loader_name].cancel()
            del delayed_stop_timers[self.loader_name]
        
        if self.cam_thread:
            self.cam_thread.stop()
        
        if self.is_recording:
            self.release_video_writer()
        
        cv2.destroyWindow(self.window_name)
        
        stats = self.counter.get_statistics()
        logger.info(f"[{self.loader_name}] Final: {stats['total_count']} "
                   f"(P:{stats['primary_count']} F:{stats['fallback_count']})")
    
    def stop(self):
        self.stopped = True
        self.send_live_count(0)
    
    def send_live_count(self, current_count):
        """Send live count to backend"""
        if not JAVA_COUNT_UPDATE_API_URL:
            return
            
        try:
            with lock:
                info = vehicle_info.get(self.loader_name, {})
                ip = relay_configs.get(self.loader_name, {}).get('ip', "NA")
                target = targets.get(self.loader_name, 0)
                stats = self.counter.get_statistics()
                
                payload = {
                    "loader": self.loader_name,
                    "ip": ip,
                    "actualBags": current_count,
                    "target": target,
                    "vehicleNumber": info.get("vehicleNumber", "NA"),
                    "status": "counting"
                }
            
            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=0.3)
        except:
            pass
    
    def handle_target_reached(self, final_count):
        """Handle target reached"""
        stop_belt(self.loader_name)
        
        if self.is_recording:
            self.schedule_delayed_stop()
        
        self.send_final_count(final_count)
        
        with lock:
            targets[self.loader_name] = 0
    
    def send_final_count(self, final_count):
        """Send final count"""
        if not JAVA_COUNT_UPDATE_API_URL:
            return
            
        try:
            with lock:
                info = vehicle_info.get(self.loader_name, {})
                stats = self.counter.get_statistics()
                
                payload = {
                    "loader": self.loader_name,
                    "actualBags": final_count,
                    "target": targets.get(self.loader_name, 0),
                    "vehicleNumber": info.get("vehicleNumber", "NA"),
                    "status": "completed",
                    "stopTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "primaryCount": stats["primary_count"],
                    "fallbackCount": stats["fallback_count"]
                }
            
            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=5)
            logger.info(f"✅ Final count sent: {final_count}")
        except Exception as e:
            logger.error(f"❌ Failed to send final count: {e}")
    
    def reset_counter(self):
        """Reset counter for new vehicle"""
        if self.loader_name in delayed_stop_timers:
            delayed_stop_timers[self.loader_name].cancel()
            del delayed_stop_timers[self.loader_name]
        
        if self.is_recording:
            self.release_video_writer()
        
        self.counter.reset()
        self.target_reached_time = None
        
        with lock:
            live_counts[self.loader_name] = 0
            vehicle_info[self.loader_name]["stopTime"] = "NA"
        
        # Start recording
        ret, frame = self.cam_thread.get_frame()
        if ret:
            h, w = frame.shape[:2]
            self.init_video_writer(w, h)
        
        self.send_live_count(0)


# ==============================================================================
# MODEL LOADING WITH TENSORRT SUPPORT
# ==============================================================================

def load_models_tensorrt(model_paths):
    """Load YOLO models with TensorRT engine support"""
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
        
        logger.info(f"Loading model for {loader_name}: {model_path}")
        
        try:
            # Check if TensorRT engine file
            if model_path.endswith('.engine'):
                logger.info(f"✅ Loading TensorRT engine: {model_path}")
                model = YOLO(model_path, task="detect")
                logger.info(f"✅ TensorRT engine loaded for {loader_name}")
            else:
                # Load regular .pt model
                logger.info(f"Loading PyTorch model: {model_path}")
                model = YOLO(model_path, task="detect")
                
                if torch.cuda.is_available():
                    model.to('cuda')
                    model.fuse()
                    try:
                        model.half()  # FP16 optimization
                        logger.info(f"✅ Model loaded with FP16 on GPU")
                    except:
                        logger.warning(f"⚠️ FP16 failed, using FP32")
                else:
                    logger.warning(f"⚠️ CUDA not available, using CPU")
            
            models[loader_name] = model
            logger.info(f"✅ Model ready for {loader_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model for {loader_name}: {e}")
            raise


def start_api_server():
    """Start Flask API server"""
    logger.info(f"🌐 Starting API server on port {PYTHON_API_PORT}")
    t = Thread(target=lambda: flask_app.run(host='0.0.0.0', port=PYTHON_API_PORT, debug=False))
    t.daemon = True
    t.start()


def start_tracking_optimized(camera_urls):
    """Start optimized tracking threads"""
    for _, worker in video_threads.items():
        worker.stop()
    video_threads.clear()
    
    loader_mapping = {
        0: "Loader-BC03",
        1: "Loader-BC02"
    }
    
    for idx in range(min(2, len(camera_urls))):
        loader = loader_mapping.get(idx)
        if not loader:
            continue
            
        worker = OptimizedRTSPBagCounter(camera_urls[idx], loader)
        worker.daemon = True
        worker.start()
        video_threads[loader] = worker
        
        logger.info(f"✅ Started optimized worker for {loader}")
    
    logger.info(f"✅ All {len(video_threads)} optimized workers started")


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("TensorRT Optimized Bag Counting System")
    logger.info("For Jetson Orin Nano - NO DOUBLE COUNTING")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        camera_urls, model_paths = load_config_properties()
        
        if not camera_urls:
            logger.error("❌ No cameras configured")
            return
        
        logger.info(f"✅ Loaded {len(camera_urls)} camera(s)")
        
        # Load models with TensorRT support
        load_models_tensorrt(model_paths)
        
        # Start API server
        start_api_server()
        
        # Start optimized tracking
        start_tracking_optimized(camera_urls)
        
        logger.info("✅ Optimized system running")
        logger.info(f"   Frame skip: {JETSON_OPTIMIZATION['frame_skip']}")
        logger.info(f"   Inference size: {JETSON_OPTIMIZATION['inference_size']}")
        logger.info(f"   Target FPS: {JETSON_OPTIMIZATION['max_fps']}")
        logger.info("=" * 60)
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        
        for timer in delayed_stop_timers.values():
            timer.cancel()
        delayed_stop_timers.clear()
        
        for worker in video_threads.values():
            worker.stop()
        
        logger.info("System stopped")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()