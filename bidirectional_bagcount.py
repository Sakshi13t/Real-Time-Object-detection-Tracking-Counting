"""
RTSP Camera Bag Counting System - Single Camera with Bidirectional Counting and ROI
Real-time bag counting with Java API integration, relay control, and Region of Interest
Supports count increment (bottom-to-top) and decrement (top-to-bottom) for ANY bag crossing in reverse
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
    logger.error("Segmentation fault detected!")
    sys.exit(1)

signal.signal(signal.SIGSEGV, handle_segfault)

# === GLOBALS ===
model = None
lock = Lock()
target = 0
relay_tripped = False
relay_config = {}
live_count = 0
vehicle_info = {}
last_command_time = 0
video_thread = None
loader_params = {}
roi_polygon = None

PYTHON_API_PORT = 8888
JAVA_COUNT_UPDATE_API_URL = ""
OUTPUT_VIDEO_DIR = "output_videos"
LOADER_NAME = "Loader-BC03"
ROI_CONFIG_FILE = "roi_config.json"
USE_ROI = True  # Set to False to disable ROI

# === FLASK SERVER SETUP ===
flask_app = Flask(__name__)

# ==============================================================================
# ROI FUNCTIONS
# ==============================================================================

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def box_in_roi(box, roi_polygon):
    """Check if bounding box is inside ROI (checks center point)"""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return point_in_polygon((center_x, center_y), roi_polygon)


def save_roi_config(points, filepath=ROI_CONFIG_FILE):
    """Save ROI configuration to file"""
    try:
        config = {
            'roi_points': points,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"ROI configuration saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save ROI config: {e}")
        return False

def load_roi_config(filepath=ROI_CONFIG_FILE):
    """Load ROI configuration from file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)
            points = [tuple(p) for p in config.get('roi_points', [])]
            if points:
                logger.info(f"ROI configuration loaded from {filepath}")
                logger.info(f"   {len(points)} points, saved on {config.get('timestamp', 'unknown')}")
                return points
    except Exception as e:
        logger.error(f"Failed to load ROI config: {e}")
    return None

class ROISelector:
    """Interactive ROI selector"""
    
    def __init__(self, frame, window_name="ROI Selection"):
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        self.window_name = window_name
        self.points = []
        self.current_point = None
        self.done = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point on left click
            self.points.append((x, y))
            logger.info(f"ROI point added: ({x}, {y})")
            self.draw_roi()
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # Show preview of next point
            self.current_point = (x, y)
            self.draw_roi()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point on right click
            if self.points:
                removed = self.points.pop()
                logger.info(f"ROI point removed: {removed}")
                self.draw_roi()
    
    def draw_roi(self):
        """Draw ROI on frame"""
        self.frame = self.original_frame.copy()
        
        # Draw existing points
        for i, point in enumerate(self.points):
            cv2.circle(self.frame, point, 5, (0, 255, 0), -1)
            cv2.putText(self.frame, str(i+1), (point[0]+10, point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(self.frame, self.points[i], self.points[i+1], (0, 255, 0), 2)
        
        # Draw preview line to current mouse position
        if len(self.points) > 0 and self.current_point:
            cv2.line(self.frame, self.points[-1], self.current_point, (255, 255, 0), 1)
        
        # Draw closing line if we have at least 3 points
        if len(self.points) >= 3 and self.current_point:
            cv2.line(self.frame, self.points[-1], self.points[0], (255, 0, 0), 1)
        
        # Draw instructions
        instructions = [
            "LEFT CLICK: Add point",
            "RIGHT CLICK: Remove last point",
            "ENTER: Finish selection",
            "ESC: Cancel (use full frame)",
            f"Points: {len(self.points)}"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(self.frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        cv2.imshow(self.window_name, self.frame)
    
    def select(self):
        """Start ROI selection process"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.draw_roi()
        
        logger.info("=" * 80)
        logger.info("ROI SELECTION MODE")
        logger.info("=" * 80)
        logger.info("Instructions:")
        logger.info("  - LEFT CLICK to add points around detection area")
        logger.info("  - RIGHT CLICK to remove last point")
        logger.info("  - Press ENTER when done (minimum 3 points)")
        logger.info("  - Press ESC to cancel and use full frame")
        logger.info("=" * 80)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter
                if len(self.points) >= 3:
                    self.done = True
                    logger.info(f"ROI selected with {len(self.points)} points")
                    break
                else:
                    logger.warning("Need at least 3 points to create ROI")
            
            elif key == 27:  # ESC
                self.points = []
                logger.info("ROI selection cancelled - using full frame")
                break
        
        cv2.destroyWindow(self.window_name)
        return self.points


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def start_belt():
    """Start conveyor belt via relay"""
    global last_command_time
    COOLDOWN_SECONDS = 2
    
    current_time = time.time()
    if current_time - last_command_time < COOLDOWN_SECONDS:
        return False

    ip = relay_config.get('ip')
    port = relay_config.get('port', 5000)
    command = relay_config.get('relay_start_cmd', '*R4#0#$#')

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
                last_command_time = time.time()
                logger.info(f"Belt started")
                return True
        except Exception as e:
            logger.warning(f"Belt start attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return False


def stop_belt():
    """Stop conveyor belt via relay"""
    global last_command_time
    COOLDOWN_SECONDS = 1
    
    current_time = time.time()
    if current_time - last_command_time < COOLDOWN_SECONDS:
        return False

    ip = relay_config.get('ip')
    port = relay_config.get('port', 5000)
    command = relay_config.get('relay_stop_cmd', '*R4#1#$#')

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
                last_command_time = time.time()
                logger.info(f"Belt stopped")
                return True
        except Exception as e:
            logger.warning(f"Belt stop attempt {attempt+1} failed: {e}")
            time.sleep(1)
    return False


def load_config_properties(file_path="config.properties"):
    """Load configuration from properties file"""
    global PYTHON_API_PORT, JAVA_COUNT_UPDATE_API_URL, loader_params, relay_config, LOADER_NAME, USE_ROI, ROI_CONFIG_FILE
    
    camera_url = None
    model_path = None
   
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
                        relay_config['port'] = int(value)
                    elif key == "camera_url":
                        camera_url = value
                    elif key == "loader_name":
                        LOADER_NAME = value
                    elif key == "model_path":
                        model_path = value
                        loader_params['model_path'] = value
                    elif key == "relay_ip":
                        relay_config['ip'] = value
                    elif key == "relay_start_cmd":
                        relay_config['relay_start_cmd'] = value
                    elif key == "relay_stop_cmd":
                        relay_config['relay_stop_cmd'] = value
                    elif key == "confidence":
                        loader_params['confidence'] = float(value)
                    elif key == "max_age":
                        loader_params['max_age'] = int(value)
                    elif key == "min_hits":
                        loader_params['min_hits'] = int(value)
                    elif key == "iou_threshold":
                        loader_params['iou_threshold'] = float(value)
                    elif key == "counting_line_y":
                        loader_params['counting_line_y'] = float(value)
                    elif key == "center_dot_position":
                        loader_params['center_dot_position'] = float(value)
                    elif key == "use_roi":
                        USE_ROI = value.lower() in ('true', 'yes', '1')
                    elif key == "roi_config_file":
                        ROI_CONFIG_FILE = value

                except Exception as e:
                    logger.error(f"Config parse error for line: {line} - {e}")
                    
    except Exception as e:
        logger.error(f"Failed to read config file: {e}")
        raise
        
    if 'port' not in relay_config:
        relay_config['port'] = 5000
        
    if not os.path.exists(OUTPUT_VIDEO_DIR):
        os.makedirs(OUTPUT_VIDEO_DIR)
    
    logger.info(f"{LOADER_NAME} parameters: {loader_params}")
    logger.info(f"ROI enabled: {USE_ROI}")
    
    return camera_url, model_path


# ==============================================================================
# FLASK API
# ==============================================================================

@flask_app.route('/api/getTargetForAI', methods=['POST'])
def get_target_from_java():
    """Receive target from Java backend"""
    global target, relay_tripped, vehicle_info, video_thread
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data"}), 400
       
        target_bags_str = data.get("total_bags")
        vehicle_number = data.get("vehicle_number", "NA")
        material_type = data.get("materialType", "NA")
        start_time = data.get("startTime", "NA")
       
        if not target_bags_str:
            return jsonify({"status": "error", "message": "Missing total_bags"}), 400

        try:
            target_bags = int(target_bags_str)
        except ValueError:
            return jsonify({"status": "error", "message": "total_bags must be int"}), 400

        with lock:
            target = target_bags
            relay_tripped = False
            vehicle_info = {
                "vehicleNumber": vehicle_number,
                "materialType": material_type,
                "startTime": start_time,
                "stopTime": "NA"
            }
       
        video_path = "N/A"
        if video_thread:
            video_thread.reset_counter()
            video_path = video_thread.output_path
            logger.info(f"New video path: {video_path}")

        start_belt()
        
        logger.info(f"🎯 Target set: {target_bags} bags")
        return jsonify({"status": "success", "video_path": video_path}), 200
       
    except Exception as e:
        logger.error(f"API Error in getTargetForAI: {e}")
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
# BAG COUNTER WITH BIDIRECTIONAL COUNTING AND ROI
# ==============================================================================

class RTSPBagCounter(Thread):
    """Production bag counter with bidirectional line-crossing logic and ROI"""
    
    def __init__(self, url, roi_polygon=None):
        super().__init__()
        self.url = url
        self.roi_polygon = roi_polygon
        self.stopped = False
        
        # Get loader-specific parameters
        self.confidence_threshold = loader_params.get('confidence', 0.25)
        self.counting_line_y = loader_params.get('counting_line_y', 0.52)
        self.center_dot_position = loader_params.get('center_dot_position', 0.5)
        
        max_age = loader_params.get('max_age', 20)
        min_hits = loader_params.get('min_hits', 1)
        iou_threshold = loader_params.get('iou_threshold', 0.3)
        
        # Initialize tracker
        self.tracker = Sort(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )
        
        # Counting data - Modified for bidirectional counting
        self.counted_bags = {}  # track_id -> 'counted' or 'uncounted'
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
        
        self.window_name = f"Bag Counter - {LOADER_NAME}"
        
        logger.info(f"Counter initialized: {LOADER_NAME} (line={self.counting_line_y}, dot={self.center_dot_position})")
        if self.roi_polygon:
            logger.info(f"ROI enabled with {len(self.roi_polygon)} points")
        else:
            logger.info(f"ROI disabled - using full frame")
    
    def init_video_writer(self, frame_width, frame_height):
        """Initialize video writer"""
        try:
            with lock:
                vehicle_number = vehicle_info.get("vehicleNumber", "NA")
            vehicle_number = "".join(c for c in vehicle_number if c.isalnum() or c in ('-', '_'))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(OUTPUT_VIDEO_DIR, f"{LOADER_NAME}_{vehicle_number}_{timestamp}.avi")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, 10.0, (frame_width, frame_height))
            self.is_recording = True
            logger.info(f"Video recording started: {self.output_path}")
        except Exception as e:
            logger.error(f"Video writer error: {e}")
            self.is_recording = False
    
    def release_video_writer(self):
        """Release video writer"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            logger.info(f"Video recording stopped: {self.output_path}")
    
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
    
    def draw_roi(self, frame):
        """Draw ROI polygon on frame"""
        if self.roi_polygon and len(self.roi_polygon) >= 3:
            # Draw semi-transparent overlay outside ROI
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.roi_polygon)], 255)
            
            # Create darkened overlay
            overlay = frame.copy()
            overlay[mask == 0] = overlay[mask == 0] * 0.3
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Draw ROI boundary
            cv2.polylines(frame, [np.array(self.roi_polygon)], True, (0, 255, 255), 2)
            
            # Draw ROI label
            cv2.putText(frame, "DETECTION REGION", (10, self.height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def check_line_crossing(self, track_id, center_x, center_y, line_y):
        """
        Check if tracker crossed the counting line and in which direction
        Returns: 'up' (bottom-to-top), 'down' (top-to-bottom), or None
        """
        current_time = time.time()
        self.tracker_positions[track_id].append((center_x, center_y, current_time))
        
        # Keep only last 10 positions
        if len(self.tracker_positions[track_id]) > 10:
            self.tracker_positions[track_id] = self.tracker_positions[track_id][-10:]
        
        # Need at least 2 positions
        if len(self.tracker_positions[track_id]) < 2:
            return None
        
        prev_y = self.tracker_positions[track_id][-2][1]
        curr_y = self.tracker_positions[track_id][-1][1]
        
        # Check crossing from bottom to top (counting direction - INCREMENT)
        if prev_y > line_y and curr_y <= line_y:
            return 'up'
        
        # Check crossing from top to bottom (opposite direction - DECREMENT)
        elif prev_y < line_y and curr_y >= line_y:
            return 'down'
        
        return None
    
    def process_frame(self, frame):
        """Process a single frame with bidirectional counting logic and ROI filtering"""
        frame_start = time.time()
        
        # Get counting line
        line_p1, line_p2 = self.get_line_coordinates()
        line_y = line_p1[1]
        
        # YOLO Detection
        try:
            results = model.predict(
                frame,
                imgsz=640,
                verbose=False,
                conf=self.confidence_threshold
            )[0]
        except Exception as e:
            logger.error(f"YOLO error: {e}")
            return frame
        
        # Prepare detections for tracker - FILTER BY ROI
        detections = np.empty((0, 5))
        detections_outside_roi = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Check if detection is in ROI
            if self.roi_polygon:
                if box_in_roi((x1, y1, x2, y2), self.roi_polygon):
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                else:
                    detections_outside_roi.append((x1, y1, x2, y2))
            else:
                # No ROI filtering
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
        
        # Update tracker (only with detections inside ROI)
        tracked_objects = self.tracker.update(detections)
        
        # Process tracked objects - UPDATED LOGIC FOR ALL BAGS
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Get tracking point
            center_x, center_y = self.get_tracking_point(x1, y1, x2, y2)
            
            # Check line crossing direction
            crossing_direction = self.check_line_crossing(track_id, center_x, center_y, line_y)
            
            if crossing_direction:
                if crossing_direction == 'up':  # Bottom to top - INCREMENT
                    # Only count if not already counted
                    if self.counted_bags.get(track_id) != 'counted':
                        self.counted_bags[track_id] = 'counted'
                        self.total_count += 1
                        logger.info(f"Bag #{self.total_count} COUNTED (ID: {track_id}) - crossed UP")
                
                elif crossing_direction == 'down':  # Top to bottom - DECREMENT
                    # Decrement counter for ANY bag crossing in reverse direction
                    # Prevent count from going below 0
                    if self.total_count > 0:
                        self.total_count -= 1
                        self.counted_bags[track_id] = 'uncounted'
                        logger.info(f"Bag UNCOUNTED (ID: {track_id}) - crossed DOWN. New count: {self.total_count}")
                    else:
                        logger.warning(f"Bag crossed DOWN (ID: {track_id}) but count is already 0")
                        self.counted_bags[track_id] = 'uncounted'
            
            # Determine color based on state
            bag_state = self.counted_bags.get(track_id, 'uncounted')
            color = (0, 255, 0) if bag_state == 'counted' else (0, 165, 255)  # Green if counted, Orange if not
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw tracking point
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw ID and state
            label = f"ID:{track_id}"
            if bag_state == 'counted':
                label += " [COUNTED]"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw detections outside ROI (in gray, not tracked)
        for (x1, y1, x2, y2) in detections_outside_roi:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(frame, "OUT OF ROI", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Draw ROI
        frame = self.draw_roi(frame)
        
        # Draw counting line
        cv2.line(frame, line_p1, line_p2, (0, 0, 255), 3)
        label_y = line_y - 10 if line_y > 30 else line_y + 25
        cv2.putText(frame, "COUNTING LINE", (line_p1[0] + 10, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw direction indicator
        arrow_x = self.width - 60
        arrow_start_y = line_y + 40
        arrow_end_y = line_y - 20
        cv2.arrowedLine(frame, (arrow_x, arrow_start_y), (arrow_x, arrow_end_y), 
                       (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(frame, "COUNT", (arrow_x - 40, arrow_start_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update live count
        with lock:
            global live_count, relay_tripped
            live_count = self.total_count
            current_target = target
            is_tripped = relay_tripped
        
        # Send live count update
        if self.last_sent_count is None or self.total_count != self.last_sent_count:
            self.send_live_count(self.total_count)
            self.last_sent_count = self.total_count
        
        # Check if target reached
        if current_target > 0 and self.total_count >= current_target and not is_tripped:
            logger.info(f"🎯 Target {current_target} reached!")
            with lock:
                vehicle_info["stopTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                relay_tripped = True
            self.handle_target_reached(self.total_count)
        
        # Draw statistics (dark red, top-right)
        DARK_RED = (0, 0, 150)
        MARGIN = 10
        
        # Camera name
        camera_text = f"Camera: {LOADER_NAME}"
        (text_w, text_h), _ = cv2.getTextSize(camera_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x_camera = self.width - text_w - MARGIN
        cv2.putText(frame, camera_text, (x_camera, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_RED, 2)
        
        # Count
        count_text = f"Count: {self.total_count} / {current_target}"
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
        else:
            logger.error(f"Failed to get initial frame")
            return
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        logger.info(f"Starting processing (recording starts when target is set)")
        
        try:
            while not self.stopped:
                ret, frame = self.cam_thread.get_frame()
                
                if not ret:
                    logger.warning(f"No frame. Reconnecting...")
                    self.cam_thread.stop()
                    time.sleep(2)
                    self.cam_thread = ThreadedCamera(self.url)
                    time.sleep(1)
                    continue
                
                self.frame_count += 1
                
                # Process frame only if we have a target or are recording
                with lock:
                    has_target = target > 0 or self.is_recording
                
                if has_target or self.frame_count == 1:
                    processed_frame = self.process_frame(frame)
                else:
                    # Just show live view without processing when no target
                    processed_frame = frame.copy()
                    # Draw ROI even when waiting for target
                    processed_frame = self.draw_roi(processed_frame)
                    cv2.putText(processed_frame, f"{LOADER_NAME}: WAITING FOR TARGET", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Save to video ONLY if recording is active
                if self.is_recording and self.video_writer:
                    self.video_writer.write(processed_frame)
                
                # Display
                cv2.imshow(self.window_name, processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info(f"Quit key pressed")
                    break
                
                # Log statistics
                if self.frame_count % 100 == 0:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    recording_status = "RECORDING" if self.is_recording else "IDLE (Waiting for target)"
                    logger.info(f"Frames: {self.frame_count}, Count: {self.total_count}, Avg FPS: {avg_fps:.1f}, Status: {recording_status}")
        
        except KeyboardInterrupt:
            logger.info(f"Interrupted")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"🧹 Cleaning up")
        
        if self.cam_thread:
            self.cam_thread.stop()
        
        # Stop recording if active
        if self.is_recording:
            self.release_video_writer()
        
        cv2.destroyWindow(self.window_name)
        
        # Final statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Final Stats - Frames: {self.frame_count}, Count: {self.total_count}, Duration: {elapsed:.1f}s")
    
    def stop(self):
        """Stop the counter"""
        self.stopped = True
        self.send_live_count(0)
    
    def send_live_count(self, current_count):
        """Send live count to Java backend"""
        if not JAVA_COUNT_UPDATE_API_URL:
            return
            
        with lock:
            info = vehicle_info.copy()
            ip = relay_config.get('ip', "NA")
            current_target = target
            
            payload = {
                "loader": LOADER_NAME,
                "ip": ip,
                "actualBags": current_count,
                "target": current_target,
                "vehicleNumber": info.get("vehicleNumber", "NA"),
                "materialType": info.get("materialType", "NA"),
                "status": "counting",
                "startTime": info.get("startTime", "NA"),
                "stopTime": info.get("stopTime", "NA")
            }
        
        try:
            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=0.5)
        except:
            pass
    
    def handle_target_reached(self, final_count):
        """Handle target reached event"""
        # Stop belt
        stop_belt()
        
        # Stop recording
        self.stop_recording()
        
        # Send final count
        self.send_final_count(final_count)
        
        # Clear target
        with lock:
            global target
            target = 0
    
    def send_final_count(self, final_count):
        """Send final count to Java backend"""
        if not JAVA_COUNT_UPDATE_API_URL:
            return
            
        try:
            with lock:
                info = vehicle_info.copy()
                ip = relay_config.get('ip', "NA")
                current_target = target
                payload = {
                    "loader": LOADER_NAME,
                    "ip": ip,
                    "actualBags": final_count,
                    "target": current_target,
                    "vehicleNumber": info.get("vehicleNumber", "NA"),
                    "materialType": info.get("materialType", "NA"),
                    "status": "completed",
                    "startTime": info.get("startTime", "NA"),
                    "stopTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            requests.post(JAVA_COUNT_UPDATE_API_URL, json=payload, timeout=5)
            logger.info(f"Final count sent: {final_count}")
        except Exception as e:
            logger.error(f"Failed to send final count: {e}")
    
    def reset_counter(self):
        """Reset counter for new vehicle and start recording"""
        # Stop any existing recording first
        if self.is_recording:
            self.stop_recording()
        
        with lock:
            self.counted_bags.clear()
            self.tracker_positions.clear()
            self.total_count = 0
            self.last_sent_count = None
            global live_count
            live_count = 0
            vehicle_info["stopTime"] = "NA"
        
        # Start new recording
        self.start_recording()
        
        # Now send initial count
        self.send_live_count(0)
        
        logger.info(f"Counter reset - Recording started")


# ==============================================================================
# MAIN
# ==============================================================================

def load_model(model_path):
    """Load YOLO model"""
    global model
    
    logger.info(f"Loading YOLO model: {model_path}")
    
    try:
        model = YOLO(model_path, task="detect")
        model.fuse()
        
        if torch.cuda.is_available():
            model.to('cuda')
            logger.info(f"Model loaded on GPU")
            try:
                model.half()
                logger.info(f"Model converted to FP16")
            except:
                logger.warning(f"FP16 conversion failed, using FP32")
        else:
            logger.info(f"Model loaded on CPU")
        
        logger.info(f"Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def start_api_server():
    """Start Flask API server"""
    logger.info(f"Starting API server on port {PYTHON_API_PORT}")
    t = Thread(target=lambda: flask_app.run(host='0.0.0.0', port=PYTHON_API_PORT, debug=False))
    t.daemon = True
    t.start()


def setup_roi(camera_url):
    """Setup ROI - either load existing or create new"""
    global roi_polygon
    
    if not USE_ROI:
        logger.info("ROI disabled in configuration")
        roi_polygon = None
        return
    
    # Try to load existing ROI
    roi_polygon = load_roi_config(ROI_CONFIG_FILE)
    
    if not roi_polygon:
        logger.info("No saved ROI found. Starting ROI selection...")
        logger.info("Attempting to capture first frame from camera...")
        
        # Try to get a frame from the camera
        cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        ret, first_frame = None, None
        for attempt in range(10):
            ret, first_frame = cap.read()
            if ret:
                break
            time.sleep(0.5)
        
        cap.release()
        
        if ret and first_frame is not None:
            selector = ROISelector(first_frame)
            roi_polygon = selector.select()
            
            if roi_polygon and len(roi_polygon) >= 3:
                save_roi_config(roi_polygon, ROI_CONFIG_FILE)
            else:
                logger.info("No ROI selected - using full frame")
                roi_polygon = None
        else:
            logger.error("Could not capture frame for ROI selection")
            logger.info("Proceeding without ROI (full frame will be used)")
            roi_polygon = None
    else:
        logger.info("Using previously saved ROI")
        logger.info(f"To create new ROI, delete: {ROI_CONFIG_FILE}")


def start_tracking(camera_url):
    """Start tracking thread"""
    global video_thread
    
    if video_thread:
        video_thread.stop()
    
    worker = RTSPBagCounter(camera_url, roi_polygon)
    worker.daemon = True
    worker.start()
    video_thread = worker
    
    logger.info(f"Camera worker started")


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("RTSP Production Bag Counting System - Single Camera with ROI")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        camera_url, model_path = load_config_properties()
        
        if not camera_url:
            logger.error("No camera configured. Exiting.")
            return
        
        logger.info(f"Camera configured: {camera_url}")
        
        # Load YOLO model
        load_model(model_path)
        
        # Setup ROI
        setup_roi(camera_url)
        
        # Start API server
        start_api_server()
        
        # Start tracking
        start_tracking(camera_url)
        
        logger.info("System running. Press Ctrl+C to exit.")
        logger.info("=" * 60)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        
        # Stop worker
        if video_thread:
            video_thread.stop()
        
        logger.info("=" * 60)
        logger.info("System stopped")
        logger.info("=" * 60)
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()