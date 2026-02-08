"""
RTSP Camera Bag Counting System - TEST VERSION WITH ROI SELECTION
Real-time bag counting with bidirectional logic and custom detection region
NO API REQUIRED - For testing with video files
"""

import cv2
import numpy as np
import time
from datetime import datetime
import logging
from pathlib import Path
import os
import sys
from threading import Thread, Lock
from ultralytics import YOLO
from sort import Sort
import torch
from collections import defaultdict
import json

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_bag_counting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === GLOBALS ===
model = None
lock = Lock()

# Test configuration - EDIT THESE
VIDEO_PATH = "jk.mp4"  # Change this to your video file path
MODEL_PATH = "best.pt"      # Change this to your model path
LOADER_NAME = "TEST-LOADER"
TARGET_COUNT = 30               # Set a test target (optional, just for display)
ROI_CONFIG_FILE = "roi_config.json"  # File to save/load ROI coordinates

# Detection parameters
CONFIDENCE_THRESHOLD = 0.25
MAX_AGE = 20
MIN_HITS = 1
IOU_THRESHOLD = 0.3
COUNTING_LINE_Y = 0.2          # Line position (0.0=top, 1.0=bottom)
CENTER_DOT_POSITION = 0.5       # Tracking point (0.0=top, 0.5=middle, 1.0=bottom)

# Video playback control
PLAYBACK_SPEED = 1.0            # 1.0 = normal speed, 0.5 = half speed, 2.0 = double speed
FRAME_DELAY = int(30 / PLAYBACK_SPEED)  # Delay in milliseconds

# ROI (Region of Interest) settings
USE_ROI = True                  # Set to False to disable ROI
roi_points = []                 # Will store ROI polygon points
roi_selected = False


# ==============================================================================
# ROI SELECTION FUNCTIONS
# ==============================================================================

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
            "ESC: Cancel",
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
        logger.info("  - LEFT CLICK to add points")
        logger.info("  - RIGHT CLICK to remove last point")
        logger.info("  - Press ENTER when done (minimum 3 points)")
        logger.info("  - Press ESC to cancel and use full frame")
        logger.info("=" * 80)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter
                if len(self.points) >= 3:
                    self.done = True
                    logger.info(f"✅ ROI selected with {len(self.points)} points")
                    break
                else:
                    logger.warning("⚠️ Need at least 3 points to create ROI")
            
            elif key == 27:  # ESC
                self.points = []
                logger.info("❌ ROI selection cancelled - using full frame")
                break
        
        cv2.destroyWindow(self.window_name)
        return self.points


def save_roi_config(points, filepath=ROI_CONFIG_FILE):
    """Save ROI configuration to file"""
    try:
        config = {
            'roi_points': points,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"💾 ROI configuration saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save ROI config: {e}")
        return False


def load_roi_config(filepath=ROI_CONFIG_FILE):
    """Load ROI configuration from file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)
            points = [tuple(p) for p in config.get('roi_points', [])]
            if points:
                logger.info(f"✅ ROI configuration loaded from {filepath}")
                logger.info(f"   {len(points)} points, saved on {config.get('timestamp', 'unknown')}")
                return points
    except Exception as e:
        logger.error(f"❌ Failed to load ROI config: {e}")
    return None


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


# ==============================================================================
# BAG COUNTER WITH BIDIRECTIONAL COUNTING AND ROI
# ==============================================================================

class VideoBagCounter:
    """Bag counter for video file testing with bidirectional logic and ROI"""
    
    def __init__(self, video_path, roi_polygon=None):
        self.video_path = video_path
        self.roi_polygon = roi_polygon
        self.stopped = False
        
        # Initialize tracker
        self.tracker = Sort(
            max_age=MAX_AGE,
            min_hits=MIN_HITS,
            iou_threshold=IOU_THRESHOLD
        )
        
        # Counting data - Modified for bidirectional counting
        self.counted_bags = {}  # track_id -> 'counted' or 'uncounted'
        self.total_count = 0
        self.tracker_positions = defaultdict(list)
        
        # Track crossing events for detailed logging
        self.increment_events = []
        self.decrement_events = []
        
        # Statistics
        self.frame_count = 0
        self.start_time = datetime.now()
        self.fps_history = []
        
        # Video capture
        self.cap = None
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.original_fps = 0
        
        self.window_name = f"Bag Counter TEST - {LOADER_NAME}"
        
        logger.info(f"✅ Counter initialized for video: {video_path}")
        logger.info(f"📊 Parameters: line={COUNTING_LINE_Y}, dot={CENTER_DOT_POSITION}, conf={CONFIDENCE_THRESHOLD}")
        if self.roi_polygon:
            logger.info(f"🔲 ROI enabled with {len(self.roi_polygon)} points")
        else:
            logger.info(f"🔲 ROI disabled - using full frame")
    
    def get_line_coordinates(self):
        """Get counting line coordinates"""
        y = int(self.height * COUNTING_LINE_Y)
        return (0, y), (self.width, y)
    
    def get_tracking_point(self, x1, y1, x2, y2):
        """Get tracking point based on center_dot_position"""
        center_x = (x1 + x2) // 2
        center_y = int(y1 + (y2 - y1) * CENTER_DOT_POSITION)
        return center_x, center_y
    
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
        
        # Check crossing from bottom to top (counting direction)
        if prev_y > line_y and curr_y <= line_y:
            return 'up'
        
        # Check crossing from top to bottom (belt reversal)
        elif prev_y < line_y and curr_y >= line_y:
            return 'down'
        
        return None
    
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
                conf=CONFIDENCE_THRESHOLD
            )[0]
        except Exception as e:
            logger.error(f"❌ YOLO error: {e}")
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
        
        # Process tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Get tracking point
            center_x, center_y = self.get_tracking_point(x1, y1, x2, y2)
            
            # Check line crossing direction
            crossing_direction = self.check_line_crossing(track_id, center_x, center_y, line_y)
            
            if crossing_direction:
                current_state = self.counted_bags.get(track_id, 'uncounted')
                
                if crossing_direction == 'up':  # Bottom to top - INCREMENT
                    if current_state == 'uncounted':
                        self.counted_bags[track_id] = 'counted'
                        self.total_count += 1
                        event = {
                            'frame': self.frame_count,
                            'track_id': track_id,
                            'count': self.total_count,
                            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        }
                        self.increment_events.append(event)
                        logger.info(f"✅ [Frame {self.frame_count}] Bag #{self.total_count} COUNTED (ID: {track_id}) - crossed UP")
                
                elif crossing_direction == 'down':  # Top to bottom - DECREMENT
                    if current_state == 'counted':
                        self.counted_bags[track_id] = 'uncounted'
                        self.total_count -= 1
                        event = {
                            'frame': self.frame_count,
                            'track_id': track_id,
                            'count': self.total_count,
                            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        }
                        self.decrement_events.append(event)
                        logger.info(f"⬇️ [Frame {self.frame_count}] Bag UNCOUNTED (ID: {track_id}) - crossed DOWN. New count: {self.total_count}")
            
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
        
        # Draw statistics (dark red, top-right)
        DARK_RED = (0, 0, 150)
        MARGIN = 10
        
        # Camera name
        camera_text = f"TEST MODE: {LOADER_NAME}"
        (text_w, text_h), _ = cv2.getTextSize(camera_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x_camera = self.width - text_w - MARGIN
        cv2.putText(frame, camera_text, (x_camera, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_RED, 2)
        
        # Count
        count_text = f"Count: {self.total_count} / {TARGET_COUNT}"
        (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x_count = self.width - text_w - MARGIN
        cv2.putText(frame, count_text, (x_count, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, DARK_RED, 3)
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}/{self.total_frames}"
        (text_w, text_h), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x_frame = self.width - text_w - MARGIN
        cv2.putText(frame, frame_text, (x_frame, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DARK_RED, 2)
        
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
        cv2.putText(frame, fps_text, (x_fps, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DARK_RED, 2)
        
        # Event counters (bottom-right)
        events_text = f"UP: {len(self.increment_events)} | DOWN: {len(self.decrement_events)}"
        cv2.putText(frame, events_text, (self.width - 200, self.height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp (bottom-left)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main processing loop for video file"""
        logger.info(f"📹 Opening video file: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            logger.error(f"❌ Failed to open video file: {self.video_path}")
            return
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"📊 Video properties: {self.width}x{self.height}, {self.total_frames} frames, {self.original_fps:.2f} FPS")
        logger.info(f"⏯️ Playback speed: {PLAYBACK_SPEED}x")
        
        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        logger.info(f"▶️ Starting video processing")
        logger.info("=" * 80)
        
        try:
            while not self.stopped:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.info("📹 End of video reached")
                    break
                
                self.frame_count += 1
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow(self.window_name, processed_frame)
                
                # Control playback speed
                key = cv2.waitKey(FRAME_DELAY) & 0xFF
                
                if key == ord('q'):
                    logger.info(f"🛑 Quit key pressed")
                    break
                elif key == ord(' '):  # Spacebar to pause
                    logger.info("⏸️ PAUSED - Press any key to continue")
                    cv2.waitKey(0)
                elif key == ord('r'):  # R to restart
                    logger.info("🔄 Restarting video")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.reset_counter()
                
                # Log statistics every 100 frames
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% | Count: {self.total_count} | Frame: {self.frame_count}/{self.total_frames}")
        
        except KeyboardInterrupt:
            logger.info(f"⏹️ Interrupted")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources and print summary"""
        logger.info("=" * 80)
        logger.info(f"🧹 Cleaning up")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyWindow(self.window_name)
        
        # Final statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   Total Frames Processed: {self.frame_count}")
        logger.info(f"   Final Count: {self.total_count}")
        logger.info(f"   Increment Events: {len(self.increment_events)}")
        logger.info(f"   Decrement Events: {len(self.decrement_events)}")
        logger.info(f"   Duration: {elapsed:.1f}s")
        logger.info(f"   Average FPS: {self.frame_count / elapsed:.1f}")
        
        # Print detailed event log
        if self.increment_events or self.decrement_events:
            logger.info("")
            logger.info("=" * 80)
            logger.info("📋 DETAILED EVENT LOG:")
            logger.info("=" * 80)
            
            all_events = []
            for event in self.increment_events:
                all_events.append(('INCREMENT', event))
            for event in self.decrement_events:
                all_events.append(('DECREMENT', event))
            
            all_events.sort(key=lambda x: x[1]['frame'])
            
            for event_type, event in all_events:
                symbol = "✅" if event_type == "INCREMENT" else "⬇️"
                logger.info(f"{symbol} Frame {event['frame']:5d} | ID {event['track_id']:3d} | {event_type:9s} | Count: {event['count']:3d} | {event['timestamp']}")
        
        logger.info("=" * 80)
    
    def stop(self):
        """Stop the counter"""
        self.stopped = True
    
    def reset_counter(self):
        """Reset counter"""
        self.counted_bags.clear()
        self.tracker_positions.clear()
        self.total_count = 0
        self.increment_events.clear()
        self.decrement_events.clear()
        self.frame_count = 0
        logger.info(f"🔄 Counter reset")


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
            logger.info(f"✅ Model loaded on GPU")
            try:
                model.half()
                logger.info(f"✅ Model converted to FP16")
            except:
                logger.warning(f"⚠️ FP16 conversion failed, using FP32")
        else:
            logger.info(f"⚠️ Model loaded on CPU")
        
        logger.info(f"✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("RTSP Bag Counting System - TEST MODE WITH ROI SELECTION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("CONTROLS:")
    logger.info("  Q - Quit")
    logger.info("  SPACE - Pause/Resume")
    logger.info("  R - Restart video")
    logger.info("")
    logger.info("=" * 80)
    
    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"❌ Video file not found: {VIDEO_PATH}")
        logger.error(f"Please update VIDEO_PATH in the script to point to your test video")
        return
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        logger.error(f"Please update MODEL_PATH in the script to point to your model")
        return
    
    try:
        # Load YOLO model
        load_model(MODEL_PATH)
        
        # Handle ROI selection
        roi_polygon = None
        
        if USE_ROI:
            # Try to load existing ROI
            roi_polygon = load_roi_config()
            
            # If no saved ROI or user wants to reselect
            if not roi_polygon:
                logger.info("📹 Opening video for ROI selection...")
                cap = cv2.VideoCapture(VIDEO_PATH)
                ret, first_frame = cap.read()
                cap.release()
                
                if ret:
                    selector = ROISelector(first_frame)
                    roi_polygon = selector.select()
                    
                    if roi_polygon and len(roi_polygon) >= 3:
                        save_roi_config(roi_polygon)
                    else:
                        logger.info("⚠️ No ROI selected - using full frame")
                        roi_polygon = None
                else:
                    logger.error("❌ Could not read first frame for ROI selection")
            else:
                logger.info("✅ Using previously saved ROI")
                logger.info("💡 Delete roi_config.json to create a new ROI")
        
        # Create and run counter
        counter = VideoBagCounter(VIDEO_PATH, roi_polygon)
        counter.run()
        
        logger.info("=" * 80)
        logger.info("Test completed")
        logger.info("=" * 80)
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()