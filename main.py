"""
Traffic Flow Analysis System
A comprehensive solution for multi-lane vehicle counting using computer vision
"""

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from datetime import datetime
import os
import sys
from urllib import request
import ssl
import certifi
from ultralytics import YOLO
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("Installing deep-sort-realtime...")
    import subprocess
    subprocess.check_call(["pip", "install", "deep-sort-realtime==1.3.2"])
    from deep_sort_realtime.deepsort_tracker import DeepSort
import time

class LaneManager:
    """Manages lane definitions and vehicle-lane assignments"""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lanes = self._define_lanes()
        self.lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR colors for each lane
        
    def _define_lanes(self):
        """Define three lanes based on frame dimensions - for aerial/drone view"""
        h, w = self.frame_height, self.frame_width
        
        # Extended lanes to capture vehicles at edges
        # Lane boundaries now extend further to capture edge vehicles
        lanes = {
            # Lane 1 - Left lane (extended left edge)
            1: np.array([
                [int(w * 0.05), int(h * 0.45)],  # Extended left edge
                [int(w * 0.40), int(h * 0.45)],  # Top-right
                [int(w * 0.35), int(h * 0.85)],  # Bottom-right
                [int(w * 0.00), int(h * 0.85)]   # Extended to edge
            ], np.int32),
            
            # Lane 2 - Middle lane
            2: np.array([
                [int(w * 0.40), int(h * 0.45)],  # Top-left
                [int(w * 0.65), int(h * 0.45)],  # Top-right
                [int(w * 0.60), int(h * 0.85)],  # Bottom-right
                [int(w * 0.35), int(h * 0.85)]   # Bottom-left
            ], np.int32),
            
            # Lane 3 - Right lane (extended right edge)
            3: np.array([
                [int(w * 0.65), int(h * 0.45)],  # Top-left
                [int(w * 0.95), int(h * 0.45)],  # Extended right edge
                [int(w * 1.00), int(h * 0.85)],  # Extended to edge
                [int(w * 0.60), int(h * 0.85)]   # Bottom-left
            ], np.int32)
        }
        
        return lanes
    
    def get_lane(self, x, y):
        """Determine which lane a point (x, y) belongs to"""
        point = np.array([x, y], dtype=np.float32)
        
        for lane_id, polygon in self.lanes.items():
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                return lane_id
        return None
    
    def draw_lanes(self, frame):
        """Draw lane boundaries on the frame"""
        overlay = frame.copy()
        
        for lane_id, polygon in self.lanes.items():
            # Fill the polygon with semi-transparent color
            cv2.fillPoly(overlay, [polygon], self.lane_colors[lane_id - 1])
            # Draw the polygon boundary
            cv2.polylines(frame, [polygon], True, self.lane_colors[lane_id - 1], 2)
        
        # Apply the overlay with transparency
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Add lane labels
        for lane_id, polygon in self.lanes.items():
            center = np.mean(polygon, axis=0).astype(int)
            cv2.putText(frame, f"Lane {lane_id}", tuple(center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

class VehicleTracker:
    """Handles vehicle detection and tracking"""
    
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed
        
        # Initialize DeepSORT tracker with compatible parameters
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=0.8,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=torch.cuda.is_available()
        )
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def detect_vehicles(self, frame):
        """Detect vehicles in the frame using YOLO"""
        results = self.model(frame, conf=0.3, classes=self.vehicle_classes)  # Lowered confidence for better detection
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Format for DeepSORT: [x1, y1, width, height]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox, conf, cls))
        
        return detections
    
    def update_tracks(self, detections, frame):
        """Update tracking with new detections"""
        if not detections:
            return self.tracker.update_tracks([], frame=frame)
        
        # Format detections for DeepSORT
        bboxes = [d[0] for d in detections]
        confidences = [d[1] for d in detections]
        classes = [d[2] for d in detections]
        
        # Create detection list for DeepSORT
        dets = []
        for bbox, conf, cls in zip(bboxes, confidences, classes):
            dets.append((bbox, conf, cls))
        
        # Update tracks - MUST pass frame for embeddings
        tracks = self.tracker.update_tracks(dets, frame=frame)
        return tracks

class TrafficFlowAnalyzer:
    """Main class for traffic flow analysis"""
    
    def __init__(self, video_path=None, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.cap = None
        self.fps = 0
        self.frame_width = 0
        self.frame_height = 0
        
        self.lane_manager = None
        self.vehicle_tracker = VehicleTracker()
        
        # Tracking data
        self.vehicle_data = []
        self.lane_counts = {1: 0, 2: 0, 3: 0}
        self.tracked_vehicles = {}  # Track which vehicles we've already counted
        self.current_frame = 0
        
        # For counting line crossing
        self.counting_lines = {}
        self.vehicles_counted = set()  # Store IDs of vehicles that have been counted
        
    def download_video(self, url, output_path="traffic_video.mp4"):
        """Download video from YouTube (simplified - you'll need yt-dlp for actual YouTube)"""
        print(f"Note: For YouTube videos, please use yt-dlp to download the video first")
        print(f"Command: yt-dlp -o {output_path} {url}")
        return output_path
    
    def initialize_video(self):
        """Initialize video capture and properties"""
        if not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.lane_manager = LaneManager(self.frame_width, self.frame_height)
        
        # Define counting lines at the bottom edge of each lane (85% of height)
        self.counting_line_y = int(self.frame_height * 0.80)  # Counting line position
        
        print(f"Video initialized: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        print(f"Counting line set at y={self.counting_line_y}")
        return True
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect vehicles
        detections = self.vehicle_tracker.detect_vehicles(frame)
        
        # Update tracks - pass frame for embeddings
        tracks = self.vehicle_tracker.update_tracks(detections, frame)
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), (self.frame_width, self.counting_line_y), 
                (255, 255, 0), 2)
        cv2.putText(frame, "Counting Line", (10, self.counting_line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Process each tracked vehicle
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            
            # Calculate center point of vehicle (bottom center for better counting)
            center_x = int((bbox[0] + bbox[2]) / 2)
            bottom_y = int(bbox[3])  # Use bottom of bounding box for counting
            
            # Determine lane
            lane = self.lane_manager.get_lane(center_x, bottom_y)
            
            if lane is not None:
                # Check if vehicle crossed the counting line
                vehicle_key = f"{track_id}_{lane}"
                
                # Initialize tracking if new vehicle
                if track_id not in self.tracked_vehicles:
                    self.tracked_vehicles[track_id] = {
                        'first_lane': lane,
                        'current_lane': lane,
                        'first_frame': self.current_frame,
                        'last_frame': self.current_frame,
                        'positions': [(center_x, bottom_y)],
                        'crossed_line': False
                    }
                else:
                    self.tracked_vehicles[track_id]['current_lane'] = lane
                    self.tracked_vehicles[track_id]['last_frame'] = self.current_frame
                    self.tracked_vehicles[track_id]['positions'].append((center_x, bottom_y))
                
                # Check if vehicle crosses the counting line (moving downward)
                if (not self.tracked_vehicles[track_id]['crossed_line'] and 
                    bottom_y >= self.counting_line_y and
                    len(self.tracked_vehicles[track_id]['positions']) > 1):
                    
                    prev_y = self.tracked_vehicles[track_id]['positions'][-2][1]
                    if prev_y < self.counting_line_y:  # Crossed from top to bottom
                        if vehicle_key not in self.vehicles_counted:
                            self.vehicles_counted.add(vehicle_key)
                            self.lane_counts[lane] += 1
                            self.tracked_vehicles[track_id]['crossed_line'] = True
                            
                            # Record vehicle data
                            timestamp = self.current_frame / self.fps
                            self.vehicle_data.append({
                                'Vehicle_ID': track_id,
                                'Lane': lane,
                                'Frame': self.current_frame,
                                'Timestamp': round(timestamp, 2),
                                'X_Position': center_x,
                                'Y_Position': bottom_y
                            })
                            
                            print(f"Vehicle {track_id} counted in Lane {lane} at frame {self.current_frame}")
                
                # Draw bounding box and ID
                color = self.lane_manager.lane_colors[lane - 1]
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), 
                            color, 2)
                
                # Add vehicle info text
                label = f"ID: {track_id} | L{lane}"
                cv2.putText(frame, label, 
                          (int(bbox[0]), int(bbox[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw tracking line
                if len(self.tracked_vehicles[track_id]['positions']) > 1:
                    points = self.tracked_vehicles[track_id]['positions'][-20:]  # Last 20 positions
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], color, 1)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw statistics on the frame with improved visibility"""
        # Create a semi-transparent background for stats
        stats_bg = np.zeros((150, 300, 3), dtype=np.uint8)
        stats_bg[:] = (30, 30, 30)  # Dark background
        
        # Draw lane counts with better colors
        y_offset = 30
        
        # Lane 1 - Use bright blue for better visibility
        text = f"Lane 1: {self.lane_counts[1]} vehicles"
        cv2.putText(stats_bg, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)  # Bright blue
        y_offset += 30
        
        # Lane 2 - Green
        text = f"Lane 2: {self.lane_counts[2]} vehicles"
        cv2.putText(stats_bg, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # Lane 3 - Red
        text = f"Lane 3: {self.lane_counts[3]} vehicles"
        cv2.putText(stats_bg, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
        
        # Draw total count
        total = sum(self.lane_counts.values())
        cv2.putText(stats_bg, f"Total: {total} vehicles", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add stats background to frame with better opacity
        frame[0:150, 0:300] = cv2.addWeighted(frame[0:150, 0:300], 0.3, stats_bg, 0.7, 0)
        
        # Draw frame counter and time (larger, more visible)
        info_bg = np.zeros((80, 200, 3), dtype=np.uint8)
        info_bg[:] = (30, 30, 30)
        
        cv2.putText(info_bg, f"Frame: {self.current_frame}", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        time_str = f"Time: {self.current_frame/self.fps:.1f}s"
        cv2.putText(info_bg, time_str, 
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Place info in top-right corner
        frame[0:80, self.frame_width-200:self.frame_width] = cv2.addWeighted(
            frame[0:80, self.frame_width-200:self.frame_width], 0.3, info_bg, 0.7, 0)
        
        return frame
    
    def process_video(self):
        """Main processing loop with optimized speed"""
        if not self.initialize_video():
            return False
        
        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(self.output_dir, 'output_video.mp4')
        out = cv2.VideoWriter(out_path, fourcc, self.fps, 
                            (self.frame_width, self.frame_height))
        
        print("Processing video...")
        print("Press 'q' to stop processing early and save results")
        print("Processing in optimized mode for faster completion...")
        start_time = time.time()
        
        # Process every frame for accuracy (no skipping)
        frame_skip = 1  # Process every frame for real-time accuracy
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            if self.current_frame % frame_skip == 0:
                # Draw lanes
                frame = self.lane_manager.draw_lanes(frame)
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Draw statistics
                frame = self.draw_statistics(frame)
                
                # Write frame
                out.write(frame)
                
                # Display frame only every 10th frame for speed
                if self.current_frame % 10 == 0:
                    # Resize for display to improve performance
                    display_frame = cv2.resize(frame, (960, 540))
                    cv2.imshow('Traffic Flow Analysis', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped by user")
                        break
            else:
                # For skipped frames, still write the original
                out.write(frame)
            
            self.current_frame += 1
            
            # Progress update
            if self.current_frame % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = self.current_frame / elapsed
                remaining_frames = 9000 - self.current_frame  # Approximate total
                eta = remaining_frames / fps_actual if fps_actual > 0 else 0
                print(f"Frame {self.current_frame} | Processing: {fps_actual:.1f} FPS | ETA: {eta:.0f}s")
                print(f"Counts - L1: {self.lane_counts[1]}, L2: {self.lane_counts[2]}, L3: {self.lane_counts[3]}")
        
        # Cleanup
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save all results
        self.save_results()
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"✓ Output video saved to: {out_path}")
        print(f"✓ CSV data saved to: {os.path.join(self.output_dir, 'vehicle_counts.csv')}")
        print(f"✓ Summary saved to: {os.path.join(self.output_dir, 'summary.txt')}")
        
        return True
    
    def save_results(self):
        """Save analysis results to CSV and summary files"""
        # Save detailed vehicle data to CSV
        if self.vehicle_data:
            df = pd.DataFrame(self.vehicle_data)
            csv_path = os.path.join(self.output_dir, 'vehicle_counts.csv')
            df.to_csv(csv_path, index=False)
            print(f"Vehicle data saved to: {csv_path}")
            print(f"Total records in CSV: {len(self.vehicle_data)}")
        else:
            print("No vehicle data to save")
        
        # Save detailed tracking data
        tracking_data = []
        for vehicle_id, info in self.tracked_vehicles.items():
            tracking_data.append({
                'Vehicle_ID': vehicle_id,
                'First_Lane': info['first_lane'],
                'Last_Lane': info['current_lane'],
                'First_Frame': info['first_frame'],
                'Last_Frame': info['last_frame'],
                'Duration_Frames': info['last_frame'] - info['first_frame'],
                'Duration_Seconds': (info['last_frame'] - info['first_frame']) / self.fps
            })
        
        if tracking_data:
            tracking_df = pd.DataFrame(tracking_data)
            tracking_csv_path = os.path.join(self.output_dir, 'vehicle_tracking.csv')
            tracking_df.to_csv(tracking_csv_path, index=False)
            print(f"Tracking data saved to: {tracking_csv_path}")
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Traffic Flow Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Frames Processed: {self.current_frame}\n")
            f.write(f"Video Duration: {self.current_frame / self.fps:.2f} seconds\n\n")
            f.write("Vehicle Counts by Lane:\n")
            f.write("-" * 20 + "\n")
            for lane_id in range(1, 4):
                f.write(f"Lane {lane_id}: {self.lane_counts[lane_id]} vehicles\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Vehicles: {sum(self.lane_counts.values())}\n\n")
            f.write(f"Unique Vehicles Tracked: {len(self.tracked_vehicles)}\n")
            f.write(f"Total Detection Records: {len(self.vehicle_data)}\n")
        
        print(f"Summary saved to: {summary_path}")
        
        # Print summary to console
        print("\n" + "=" * 40)
        print("FINAL SUMMARY")
        print("=" * 40)
        for lane_id in range(1, 4):
            print(f"Lane {lane_id}: {self.lane_counts[lane_id]} vehicles")
        print(f"Total: {sum(self.lane_counts.values())} vehicles")
        print(f"Unique vehicles tracked: {len(self.tracked_vehicles)}")
        print("=" * 40)

def main():
    """Main execution function"""
    print("Traffic Flow Analysis System")
    print("=" * 40)
    
    # Configuration
    video_path = "traffic_video.mp4"  # Path to your downloaded video
    output_dir = "output"
    
    # Note about video download
    print("\nIMPORTANT: Please download the video first using:")
    print("yt-dlp -o traffic_video.mp4 https://www.youtube.com/watch?v=MNn9qKG2UFI")
    print("\nMake sure the video is saved as 'traffic_video.mp4' in the current directory")
    print("=" * 40)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"\nError: Video file '{video_path}' not found!")
        print("Please download the video first using the command above.")
        return
    
    # Create analyzer
    analyzer = TrafficFlowAnalyzer(video_path, output_dir)
    
    # Process video
    analyzer.process_video()

if __name__ == "__main__":
    main()