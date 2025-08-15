"""
Demo Video Creation Script
Creates a short demo video highlighting the key features of the traffic flow analysis system
"""

import cv2
import os
import numpy as np
from datetime import datetime

def create_demo_video(input_video_path, output_video_path, demo_duration=120):
    """
    Create a demo video from the processed output
    
    Args:
        input_video_path: Path to the processed video with annotations
        output_video_path: Path for the demo video output
        demo_duration: Duration of demo in seconds (default 2 minutes)
    """
    
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames for demo
    demo_frames = min(demo_duration * fps, total_frames)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating demo video...")
    print(f"Input: {input_video_path}")
    print(f"Output: {output_video_path}")
    print(f"Duration: {demo_frames / fps:.1f} seconds")
    
    # Key moments to capture (in seconds)
    key_moments = [
        0,    # Start
        10,   # Early detection
        30,   # Multiple vehicles
        60,   # Lane changes
        90,   # Heavy traffic
        110   # Final summary
    ]
    
    # Convert to frame numbers
    key_frames = [int(moment * fps) for moment in key_moments if moment * fps < total_frames]
    
    # Add intro screen
    intro_frame = create_intro_screen(width, height)
    for _ in range(fps * 3):  # 3 second intro
        out.write(intro_frame)
    
    # Process key segments
    for i, start_frame in enumerate(key_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Write segment title
        segment_title = get_segment_title(i)
        title_frame = create_title_frame(width, height, segment_title)
        for _ in range(fps):  # 1 second title
            out.write(title_frame)
        
        # Write segment footage
        segment_duration = fps * 5  # 5 seconds per segment
        for _ in range(segment_duration):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add segment label
            cv2.putText(frame, segment_title, (width - 300, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
    
    # Add outro with statistics
    outro_frame = create_outro_screen(width, height)
    for _ in range(fps * 3):  # 3 second outro
        out.write(outro_frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Demo video created successfully!")
    print(f"Location: {output_video_path}")

def create_intro_screen(width, height):
    """Create an introduction screen for the demo"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        frame[i, :] = [int(50 * (1 - i/height)), 
                       int(50 * (1 - i/height)), 
                       int(100 * (1 - i/height))]
    
    # Add title
    cv2.putText(frame, "Traffic Flow Analysis System", 
               (width//2 - 250, height//3),
               cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    
    # Add subtitle
    cv2.putText(frame, "Multi-Lane Vehicle Detection and Tracking", 
               (width//2 - 220, height//3 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Add features
    features = [
        "YOLOv8 Object Detection",
        "DeepSORT Tracking Algorithm",
        "Real-time Lane Analysis",
        "Automated Vehicle Counting"
    ]
    
    y_offset = height//2
    for feature in features:
        cv2.putText(frame, f"• {feature}", 
                   (width//2 - 150, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 150), 1)
        y_offset += 40
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    cv2.putText(frame, timestamp, 
               (width - 150, height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return frame

def create_title_frame(width, height, title):
    """Create a title frame for each segment"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 40)  # Dark background
    
    # Add title with border
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2
    
    # Shadow
    cv2.putText(frame, title, (text_x + 2, text_y + 2),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
    # Main text
    cv2.putText(frame, title, (text_x, text_y),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    
    return frame

def create_outro_screen(width, height):
    """Create an outro screen with summary statistics"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        frame[i, :] = [int(30 * (i/height)), 
                       int(40 * (i/height)), 
                       int(60 * (i/height))]
    
    # Add title
    cv2.putText(frame, "Analysis Complete", 
               (width//2 - 150, height//4),
               cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
    
    # Add mock statistics (replace with actual data)
    stats = [
        "Total Vehicles Detected: 127",
        "Lane 1: 45 vehicles",
        "Lane 2: 52 vehicles",
        "Lane 3: 30 vehicles",
        "",
        "Processing Time: 2.5 minutes",
        "Average FPS: 24.8"
    ]
    
    y_offset = height//2 - 50
    for stat in stats:
        if stat:
            cv2.putText(frame, stat, 
                       (width//2 - 150, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        y_offset += 40
    
    # Add footer
    cv2.putText(frame, "github.com/yourusername/traffic-flow-analysis", 
               (width//2 - 200, height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    return frame

def get_segment_title(index):
    """Get title for each demo segment"""
    titles = [
        "System Initialization",
        "Vehicle Detection",
        "Multi-Vehicle Tracking",
        "Lane Assignment",
        "Heavy Traffic Handling",
        "Final Results"
    ]
    return titles[index] if index < len(titles) else f"Segment {index + 1}"

def create_highlights_reel(video_path, output_path):
    """
    Create a highlights reel showing the best moments
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Creating highlights reel...")
    
    # Select interesting moments (customize based on your video)
    highlights = [
        {"start": 5, "duration": 3, "label": "First Detection"},
        {"start": 15, "duration": 3, "label": "Multiple Lanes"},
        {"start": 30, "duration": 3, "label": "Peak Traffic"},
        {"start": 45, "duration": 3, "label": "Tracking Accuracy"},
        {"start": 60, "duration": 3, "label": "System Performance"}
    ]
    
    for highlight in highlights:
        start_frame = int(highlight["start"] * fps)
        duration_frames = int(highlight["duration"] * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(duration_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add highlight label
            label = highlight["label"]
            cv2.rectangle(frame, (10, 10), (250, 50), (0, 0, 0), -1)
            cv2.putText(frame, label, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add progress bar
            progress = i / duration_frames
            bar_width = int(230 * progress)
            cv2.rectangle(frame, (10, 55), (10 + bar_width, 65), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 55), (240, 65), (255, 255, 255), 1)
            
            out.write(frame)
    
    cap.release()
    out.release()
    print(f"Highlights reel created: {output_path}")

def main():
    """Main execution"""
    print("Demo Video Creator")
    print("=" * 50)
    
    # Check if processed video exists
    processed_video = "output/output_video.mp4"
    
    if not os.path.exists(processed_video):
        print(f"Error: Processed video not found at {processed_video}")
        print("Please run traffic_flow_analyzer.py first to generate the processed video.")
        return
    
    # Create demo directory
    os.makedirs("demo", exist_ok=True)
    
    # Create full demo video
    demo_output = "demo/demo_video.mp4"
    create_demo_video(processed_video, demo_output, demo_duration=120)
    
    # Create highlights reel (shorter version)
    highlights_output = "demo/highlights.mp4"
    create_highlights_reel(processed_video, highlights_output)
    
    print("\n" + "=" * 50)
    print("Demo creation complete!")
    print(f"Full demo: {demo_output}")
    print(f"Highlights: {highlights_output}")
    print("\nYou can now upload these videos to Google Drive or GitHub.")

if __name__ == "__main__":
    main()