# Traffic Flow Analysis System

A comprehensive computer vision solution for multi-lane vehicle counting and tracking using YOLOv8 and DeepSORT.

## Features

- **Real-time Vehicle Detection**: Uses YOLOv8 for accurate vehicle detection
- **Multi-Lane Tracking**: Tracks vehicles across three distinct lanes
- **Duplicate Prevention**: DeepSORT algorithm ensures vehicles are counted only once
- **Visual Output**: Annotated video with lane boundaries and real-time counts
- **Data Export**: CSV file with detailed vehicle tracking information
- **Performance Optimized**: Designed for real-time or near real-time processing

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended for better performance)
- Minimum 8GB RAM
- 2GB free disk space

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/traffic-flow-analysis.git
cd traffic-flow-analysis
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Traffic Video

```bash
# Install yt-dlp if not already installed
pip install yt-dlp

# Download the video
yt-dlp -o traffic_video.mp4 https://www.youtube.com/watch?v=MNn9qKG2UFI
```

### 5. Download YOLOv8 Model (Automatic)

The script will automatically download the YOLOv8 nano model on first run.

## Project Structure

```
traffic-flow-analysis/
│
├── traffic_flow_analyzer.py   # Main script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                # Git ignore file
│
├── output/                   # Output directory (created automatically)
│   ├── output_video.mp4     # Processed video with annotations
│   ├── vehicle_counts.csv   # Vehicle tracking data
│   └── summary.txt          # Analysis summary
│
└── demo/                     # Demo materials
    └── demo_video.mp4       # Demo video (1-2 minutes)
```

## Usage

### Basic Usage

```bash
python traffic_flow_analyzer.py
```

### Advanced Usage

You can modify the script parameters in the `main()` function:

```python
video_path = "traffic_video.mp4"  # Input video path
output_dir = "output"              # Output directory
```

### Lane Configuration

Lanes are automatically defined based on video dimensions. To adjust lane boundaries for your specific video:

1. Open `traffic_flow_analyzer.py`
2. Locate the `LaneManager._define_lanes()` method
3. Adjust the polygon coordinates for each lane

Example:
```python
lanes = {
    1: np.array([
        [x1, y1], [x2, y2], [x3, y3], [x4, y4]
    ], np.int32),
    # ... more lanes
}
```

## Output Files

### 1. CSV File (`vehicle_counts.csv`)
Contains the following columns:
- `Vehicle_ID`: Unique identifier for each tracked vehicle
- `Lane`: Lane number (1, 2, or 3)
- `Frame`: Frame number when vehicle was first detected
- `Timestamp`: Time in seconds from video start

### 2. Annotated Video (`output_video.mp4`)
Features:
- Color-coded lane boundaries
- Bounding boxes around detected vehicles
- Vehicle IDs for tracking
- Real-time lane counts
- Total vehicle count

### 3. Summary Report (`summary.txt`)
Includes:
- Total frames processed
- Video duration
- Vehicle counts per lane
- Total vehicle count

## Performance Optimization

### For Better Performance:

1. **Use GPU Acceleration**
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Adjust Model Size**
   - `yolov8n.pt`: Nano (fastest, less accurate)
   - `yolov8s.pt`: Small (balanced)
   - `yolov8m.pt`: Medium (slower, more accurate)

3. **Modify Detection Confidence**
   ```python
   results = self.model(frame, conf=0.5)  # Adjust confidence threshold
   ```

4. **Skip Frames for Faster Processing**
   ```python
   # Process every nth frame
   if self.current_frame % 2 == 0:
       frame = self.process_frame(frame)
   ```

## Troubleshooting

### Common Issues:

1. **Video Download Failed**
   - Ensure yt-dlp is installed: `pip install yt-dlp`
   - Try alternative download methods or use a different video

2. **Out of Memory Error**
   - Reduce batch size
   - Use smaller YOLO model (nano version)
   - Process video in segments

3. **Low FPS Performance**
   - Ensure GPU is being utilized
   - Reduce video resolution
   - Use frame skipping

4. **Incorrect Lane Assignment**
   - Adjust lane polygon definitions
   - Ensure lanes match video perspective
   - Check coordinate system alignment

## Technical Approach

### Vehicle Detection
- **Model**: YOLOv8 (You Only Look Once v8)
- **Classes**: Cars, motorcycles, buses, trucks
- **Confidence Threshold**: 0.5

### Vehicle Tracking
- **Algorithm**: DeepSORT (Deep Simple Online and Realtime Tracking)
- **Features**: 
  - Kalman filtering for motion prediction
  - Deep appearance descriptor
  - Hungarian algorithm for assignment

### Lane Management
- **Method**: Polygon-based regions
- **Assignment**: Point-in-polygon test using OpenCV
- **Visualization**: Color-coded overlays with transparency

## Challenges and Solutions

### Challenge 1: Duplicate Counting
**Solution**: Implemented DeepSORT tracking with unique IDs to ensure each vehicle is counted only once.

### Challenge 2: Lane Occlusion
**Solution**: Used vehicle center points for lane assignment and adjusted confidence thresholds.

### Challenge 3: Real-time Performance
**Solution**: Used YOLOv8 nano model and optimized frame processing pipeline.

### Challenge 4: Variable Video Perspectives
**Solution**: Made lane definitions configurable and percentage-based relative to frame dimensions.

## Future Improvements

- [ ] Automatic lane detection using Hough transforms
- [ ] Speed estimation for vehicles
- [ ] Vehicle classification (car, truck, bus, motorcycle)
- [ ] Multi-camera support
- [ ] Web-based interface for real-time monitoring
- [ ] Integration with traffic management systems

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `opencv-python>=4.8.0`
- `ultralytics>=8.0.0`
- `deep-sort-realtime>=1.5.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `torch>=2.0.0`
- `yt-dlp>=2023.0.0`

## License

MIT License - See LICENSE file for details

## Author

[Your Name]
[Your Email]
[Your GitHub Profile]

## Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT algorithm by Nicolai Wojke
- OpenCV community
- Sample traffic video from YouTube

## Demo Video

A 1-2 minute demo video showcasing the system is available in the `demo/` directory or at:
[Google Drive Link]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].