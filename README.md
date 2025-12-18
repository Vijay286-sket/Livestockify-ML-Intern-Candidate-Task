# Bird Counting and Weight Estimation System

A complete ML prototype for counting birds and estimating weights from poultry farm CCTV video footage using computer vision and machine learning.

## Overview

This system implements:
- **Bird Detection**: Using YOLOv8 pretrained model for accurate bird detection
- **Bird Tracking**: Stable tracking across frames using IoU-based association
- **Weight Estimation**: Feature-based regression to estimate bird weights from visual features
- **FastAPI Service**: RESTful API for video analysis
- **Video Annotation**: Generates annotated videos with bounding boxes and tracking information

## Features

### Detection & Tracking
- YOLOv8 pretrained model for bird detection
- IoU-based tracking with ID assignment
- Occlusion handling through disappearance tracking
- Configurable confidence and IoU thresholds

### Weight Estimation
- Feature-based regression using visual features:
  - Bounding box area (size indicator)
  - Aspect ratio (body shape)
  - Position in frame (depth estimation)
- Relative weight index (0-1 scale)
- Calibration support for absolute weight measurements
- Confidence estimation based on detection quality

### API Endpoints
- `GET /health` - Health check
- `POST /analyze_video` - Synchronous upload & analysis, returns counts, tracking sample, weight estimates, and artifact paths
- `POST /analyze_video_async` - Asynchronous processing that returns a task ID
- `GET /task_status/{task_id}` - Check async processing status
- `GET /task_results/{task_id}` - Get async analysis results
- `GET /download/{task_id}/{file_type}` - Download processed files
- `DELETE /task/{task_id}` - Delete task and files

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch
- FastAPI

### Setup

1. **Clone or download the project**:
```bash
cd "d:/Livestockify ML Intern"
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download dataset (optional for testing)**:
```bash
python download_dataset.py
python create_sample_video.py
```

## Usage

### Command Line Interface

Process a video directly from command line:

```bash
python main.py sample_chicken_video.mp4 --output-dir outputs --conf-thresh 0.5 --fps-sample 2
```

Parameters:
- `video_path`: Path to input video
- `--output-dir`: Output directory (default: outputs)
- `--conf-thresh`: Detection confidence threshold (default: 0.5)
- `--iou-thresh`: IoU threshold (default: 0.45)
- `--fps-sample`: Frame sampling rate (default: 2)

### FastAPI Service

Start the API server:

```bash
python src/api.py
```

The server will start at `http://localhost:8001`

### API Usage

#### 1. Health Check
```bash
curl http://localhost:8001/health
```

#### 2. Analyze Video (synchronous, primary)
```bash
curl -X POST "http://localhost:8001/analyze_video" \
  -F "video=@sample_chicken_video.mp4" \
  -F "fps_sample=2" \
  -F "conf_thresh=0.5" \
  -F "iou_thresh=0.45"
```

Example response (truncated):
```json
{
  "counts": [
    {"timestamp": 0.0, "count": 3},
    {"timestamp": 0.5, "count": 4}
  ],
  "tracks_sample": [
    {
      "track_id": 1,
      "bbox": [100, 120, 200, 260],
      "confidence": 0.85,
      "weight": 1500.0,
      "weight_confidence": 0.75
    }
  ],
  "weight_estimates": {
    "total_weight": 6000.0,
    "average_weight": 1500.0,
    "unit": "grams (estimated)"
  },
  "confidence": {
    "average_confidence": 0.80,
    "min_confidence": 0.65,
    "max_confidence": 0.95
  },
  "artifacts": {
    "annotated_video": "outputs/sample_chicken_video_annotated.mp4",
    "results_json": "outputs/analysis_results.json"
  },
  "metadata": {
    "fps_sample": 2,
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }
}
```

#### 3. Analyze Video (asynchronous, optional)
```bash
curl -X POST "http://localhost:8001/analyze_video_async" \
  -F "video=@sample_chicken_video.mp4" \
  -F "fps_sample=2" \
  -F "conf_thresh=0.5" \
  -F "iou_thresh=0.45"
```

Response:
```json
{
  "task_id": "uuid-string",
  "status": "queued",
  "message": "Video uploaded successfully. Processing started."
}
```

#### 4. Check Task Status
```bash
curl http://localhost:8001/task_status/{task_id}
```

#### 5. Get Results
```bash
curl http://localhost:8001/task_results/{task_id}
```

#### 6. Download Files
```bash
# Download annotated video
curl -O http://localhost:8001/download/{task_id}/video

# Download JSON results
curl -O http://localhost:8001/download/{task_id}/json
```

## Architecture

### Core Components

1. **BirdDetector** (`src/detector.py`)
   - YOLOv8 model initialization and inference
   - Detection filtering and feature extraction
   - Visual feature computation for weight estimation

2. **SimpleTracker** (`src/tracker.py`)
   - IoU-based track association
   - ID assignment and management
   - Occlusion handling with disappearance tracking
   - Track history maintenance

3. **WeightEstimator** (`src/weight_estimator.py`)
   - Feature-based regression model
   - Heuristic weight estimation from visual features
   - Calibration support for absolute measurements
   - Confidence estimation

4. **VideoProcessor** (`src/video_processor.py`)
   - Main processing pipeline
   - Video frame processing and annotation
   - Results compilation and storage
   - Progress tracking

5. **API Service** (`src/api.py`)
   - FastAPI endpoints
   - Background task processing
   - File upload and management
   - Task status tracking

### Detection + Tracking Logic

1. **Detection**: YOLOv8 detects birds in each frame with confidence scores
2. **Feature Extraction**: Bounding box area, aspect ratio, and position features extracted
3. **Tracking**: IoU-based association assigns stable IDs across frames
4. **Occlusion Handling**: Tracks maintained for N frames after disappearance
5. **ID Switch Minimization**: IoU threshold and track history reduce switches

### Weight Estimation Method

**Approach**: Feature-based regression with heuristic calibration

**Features Used**:
- Bounding box area (primary size indicator)
- Aspect ratio (body shape characteristics)
- Normalized position (depth estimation proxy)

**Method**:
1. Extract visual features from detected birds
2. Apply trained regression model (synthetic data for demo)
3. Use calibration factor for absolute measurements
4. Provide confidence based on detection quality

**Assumptions**:
- Larger bounding boxes indicate heavier birds
- Aspect ratio correlates with body condition
- Position in frame provides depth information
- Calibration with actual weights required for accuracy

**Calibration Requirements**:
- Known weight measurements for calibration
- Scale reference object in video
- Camera distance and angle information
- Consistent lighting conditions

## Output Format

### Annotated Video
- Bounding boxes around detected birds
- Track IDs and confidence scores
- Weight estimates and confidence
- Real-time count and aggregate weight overlay

### JSON Response Structure
```json
{
  "counts": [
    {"timestamp": 0.0, "count": 3},
    {"timestamp": 0.5, "count": 4}
  ],
  "tracks_sample": [
    {
      "track_id": 1,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85,
      "weight": 1500.0,
      "weight_confidence": 0.75
    }
  ],
  "weight_estimates": {
    "total_weight": 6000.0,
    "average_weight": 1500.0,
    "unit": "grams (estimated)"
  },
  "confidence": {
    "average_confidence": 0.80,
    "min_confidence": 0.65,
    "max_confidence": 0.95
  },
  "artifacts": {
    "annotated_video": "path/to/video.mp4",
    "results_json": "path/to/results.json"
  }
}
```

## Configuration

Key parameters in `src/config.py`:

```python
# Detection
YOLO_MODEL_SIZE = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Tracking
MAX_DISAPPEARED_FRAMES = 30
MIN_TRACK_LENGTH = 5

# Processing
DEFAULT_FPS_SAMPLE = 2
MAX_VIDEO_SIZE_MB = 100

# Weight Estimation
CALIBRATION_FACTOR = 0.1
WEIGHT_FEATURES = ['bbox_area', 'aspect_ratio', 'position_x', 'position_y']
```

## Performance Considerations

### Optimization Strategies
1. **Frame Sampling**: Process every Nth frame to reduce computation
2. **Model Selection**: Use YOLOv8n for faster inference
3. **Batch Processing**: Process multiple frames when possible
4. **Memory Management**: Efficient track storage and cleanup

### Expected Performance
- **Processing Speed**: ~2-5 FPS on CPU (depends on video resolution)
- **Memory Usage**: ~500MB for typical video processing
- **Accuracy**: ~85% detection accuracy, stable tracking for moderate occlusions

## Limitations and Future Improvements

### Current Limitations
1. **Weight Estimation**: Relative estimates only, requires calibration
2. **Occlusion Handling**: Limited for severe occlusions
3. **Lighting Sensitivity**: Performance varies with lighting conditions
4. **Camera Movement**: Assumes fixed camera position

### Potential Improvements
1. **Advanced Tracking**: Implement DeepSORT or ReID features
2. **3D Estimation**: Use stereo cameras for depth estimation
3. **Model Fine-tuning**: Train on domain-specific bird data
4. **Real-time Processing**: GPU acceleration for real-time performance

## Troubleshooting

### Common Issues

1. **Model Download Fails**:
   - Check internet connection
   - Verify PyTorch installation
   - Try manual model download

2. **Video Processing Fails**:
   - Check video format compatibility
   - Verify file permissions
   - Ensure sufficient disk space

3. **API Server Issues**:
   - Check port availability
   - Verify FastAPI installation
   - Check dependency versions

### Error Messages
- `"No module named 'ultralytics'"`: Install ultralytics package
- `"Could not open video"`: Check video file path and format
- `"Task not found"`: Verify task ID is correct

## License

This project is for educational and demonstration purposes. The dataset used is under Public Domain license.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- AddAssign appropriate tests
- Update documentation
- Verify functionality before submission

## Support

For issues and questions:
1. Check this README
2. Review code comments
3. Test with provided sample video
4. Check error logs for detailed information
