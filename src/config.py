"""
Configuration settings for bird detection and tracking system.
"""

from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for bird detection and tracking."""
    
    # Model paths and settings
    YOLO_MODEL_SIZE = "yolov8n.pt"  # Nano model for faster inference
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    TARGET_CLASS = "bird"  # Class name for birds in COCO dataset
    
    # Tracking settings
    MAX_DISAPPEARED_FRAMES = 30  # Frames to keep track after bird disappears
    MIN_TRACK_LENGTH = 5  # Minimum frames to consider a valid track
    
    # Video processing
    DEFAULT_FPS_SAMPLE = 2  # Sample every Nth frame for processing
    MAX_VIDEO_SIZE_MB = 100  # Maximum video size for processing
    
    # Weight estimation settings
    WEIGHT_FEATURES = ['bbox_area', 'aspect_ratio', 'position_x', 'position_y']
    CALIBRATION_FACTOR = 0.1  # Placeholder calibration factor
    
    # Output settings
    OUTPUT_DIR = Path("outputs")
    ANNOTATED_VIDEO_SUFFIX = "_annotated"
    RESULTS_JSON = "analysis_results.json"
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8001
    UPLOAD_DIR = Path("uploads")
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get YOLO model configuration."""
        return {
            "model": cls.YOLO_MODEL_SIZE,
            "conf": cls.CONFIDENCE_THRESHOLD,
            "iou": cls.IOU_THRESHOLD,
            "device": "cpu"  # Use CPU for compatibility
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
