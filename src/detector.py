"""
Bird detection module using YOLOv8.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from ultralytics import YOLO

try:
    from config import Config
except ModuleNotFoundError:
    from .config import Config  # type: ignore


class BirdDetector:
    """Bird detection using YOLOv8 pretrained model."""
    
    def __init__(self, model_path: str = None):
        """Initialize the bird detector.
        
        Args:
            model_path: Path to YOLO model file. If None, uses default.
        """
        if model_path is None:
            model_path = Config.YOLO_MODEL_SIZE
        
        self.model = YOLO(model_path)
        self.target_class_idx = None
        self._find_target_class_idx()
    
    def _find_target_class_idx(self):
        """Find the index of the 'bird' class in the model."""
        for idx, class_name in enumerate(self.model.names.values()):
            if class_name.lower() == Config.TARGET_CLASS:
                self.target_class_idx = idx
                break
        
        if self.target_class_idx is None:
            raise ValueError(f"'{Config.TARGET_CLASS}' class not found in model")
    
    def detect(self, frame: np.ndarray, conf_thresh: float = None) -> List[Dict[str, Any]]:
        """Detect birds in a frame.
        
        Args:
            frame: Input frame as numpy array
            conf_thresh: Confidence threshold override
            
        Returns:
            List of detections with bbox, confidence, and class info
        """
        if conf_thresh is None:
            conf_thresh = Config.CONFIDENCE_THRESHOLD
        
        results = self.model(frame, conf=conf_thresh, iou=Config.IOU_THRESHOLD)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Only process bird detections
                    if int(box.cls) == self.target_class_idx:
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy().astype(int),  # [x1, y1, x2, y2]
                            'confidence': float(box.conf.cpu().numpy()),
                            'class_id': int(box.cls.cpu().numpy()),
                            'class_name': Config.TARGET_CLASS
                        }
                        detections.append(detection)
        
        return detections
    
    def get_detection_features(self, detection: Dict[str, Any], frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Extract features from detection for weight estimation.
        
        Args:
            detection: Detection dictionary
            frame_shape: (height, width) of frame
            
        Returns:
            Dictionary of extracted features
        """
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Bounding box area
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Aspect ratio
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Center position
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalized position (0-1)
        norm_x = center_x / frame_shape[1]
        norm_y = center_y / frame_shape[0]
        
        # Relative size (compared to frame)
        frame_area = frame_shape[0] * frame_shape[1]
        relative_size = bbox_area / frame_area
        
        return {
            'bbox_area': bbox_area,
            'aspect_ratio': aspect_ratio,
            'center_x': center_x,
            'center_y': center_y,
            'norm_x': norm_x,
            'norm_y': norm_y,
            'relative_size': relative_size,
            'width': width,
            'height': height
        }
