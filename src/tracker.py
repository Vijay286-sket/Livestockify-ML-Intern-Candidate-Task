"""
Bird tracking module using ByteTrack.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

try:
    from config import Config
except ModuleNotFoundError:
    from .config import Config  # type: ignore


@dataclass
class Track:
    """Track class for storing bird tracking information."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    frame_number: int
    features: Dict[str, float]
    
    def __post_init__(self):
        self.bbox = np.array(self.bbox)


class SimpleTracker:
    """Simple tracker for bird detection with ID assignment."""
    
    def __init__(self, max_disappeared: int = None):
        """Initialize the tracker.
        
        Args:
            max_disappeared: Maximum frames to keep track after disappearance
        """
        self.max_disappeared = max_disappeared or Config.MAX_DISAPPEARED_FRAMES
        self.next_id = 1
        self.tracks = {}  # track_id -> Track object
        self.disappeared = {}  # track_id -> frames disappeared
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # track_id -> history
        
    def update(self, detections: List[Dict[str, Any]], frame_number: int) -> List[Track]:
        """Update tracks with new detections.
        
        Args:
            detections: List of detections from current frame
            frame_number: Current frame number
            
        Returns:
            List of active tracks
        """
        if not detections:
            # No detections, mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._remove_track(track_id)
            return list(self.tracks.values())
        
        # Convert detections to numpy array for easier processing
        detection_bboxes = np.array([det['bbox'] for det in detections])
        detection_confs = np.array([det['confidence'] for det in detections])
        detection_features = [det.get('features', {}) for det in detections]
        
        # If no existing tracks, initialize all detections as new tracks
        if not self.tracks:
            for i, (bbox, conf, features) in enumerate(zip(detection_bboxes, detection_confs, detection_features)):
                self._add_track(bbox, conf, frame_number, features)
            return list(self.tracks.values())
        
        # Calculate IoU between existing tracks and new detections
        track_bboxes = np.array([track.bbox for track in self.tracks.values()])
        track_ids = list(self.tracks.keys())
        
        iou_matrix = self._calculate_iou_matrix(track_bboxes, detection_bboxes)
        
        # Hungarian algorithm for assignment (simplified greedy approach)
        matched_tracks, matched_detections = self._associate_detections(iou_matrix)
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            track_id = track_ids[track_idx]
            bbox = detection_bboxes[det_idx]
            conf = detection_confs[det_idx]
            features = detection_features[det_idx]
            
            # Update track
            self.tracks[track_id].bbox = bbox
            self.tracks[track_id].confidence = conf
            self.tracks[track_id].frame_number = frame_number
            self.tracks[track_id].features = features
            
            # Reset disappeared counter
            self.disappeared[track_id] = 0
            
            # Add to history
            self.track_history[track_id].append({
                'bbox': bbox.copy(),
                'frame': frame_number,
                'confidence': conf
            })
        
        # Mark unmatched tracks as disappeared
        unmatched_track_indices = set(range(len(track_ids))) - set(matched_tracks)
        for track_idx in unmatched_track_indices:
            track_id = track_ids[track_idx]
            self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                self._remove_track(track_id)
        
        # Create new tracks for unmatched detections
        unmatched_det_indices = set(range(len(detections))) - set(matched_detections)
        for det_idx in unmatched_det_indices:
            bbox = detection_bboxes[det_idx]
            conf = detection_confs[det_idx]
            features = detection_features[det_idx]
            self._add_track(bbox, conf, frame_number, features)
        
        return list(self.tracks.values())
    
    def _calculate_iou_matrix(self, tracks_bboxes: np.ndarray, detections_bboxes: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix between tracks and detections."""
        iou_matrix = np.zeros((len(tracks_bboxes), len(detections_bboxes)))
        
        for i, track_bbox in enumerate(tracks_bboxes):
            for j, det_bbox in enumerate(detections_bboxes):
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        return iou_matrix
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate_detections(self, iou_matrix: np.ndarray, iou_threshold: float = 0.3) -> Tuple[List[int], List[int]]:
        """Associate detections to tracks using greedy IoU matching."""
        matched_tracks = []
        matched_detections = []
        
        # Greedy matching - find highest IoU pairs first
        while True:
            max_iou = iou_threshold
            best_track, best_det = -1, -1
            
            for i in range(iou_matrix.shape[0]):
                for j in range(iou_matrix.shape[1]):
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_track, best_det = i, j
            
            if best_track == -1:  # No more matches above threshold
                break
            
            matched_tracks.append(best_track)
            matched_detections.append(best_det)
            
            # Mark matched row and column as used
            iou_matrix[best_track, :] = -1
            iou_matrix[:, best_det] = -1
        
        return matched_tracks, matched_detections
    
    def _add_track(self, bbox: np.ndarray, confidence: float, frame_number: int, features: Dict[str, float]):
        """Add a new track."""
        track = Track(
            track_id=self.next_id,
            bbox=bbox.copy(),
            confidence=confidence,
            frame_number=frame_number,
            features=features.copy()
        )
        
        self.tracks[self.next_id] = track
        self.disappeared[self.next_id] = 0
        self.track_history[self.next_id].append({
            'bbox': bbox.copy(),
            'frame': frame_number,
            'confidence': confidence
        })
        
        self.next_id += 1
    
    def _remove_track(self, track_id: int):
        """Remove a track."""
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]
    
    def get_track_count(self) -> int:
        """Get current number of active tracks."""
        return len(self.tracks)
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return {
            'active_tracks': len(self.tracks),
            'total_tracks_created': self.next_id - 1,
            'average_track_length': np.mean([len(history) for history in self.track_history.values()]) if self.track_history else 0
        }
