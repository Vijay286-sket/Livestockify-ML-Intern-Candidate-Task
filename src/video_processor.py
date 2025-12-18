"""
Video processing pipeline for bird detection, tracking, and weight estimation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
from tqdm import tqdm

try:
    from detector import BirdDetector
    from tracker import SimpleTracker, Track
    from weight_estimator import WeightEstimator
    from config import Config
except ModuleNotFoundError:
    from .detector import BirdDetector  # type: ignore
    from .tracker import SimpleTracker, Track  # type: ignore
    from .weight_estimator import WeightEstimator  # type: ignore
    from .config import Config  # type: ignore


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class VideoProcessor:
    """Main video processing pipeline."""
    
    def __init__(self, 
                 conf_thresh: float = None,
                 iou_thresh: float = None,
                 fps_sample: int = None):
        """Initialize the video processor.
        
        Args:
            conf_thresh: Detection confidence threshold
            iou_thresh: IoU threshold for detection
            fps_sample: Frame sampling rate (process every Nth frame)
        """
        self.conf_thresh = conf_thresh or Config.CONFIDENCE_THRESHOLD
        self.iou_thresh = iou_thresh or Config.IOU_THRESHOLD
        self.fps_sample = fps_sample or Config.DEFAULT_FPS_SAMPLE
        
        # Initialize components
        self.detector = BirdDetector()
        self.tracker = SimpleTracker()
        self.weight_estimator = WeightEstimator()
        
        # Results storage
        self.results = {
            'counts': [],
            'tracks': [],
            'weight_estimates': [],
            'metadata': {}
        }
    
    def process_video(self, video_path: str, output_dir: str = None) -> Dict[str, Any]:
        """Process a video file for bird detection, tracking, and weight estimation.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing all results
        """
        video_path = Path(video_path)
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {frame_count} frames, {duration:.2f}s")
        
        # Setup video writer for annotated output
        output_video_path = output_dir / f"{video_path.stem}{Config.ANNOTATED_VIDEO_SUFFIX}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_number = 0
        processed_frames = 0
        
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_number % self.fps_sample == 0:
                    # Detect birds
                    detections = self.detector.detect(frame, self.conf_thresh)
                    
                    # Extract features for weight estimation
                    frame_shape = frame.shape[:2]
                    for detection in detections:
                        features = self.detector.get_detection_features(detection, frame_shape)
                        detection['features'] = features
                    
                    # Update tracker
                    tracks = self.tracker.update(detections, frame_number)
                    
                    # Estimate weights
                    weight_results = self.weight_estimator.estimate_weights(tracks)
                    
                    # Store results
                    timestamp = frame_number / fps
                    self._store_frame_results(timestamp, tracks, weight_results)
                    
                    # Annotate frame
                    annotated_frame = self._annotate_frame(frame, tracks, weight_results, timestamp)
                    out.write(annotated_frame)
                    
                    processed_frames += 1
                
                frame_number += 1
                pbar.update(1)
        
        # Release video handles
        cap.release()
        out.release()
        
        # Compile final results
        final_results = self._compile_results(video_path, output_dir, processed_frames)
        
        # Convert all numpy types before saving
        final_results = convert_numpy_types(final_results)
        
        # Save results to JSON
        results_path = output_dir / Config.RESULTS_JSON
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"Processing complete!")
        print(f"Annotated video: {output_video_path}")
        print(f"Results JSON: {results_path}")
        
        return final_results
    
    def _store_frame_results(self, timestamp: float, tracks: List[Track], weight_results: Dict[str, Any]):
        """Store results for current frame."""
        # Store count
        count = len(tracks)
        self.results['counts'].append({
            'timestamp': float(timestamp),
            'count': int(count)
        })
        
        # Store track information
        frame_tracks = []
        for track in tracks:
            track_info = {
                'track_id': int(track.track_id),
                'bbox': track.bbox.tolist(),
                'confidence': float(track.confidence),
                'frame_number': int(track.frame_number),
                'features': convert_numpy_types(track.features)
            }
            
            # Add weight information
            if track.track_id in weight_results['per_bird_weights']:
                track_info['weight'] = float(weight_results['per_bird_weights'][track.track_id])
                track_info['weight_confidence'] = float(weight_results['per_bird_confidence'][track.track_id])
            
            frame_tracks.append(track_info)
        
        self.results['tracks'].append({
            'timestamp': timestamp,
            'tracks': frame_tracks
        })
        
        # Store aggregate weight information
        self.results['weight_estimates'].append({
            'timestamp': float(timestamp),
            'aggregate_weight': convert_numpy_types(weight_results['aggregate_weight']),
            'confidence_stats': convert_numpy_types(weight_results['confidence_stats']),
            'num_birds': int(weight_results['metadata']['num_birds'])
        })
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List[Track], 
                        weight_results: Dict[str, Any], timestamp: float) -> np.ndarray:
        """Annotate frame with detection and tracking information."""
        annotated_frame = frame.copy()
        
        # Colors for different tracks
        colors = self._get_colors(len(tracks))
        
        for i, track in enumerate(tracks):
            color = colors[i % len(colors)]
            bbox = track.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = [
                f"ID:{track.track_id}",
                f"Conf:{track.confidence:.2f}"
            ]
            
            # Add weight information
            if track.track_id in weight_results['per_bird_weights']:
                weight = weight_results['per_bird_weights'][track.track_id]
                weight_conf = weight_results['per_bird_confidence'][track.track_id]
                label_parts.append(f"Wt:{weight:.0f}g")
                label_parts.append(f"WtConf:{weight_conf:.2f}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add overall information overlay
        info_text = [
            f"Timestamp: {timestamp:.2f}s",
            f"Bird Count: {len(tracks)}",
            f"Total Weight: {weight_results['aggregate_weight']['total_weight']:.0f}g",
            f"Avg Weight: {weight_results['aggregate_weight']['average_weight']:.0f}g"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame
    
    def _get_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        colors = []
        for i in range(n):
            hue = i * 180 // n
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(color.tolist()))
        return colors
    
    def _compile_results(self, video_path: Path, output_dir: Path, 
                        processed_frames: int) -> Dict[str, Any]:
        """Compile final results summary."""
        
        # Calculate statistics
        counts = [c['count'] for c in self.results['counts']]
        avg_count = float(np.mean(counts)) if counts else 0.0
        max_count = int(max(counts)) if counts else 0
        min_count = int(min(counts)) if counts else 0
        
        # Weight statistics
        if self.results['weight_estimates']:
            total_weights = [w['aggregate_weight']['total_weight'] for w in self.results['weight_estimates']]
            avg_total_weight = float(np.mean(total_weights)) if total_weights else 0.0
            max_total_weight = float(max(total_weights)) if total_weights else 0.0
        else:
            avg_total_weight = max_total_weight = 0.0
        
        # Sample tracks for API response
        sample_tracks = []
        if self.results['tracks']:
            # Get tracks from middle of video
            middle_idx = len(self.results['tracks']) // 2
            sample_tracks = self.results['tracks'][middle_idx]['tracks'][:5]  # First 5 tracks
        
        # Latest per-frame weight summary (aggregate + confidence + num_birds)
        latest_weight_summary = self.results['weight_estimates'][-1] if self.results['weight_estimates'] else {}

        final_results = {
            'metadata': {
                'video_path': str(video_path),
                'output_directory': str(output_dir),
                'processed_frames': int(processed_frames),
                'confidence_threshold': float(self.conf_thresh),
                'iou_threshold': float(self.iou_thresh),
                'fps_sample': int(self.fps_sample),
                'processing_time': datetime.now().isoformat()
            },
            'counts': self.results['counts'],
            'tracks_sample': sample_tracks,
            # Expose the latest aggregate weight stats in a compact form
            'weight_estimates': latest_weight_summary,
            'statistics': {
                'bird_count': {
                    'average': avg_count,
                    'maximum': max_count,
                    'minimum': min_count,
                    'total_frames': len(counts)
                },
                'weight': {
                    'average_total_weight': avg_total_weight,
                    'maximum_total_weight': max_total_weight,
                    'unit': 'grams (estimated)'
                }
            },
            'artifacts': {
                'annotated_video': str(output_dir / f"{video_path.stem}{Config.ANNOTATED_VIDEO_SUFFIX}.mp4"),
                'results_json': str(output_dir / Config.RESULTS_JSON)
            },
            'tracking_method': 'Simple IoU-based tracking',
            'detection_model': 'YOLOv8 pretrained',
            'weight_estimation_method': 'Feature-based regression with heuristic calibration',
            'notes': [
                'Weights are relative estimates based on visual features',
                'Calibration with actual scale data required for absolute accuracy',
                'Tracking uses IoU-based association with disappearance handling',
                'Occlusions are handled by maintaining tracks for N frames after disappearance'
            ]
        }
        
        return final_results
    
    def get_sample_api_response(self) -> Dict[str, Any]:
        """Get a sample response formatted for the API."""
        if not self.results['counts']:
            return {}
        
        # Get the most recent weight estimates
        latest_weight_est = self.results['weight_estimates'][-1] if self.results['weight_estimates'] else {}
        
        # Get sample tracks
        sample_tracks = []
        if self.results['tracks']:
            middle_idx = len(self.results['tracks']) // 2
            tracks = self.results['tracks'][middle_idx]['tracks']
            sample_tracks = [
                {
                    'track_id': t['track_id'],
                    'bbox': t['bbox'],
                    'confidence': t['confidence'],
                    'weight': t.get('weight', 0),
                    'weight_confidence': t.get('weight_confidence', 0)
                }
                for t in tracks[:5]
            ]
        
        return {
            'counts': self.results['counts'][-10:],  # Last 10 counts
            'tracks_sample': sample_tracks,
            'weight_estimates': latest_weight_est.get('aggregate_weight', {}),
            'confidence': latest_weight_est.get('confidence_stats', {}),
            'metadata': {
                'method': 'feature_based_regression',
                'unit': 'grams (estimated)',
                'calibration_required': True
            }
        }
