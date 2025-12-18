"""
Weight estimation module for birds using feature-based regression.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class WeightEstimator:
    """Weight estimation using feature-based regression."""
    
    def __init__(self, calibration_factor: float = None):
        """Initialize weight estimator.
        
        Args:
            calibration_factor: Factor to convert relative weight to grams
        """
        self.calibration_factor = calibration_factor or Config.CALIBRATION_FACTOR
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_names = Config.WEIGHT_FEATURES
        
        # For demonstration, we'll use a simple heuristic model
        # In practice, this would be trained on actual weight data
        self._setup_heuristic_model()
    
    def _setup_heuristic_model(self):
        """Setup a heuristic model for weight estimation.
        
        This creates a simple model that estimates relative weight based on
        visual features. In a real system, this would be trained on actual
        weight measurements.
        """
        # Create synthetic training data for demonstration
        # This represents the relationship between visual features and weight
        np.random.seed(42)
        n_samples = 100
        
        # Generate synthetic features
        bbox_areas = np.random.uniform(1000, 10000, n_samples)  # Pixel area
        aspect_ratios = np.random.uniform(0.5, 2.0, n_samples)
        positions_x = np.random.uniform(0, 1, n_samples)
        positions_y = np.random.uniform(0, 1, n_samples)
        
        # Create synthetic weights (heuristic relationship)
        # Larger birds have larger bbox areas and specific aspect ratios
        base_weights = np.sqrt(bbox_areas) * 0.5  # Base weight from size
        aspect_adjustment = (1.0 / aspect_ratios) * 100  # Taller birds weigh more
        position_adjustment = (positions_y * 50)  # Birds further back appear smaller
        
        weights = base_weights + aspect_adjustment - position_adjustment
        weights += np.random.normal(0, 50, n_samples)  # Add noise
        
        # Scale to realistic bird weights (grams)
        weights = np.clip(weights, 500, 5000)  # Typical chicken weights
        
        # Prepare training data
        X = np.column_stack([bbox_areas, aspect_ratios, positions_x, positions_y])
        y = weights
        
        # Train the model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def extract_features(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> List[Dict[str, float]]:
        """Extract features from detections for weight estimation.
        
        Args:
            detections: List of detection dictionaries
            frame_shape: (height, width) of frame
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate features
            bbox_area = (x2 - x1) * (y2 - y1)
            aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 1.0
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Normalize position
            norm_x = center_x / frame_shape[1]
            norm_y = center_y / frame_shape[0]
            
            features = {
                'bbox_area': bbox_area,
                'aspect_ratio': aspect_ratio,
                'position_x': norm_x,
                'position_y': norm_y
            }
            
            features_list.append(features)
        
        return features_list
    
    def estimate_weights(self, tracks: List[Any]) -> Dict[str, Any]:
        """Estimate weights for tracked birds.
        
        Args:
            tracks: List of Track objects with features
            
        Returns:
            Dictionary with weight estimates and metadata
        """
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        weights = {}
        confidences = {}
        features_used = []
        
        for track in tracks:
            track_id = track.track_id
            
            # Extract features from track
            track_features = []
            for feature_name in self.feature_names:
                if feature_name in track.features:
                    track_features.append(track.features[feature_name])
                else:
                    # Use default values if feature missing
                    default_values = {
                        'bbox_area': 5000,
                        'aspect_ratio': 1.0,
                        'position_x': 0.5,
                        'position_y': 0.5
                    }
                    track_features.append(default_values.get(feature_name, 0))
            
            # Predict weight
            X = np.array(track_features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            weight_pred = self.model.predict(X_scaled)[0]
            
            # Calculate confidence based on detection confidence and bbox size
            # Larger, clearer detections have higher confidence
            size_confidence = min(1.0, track.features.get('bbox_area', 0) / 10000)
            detection_confidence = track.confidence
            combined_confidence = (size_confidence + detection_confidence) / 2
            
            # Apply calibration factor to convert to grams
            calibrated_weight = weight_pred * self.calibration_factor
            
            weights[track_id] = calibrated_weight
            confidences[track_id] = combined_confidence
            features_used.append(track_features)
        
        # Calculate aggregate statistics
        weight_values = list(weights.values())
        confidence_values = list(confidences.values())
        
        result = {
            'per_bird_weights': weights,
            'per_bird_confidence': confidences,
            'aggregate_weight': {
                'total_weight': sum(weight_values),
                'average_weight': np.mean(weight_values) if weight_values else 0,
                'median_weight': np.median(weight_values) if weight_values else 0,
                'std_weight': np.std(weight_values) if weight_values else 0,
                'min_weight': min(weight_values) if weight_values else 0,
                'max_weight': max(weight_values) if weight_values else 0
            },
            'confidence_stats': {
                'average_confidence': np.mean(confidence_values) if confidence_values else 0,
                'min_confidence': min(confidence_values) if confidence_values else 0,
                'max_confidence': max(confidence_values) if confidence_values else 0
            },
            'metadata': {
                'method': 'feature_based_regression',
                'calibration_factor': self.calibration_factor,
                'features_used': self.feature_names,
                'num_birds': len(weights),
                'weight_unit': 'grams (estimated)',
                'note': 'Weights are relative estimates. Calibration with actual scale data required for absolute accuracy.'
            }
        }
        
        return result
    
    def get_weight_index(self, tracks: List[Any]) -> Dict[str, float]:
        """Get relative weight index (normalized 0-1) for birds.
        
        This provides a relative size comparison without requiring calibration.
        
        Args:
            tracks: List of Track objects
            
        Returns:
            Dictionary mapping track_id to weight index (0-1)
        """
        if not tracks:
            return {}
        
        # Extract bbox areas as primary size indicator
        areas = []
        for track in tracks:
            area = track.features.get('bbox_area', 0)
            areas.append(area)
        
        if not areas or max(areas) == 0:
            return {track.track_id: 0.5 for track in tracks}
        
        # Normalize to 0-1 scale
        min_area, max_area = min(areas), max(areas)
        range_area = max_area - min_area
        
        weight_indices = {}
        for track, area in zip(tracks, areas):
            if range_area > 0:
                index = (area - min_area) / range_area
            else:
                index = 0.5  # All birds same size
            weight_indices[track.track_id] = index
        
        return weight_indices
    
    def calibrate_from_known_weights(self, known_weights: Dict[int, float], tracks: List[Any]) -> float:
        """Calibrate the model using known weight measurements.
        
        Args:
            known_weights: Dictionary mapping track_id to known weight in grams
            tracks: List of tracks with features
            
        Returns:
            Updated calibration factor
        """
        if not known_weights or not tracks:
            return self.calibration_factor
        
        # Extract features and weights for calibration
        calibration_features = []
        calibration_weights = []
        
        for track in tracks:
            track_id = track.track_id
            if track_id in known_weights:
                track_features = []
                for feature_name in self.feature_names:
                    track_features.append(track.features.get(feature_name, 0))
                
                calibration_features.append(track_features)
                calibration_weights.append(known_weights[track_id])
        
        if not calibration_features:
            return self.calibration_factor
        
        # Predict weights using current model
        X = np.array(calibration_features)
        X_scaled = self.scaler.transform(X)
        predicted_weights = self.model.predict(X_scaled)
        
        # Calculate calibration factor
        actual_weights = np.array(calibration_weights)
        ratios = actual_weights / predicted_weights
        
        # Use median ratio to avoid outliers
        new_calibration_factor = np.median(ratios)
        
        self.calibration_factor = new_calibration_factor
        return new_calibration_factor


# Import at the end to avoid circular imports, supporting both script/package modes
try:
    from config import Config
except ModuleNotFoundError:
    from .config import Config  # type: ignore
