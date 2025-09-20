# ar_engine/marker_tracker.py
"""
Core marker tracking functionality using OpenCV
Handles feature detection, matching, and homography estimation
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarkerPoint:
    """Represents a feature point in the marker"""
    x: float
    y: float
    response: float
    angle: float
    scale: float

@dataclass
class TrackingResult:
    """Results from marker tracking"""
    detected: bool
    homography: Optional[np.ndarray]
    matches_count: int
    detection_time: float
    confidence_score: float
    transformed_corners: Optional[np.ndarray]

class ARMarkerTracker:
    """Advanced marker tracking using multiple algorithms"""
    
    def __init__(self, 
                 algorithm='ORB',
                 max_features=1000,
                 match_threshold=0.7,
                 min_matches=15,
                 ransac_threshold=5.0):
        """
        Initialize marker tracker
        
        Args:
            algorithm: Feature detection algorithm ('ORB', 'SIFT', 'AKAZE')
            max_features: Maximum features to detect
            match_threshold: Feature matching threshold
            min_matches: Minimum matches required for detection
            ransac_threshold: RANSAC threshold for homography
        """
        self.algorithm = algorithm
        self.max_features = max_features
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        
        # Initialize feature detector
        self._init_detector()
        
        # Initialize matcher
        if algorithm == 'SIFT':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Performance tracking
        self.detection_history = []
        self.performance_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'average_time': 0.0,
            'average_confidence': 0.0
        }
    
    def _init_detector(self):
        """Initialize the feature detector based on algorithm choice"""
        if self.algorithm == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
        elif self.algorithm == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
        elif self.algorithm == 'AKAZE':
            self.detector = cv2.AKAZE_create()
        else:
            logger.warning(f"Unknown algorithm {self.algorithm}, falling back to ORB")
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.algorithm = 'ORB'
    
    def create_marker_template(self, image_path: str) -> Dict:
        """
        Create marker template from image
        
        Args:
            image_path: Path to marker image
            
        Returns:
            Dictionary containing marker template data
        """
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast if needed
            gray = self._enhance_contrast(gray)
            
            # Detect features
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            
            if len(keypoints) < self.min_matches:
                logger.warning(f"Only {len(keypoints)} features found, tracking may be unstable")
            
            # Convert keypoints to serializable format
            kp_data = []
            for kp in keypoints:
                kp_data.append({
                    'x': float(kp.pt[0]),
                    'y': float(kp.pt[1]),
                    'angle': float(kp.angle) if kp.angle >= 0 else 0.0,
                    'size': float(kp.size),
                    'response': float(kp.response)
                })
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(keypoints, gray)
            
            template = {
                'image_path': image_path,
                'width': img.shape[1],
                'height': img.shape[0],
                'keypoints': kp_data,
                'descriptors': descriptors.tobytes() if descriptors is not None else b'',
                'feature_count': len(keypoints),
                'algorithm': self.algorithm,
                'quality_score': quality_score,
                'creation_time': time.time()
            }
            
            logger.info(f"✅ Marker template created: {len(keypoints)} features, quality: {quality_score:.2f}")
            return template
            
        except Exception as e:
            logger.error(f"❌ Failed to create marker template: {str(e)}")
            return None
    
    def track_marker(self, frame: np.ndarray, template: Dict) -> TrackingResult:
        """
        Track marker in video frame
        
        Args:
            frame: Input video frame (BGR)
            template: Marker template from create_marker_template()
            
        Returns:
            TrackingResult object with detection results
        """
        start_time = time.time()
        
        try:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            gray_frame = self._enhance_contrast(gray_frame)
            
            # Detect features in current frame
            kp_frame, des_frame = self.detector.detectAndCompute(gray_frame, None)
            
            if des_frame is None or len(kp_frame) < 10:
                return TrackingResult(False, None, 0, time.time() - start_time, 0.0, None)
            
            # Reconstruct template descriptors
            template_desc = np.frombuffer(template['descriptors'], dtype=np.uint8)
            if len(template_desc) == 0:
                return TrackingResult(False, None, 0, time.time() - start_time, 0.0, None)
            
            # Reshape descriptors based on algorithm
            desc_size = 32 if self.algorithm in ['ORB', 'AKAZE'] else 128
            template_desc = template_desc.reshape(-1, desc_size)
            
            # Match features
            matches = self.matcher.match(template_desc, des_frame)
            
            if len(matches) < self.min_matches:
                return TrackingResult(False, None, len(matches), time.time() - start_time, 0.0, None)
            
            # Sort matches by distance (quality)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter good matches
            good_matches = []
            for match in matches:
                if match.distance < self.match_threshold * matches[0].distance if matches[0].distance > 0 else self.match_threshold:
                    good_matches.append(match)
            
            if len(good_matches) < self.min_matches:
                return TrackingResult(False, None, len(good_matches), time.time() - start_time, 0.0, None)
            
            # Extract matched points
            src_pts = []
            dst_pts = []
            
            for match in good_matches:
                template_kp = template['keypoints'][match.queryIdx]
                src_pts.append([template_kp['x'], template_kp['y']])
                dst_pts.append(kp_frame[match.trainIdx].pt)
            
            src_pts = np.float32(src_pts).reshape(-1, 1, 2)
            dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
            
            # Find homography
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, 
                cv2.RANSAC, 
                self.ransac_threshold
            )
            
            if homography is None:
                return TrackingResult(False, None, len(good_matches), time.time() - start_time, 0.0, None)
            
            # Calculate transformed corners
            h, w = template['height'], template['width']
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, homography)
            
            # Calculate confidence score
            inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
            confidence = min(1.0, (len(good_matches) / 50) * inlier_ratio)
            
            detection_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(True, detection_time, confidence)
            
            return TrackingResult(
                detected=True,
                homography=homography,
                matches_count=len(good_matches),
                detection_time=detection_time,
                confidence_score=confidence,
                transformed_corners=transformed_corners
            )
            
        except Exception as e:
            logger.error(f"Marker tracking error: {str(e)}")
            detection_time = time.time() - start_time
            self._update_performance_stats(False, detection_time, 0.0)
            return TrackingResult(False, None, 0, detection_time, 0.0, None)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization for better feature detection"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _calculate_quality_score(self, keypoints: List, image: np.ndarray) -> float:
        """Calculate marker quality score based on feature distribution and strength"""
        if len(keypoints) == 0:
            return 0.0
        
        # Feature count score (0-40 points)
        count_score = min(40, len(keypoints) / 25 * 40)
        
        # Feature strength score (0-30 points)
        avg_response = np.mean([kp.response for kp in keypoints])
        strength_score = min(30, avg_response * 1000)
        
        # Spatial distribution score (0-30 points)
        if len(keypoints) >= 4:
            points = np.array([kp.pt for kp in keypoints])
            hull = cv2.convexHull(points.astype(np.float32))
            hull_area = cv2.contourArea(hull)
            image_area = image.shape[0] * image.shape[1]
            distribution_score = min(30, (hull_area / image_area) * 60)
        else:
            distribution_score = 0
        
        total_score = (count_score + strength_score + distribution_score) / 100
        return min(1.0, total_score)
    
    def _update_performance_stats(self, success: bool, detection_time: float, confidence: float):
        """Update performance tracking statistics"""
        self.performance_stats['total_detections'] += 1
        
        if success:
            self.performance_stats['successful_detections'] += 1
        
        # Update averages
        total = self.performance_stats['total_detections']
        self.performance_stats['average_time'] = (
            (self.performance_stats['average_time'] * (total - 1) + detection_time) / total
        )
        
        if success:
            success_count = self.performance_stats['successful_detections']
            self.performance_stats['average_confidence'] = (
                (self.performance_stats['average_confidence'] * (success_count - 1) + confidence) / success_count
            )
        
        # Keep history of last 100 detections
        self.detection_history.append({
            'success': success,
            'time': detection_time,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        total = self.performance_stats['total_detections']
        successful = self.performance_stats['successful_detections']
        
        return {
            'algorithm': self.algorithm,
            'total_detections': total,
            'successful_detections': successful,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'average_detection_time': self.performance_stats['average_time'],
            'average_confidence': self.performance_stats['average_confidence'],
            'settings': {
                'max_features': self.max_features,
                'match_threshold': self.match_threshold,
                'min_matches': self.min_matches,
                'ransac_threshold': self.ransac_threshold
            }
        }
    
    def optimize_settings(self, target_fps: int = 30):
        """Auto-optimize settings for target FPS"""
        avg_time = self.performance_stats['average_time']
        target_time = 1.0 / target_fps * 0.5  # Use 50% of frame time for detection
        
        if avg_time > target_time:
            # Reduce features for better performance
            self.max_features = max(500, int(self.max_features * 0.8))
            self._init_detector()
            logger.info(f"Optimized: Reduced features to {self.max_features} for {target_fps} FPS")
        elif avg_time < target_time * 0.5 and self.performance_stats['average_confidence'] < 0.8:
            # Increase features for better accuracy
            self.max_features = min(2000, int(self.max_features * 1.2))
            self._init_detector()
            logger.info(f"Optimized: Increased features to {self.max_features} for better accuracy")
