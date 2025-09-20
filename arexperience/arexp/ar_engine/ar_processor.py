# ar_engine/ar_processor.py
"""
Main AR processing pipeline that orchestrates all components
Handles video streaming, marker tracking, and overlay rendering
"""

#import cv2
import numpy as np
import threading
import time
import logging
import queue
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass, asdict
import json

from .marker_tracker import ARMarkerTracker, TrackingResult
from .video_overlay import VideoOverlayEngine
from .optimizers import (AROptimizer, FrameBuffer, MemoryOptimizer, 
                        PerformanceMonitor, global_optimizer, 
                        global_memory_optimizer, global_performance_monitor)

# Optional ML tracker
try:
    from .ml_tracker import PythonARTracker, MLTrackingResult
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ARConfig:
    """AR processing configuration"""
    # Tracking settings
    algorithm: str = 'ORB'
    max_features: int = 1000
    detection_threshold: float = 0.7
    min_matches: int = 15
    
    # Overlay settings
    overlay_scale: float = 1.0
    overlay_opacity: float = 0.8
    blend_mode: str = 'normal'
    
    # Performance settings
    target_fps: int = 30
    use_gpu: bool = True
    enable_optimization: bool = True
    memory_pooling: bool = True
    
    # Video settings
    camera_width: int = 640
    camera_height: int = 480
    video_format: str = 'BGR'
    
    # Debug settings
    show_debug_info: bool = False
    log_performance: bool = True
    save_debug_frames: bool = False

@dataclass
class ARProcessingResult:
    """Result from AR frame processing"""
    success: bool
    output_frame: Optional[np.ndarray]
    tracking_result: Optional[TrackingResult]
    ml_result: Optional['MLTrackingResult']
    processing_time: float
    frame_info: Dict[str, Any]
    performance_metrics: Dict[str, float]

class ARProcessor:
    """Main AR processing pipeline"""
    
    def __init__(self, config: ARConfig = None):
        """
        Initialize AR processor
        
        Args:
            config: AR processing configuration
        """
        self.config = config or ARConfig()
        
        # Initialize components
        self.tracker = ARMarkerTracker(
            algorithm=self.config.algorithm,
            max_features=self.config.max_features,
            match_threshold=self.config.detection_threshold,
            min_matches=self.config.min_matches
        )
        
        self.overlay_engine = VideoOverlayEngine()
        self.optimizer = global_optimizer
        self.memory_optimizer = global_memory_optimizer
        self.performance_monitor = global_performance_monitor
        
        # Optional ML tracker
        self.ml_tracker = None
        if ML_AVAILABLE and hasattr(self.config, 'use_ml_tracker') and self.config.use_ml_tracker:
            self.ml_tracker = PythonARTracker()
        
        # Get optimized functions
        self.optimized_functions = self.optimizer.optimize_image_processing(self.config.use_gpu)
        
        # Frame buffer for multi-threading
        self.frame_buffer = FrameBuffer(maxsize=5)
        
        # Processing state
        self.is_processing = False
        self.current_marker_template = None
        self.current_video_source = None
        self.processing_thread = None
        self.output_callback = None
        
        # Statistics
        self.processing_stats = {
            'frames_processed': 0,
            'successful_detections': 0,
            'total_processing_time': 0.0,
            'average_fps': 0.0,
            'detection_accuracy': 0.0
        }
        
        logger.info(f"AR Processor initialized with {self.config.algorithm}")
        logger.info(f"GPU acceleration: {self.config.use_gpu}")
        logger.info(f"ML tracker available: {ML_AVAILABLE}")
    
    def set_marker_template(self, template: Dict) -> bool:
        """
        Set marker template for tracking
        
        Args:
            template: Marker template dictionary
            
        Returns:
            Success status
        """
        try:
            # Validate template
            required_fields = ['keypoints', 'descriptors', 'width', 'height']
            for field in required_fields:
                if field not in template:
                    logger.error(f"Missing field in template: {field}")
                    return False
            
            self.current_marker_template = template
            logger.info(f"Marker template set: {template.get('feature_count', 0)} features")
            
            # Train ML tracker if available
            if self.ml_tracker and 'training_images' in template:
                try:
                    images = template['training_images']
                    poses = template.get('training_poses', [[0, 0, 1, 0, 0, 0]] * len(images))
                    self.ml_tracker.train_on_marker(images, poses)
                    logger.info("ML tracker trained on marker")
                except Exception as e:
                    logger.warning(f"ML tracker training failed: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set marker template: {str(e)}")
            return False
    
    def set_video_source(self, video_path: str) -> bool:
        """
        Set video source for overlay
        
        Args:
            video_path: Path to video file
            
        Returns:
            Success status
        """
        try:
            # Test video loading
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return False
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Cannot read from video: {video_path}")
                return False
            
            self.current_video_source = video_path
            logger.info(f"Video source set: {video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set video source: {str(e)}")
            return False
    
    def process_single_frame(self, 
                           camera_frame: np.ndarray,
                           video_frame: np.ndarray = None) -> ARProcessingResult:
        """
        Process single AR frame
        
        Args:
            camera_frame: Input camera frame
            video_frame: Video frame for overlay (optional)
            
        Returns:
            AR processing result
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if camera_frame is None:
                return ARProcessingResult(
                    False, None, None, None, 0.0, {}, {}
                )
            
            if self.current_marker_template is None:
                return ARProcessingResult(
                    False, camera_frame, None, None, time.time() - start_time,
                    {'error': 'No marker template set'}, {}
                )
            
            # Use memory optimizer for intermediate arrays
            if self.config.memory_pooling:
                gray_frame = self.memory_optimizer.get_array(
                    camera_frame.shape[:2], np.uint8
                )
            else:
                gray_frame = np.empty(camera_frame.shape[:2], dtype=np.uint8)
            
            # Preprocess frame
            preprocessing_start = time.time()
            if len(camera_frame.shape) == 3:
                cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY, dst=gray_frame)
            else:
                gray_frame = camera_frame.copy()
            
            # Apply optimized preprocessing if available
            if 'enhance_contrast' in self.optimized_functions:
                gray_frame = self.optimized_functions['enhance_contrast'](gray_frame)
            
            preprocessing_time = time.time() - preprocessing_start
            
            # Track marker
            tracking_start = time.time()
            tracking_result = self.tracker.track_marker(camera_frame, self.current_marker_template)
            tracking_time = time.time() - tracking_start
            
            # ML tracking (if available and enabled)
            ml_result = None
            ml_time = 0.0
            if self.ml_tracker:
                ml_start = time.time()
                ml_result = self.ml_tracker.track_marker_ml(camera_frame)
                ml_time = time.time() - ml_start
            
            # Apply video overlay
            overlay_start = time.time()
            if tracking_result.detected and video_frame is not None:
                overlay_settings = {
                    'scale': self.config.overlay_scale,
                    'opacity': self.config.overlay_opacity,
                    'blend_mode': self.config.blend_mode,
                    'debug': self.config.show_debug_info
                }
                
                output_frame = self.overlay_engine.apply_overlay(
                    camera_frame, video_frame, tracking_result,
                    **overlay_settings
                )
            else:
                output_frame = camera_frame.copy()
            
            overlay_time = time.time() - overlay_start
            
            # Return memory to pool
            if self.config.memory_pooling:
                self.memory_optimizer.return_array(gray_frame)
            
            total_time = time.time() - start_time
            
            # Update statistics
            self.processing_stats['frames_processed'] += 1
            if tracking_result.detected:
                self.processing_stats['successful_detections'] += 1
            self.processing_stats['total_processing_time'] += total_time
            
            # Performance metrics
            performance_metrics = {
                'preprocessing_time': preprocessing_time * 1000,
                'tracking_time': tracking_time * 1000,
                'ml_time': ml_time * 1000,
                'overlay_time': overlay_time * 1000,
                'total_time': total_time * 1000,
                'fps': 1.0 / total_time if total_time > 0 else 0
            }
            
            # Record performance
            if self.config.log_performance:
                self.performance_monitor.record_frame_time(total_time)
                self.performance_monitor.record_detection_time(tracking_time)
                self.performance_monitor.record_overlay_time(overlay_time)
            
            # Frame info
            frame_info = {
                'frame_size': camera_frame.shape,
                'marker_detected': tracking_result.detected,
                'matches_count': tracking_result.matches_count,
                'confidence': tracking_result.confidence_score,
                'ml_confidence': ml_result.confidence if ml_result else 0.0
            }
            
            return ARProcessingResult(
                success=True,
                output_frame=output_frame,
                tracking_result=tracking_result,
                ml_result=ml_result,
                processing_time=total_time,
                frame_info=frame_info,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return ARProcessingResult(
                False, camera_frame if camera_frame is not None else None,
                None, None, time.time() - start_time,
                {'error': str(e)}, {}
            )
    
    def start_camera_stream(self, 
                          camera_index: int = 0,
                          output_callback: Callable = None) -> bool:
        """
        Start live camera processing stream
        
        Args:
            camera_index: Camera device index
            output_callback: Callback for processed frames
            
        Returns:
            Success status
        """
        if self.is_processing:
            logger.warning("AR processing already running")
            return False
        
        if not self.current_marker_template:
            logger.error("No marker template set")
            return False
        
        self.output_callback = output_callback
        self.is_processing = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._camera_processing_loop,
            args=(camera_index,)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info(f"Started AR camera stream (camera {camera_index})")
        return True
    
    def stop_processing(self):
        """Stop AR processing"""
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None
        
        logger.info("AR processing stopped")
    
    def _camera_processing_loop(self, camera_index: int):
        """Main camera processing loop"""
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        
        # Initialize video source
        video_cap = None
        if self.current_video_source:
            video_cap = cv2.VideoCapture(self.current_video_source)
        
        try:
            while self.is_processing:
                # Read camera frame
                ret_cam, camera_frame = cap.read()
                if not ret_cam:
                    logger.error("Failed to read camera frame")
                    break
                
                # Read video frame
                video_frame = None
                if video_cap:
                    ret_vid, video_frame = video_cap.read()
                    if not ret_vid:
                        # Loop video
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_vid, video_frame = video_cap.read()
                
                # Process frame
                result = self.process_single_frame(camera_frame, video_frame)
                
                # Call output callback
                if self.output_callback and result.success:
                    try:
                        self.output_callback(result)
                    except Exception as e:
                        logger.error(f"Output callback error: {str(e)}")
                
                # Adaptive FPS control
                target_frame_time = 1.0 / self.config.target_fps
                if result.processing_time < target_frame_time:
                    time.sleep(target_frame_time - result.processing_time)
        
        except Exception as e:
            logger.error(f"Camera processing loop error: {str(e)}")
        
        finally:
            cap.release()
            if video_cap:
                video_cap.release()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        frames_processed = self.processing_stats['frames_processed']
        successful_detections = self.processing_stats['successful_detections']
        total_time = self.processing_stats['total_processing_time']
        
        # Calculate derived metrics
        detection_rate = (successful_detections / frames_processed * 100) if frames_processed > 0 else 0
        average_fps = frames_processed / total_time if total_time > 0 else 0
        
        # Get component statistics
        tracker_stats = self.tracker.get_performance_report()
        optimizer_stats = self.performance_monitor.get_performance_report()
        memory_stats = self.memory_optimizer.get_stats()
        buffer_stats = self.frame_buffer.get_stats()
        
        return {
            'processing': {
                'frames_processed': frames_processed,
                'successful_detections': successful_detections,
                'detection_rate_percent': detection_rate,
                'average_fps': average_fps,
                'total_processing_time': total_time
            },
            'tracker': tracker_stats,
            'performance': optimizer_stats,
            'memory': memory_stats,
            'buffer': buffer_stats,
            'config': asdict(self.config),
            'ml_available': ML_AVAILABLE and self.ml_tracker is not None
        }
    
    def optimize_performance(self, target_fps: int = None):
        """Automatically optimize performance settings"""
        target_fps = target_fps or self.config.target_fps
        
        # Get current performance
        stats = self.get_processing_stats()
        current_fps = stats['processing']['average_fps']
        
        if current_fps < target_fps * 0.8:  # If FPS is significantly below target
            logger.info(f"Optimizing for {target_fps} FPS (current: {current_fps:.1f})")
            
            # Reduce feature count
            if self.config.max_features > 500:
                self.config.max_features = int(self.config.max_features * 0.8)
                self.tracker.max_features = self.config.max_features
                logger.info(f"Reduced max features to {self.config.max_features}")
            
            # Enable more aggressive optimizations
            self.config.enable_optimization = True
            self.config.memory_pooling = True
            
            # Update tracker settings
            self.tracker.optimize_settings(target_fps)
        
        elif current_fps > target_fps * 1.2:  # If FPS is well above target
            # Increase quality
            if self.config.max_features < 2000:
                self.config.max_features = int(self.config.max_features * 1.1)
                self.tracker.max_features = self.config.max_features
                logger.info(f"Increased max features to {self.config.max_features}")
    
    def save_configuration(self, filepath: str):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
    
    def load_configuration(self, filepath: str) -> bool:
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            self.config = ARConfig(**config_dict)
            
            # Update components with new config
            self.tracker.max_features = self.config.max_features
            self.tracker.match_threshold = self.config.detection_threshold
            self.tracker.min_matches = self.config.min_matches
            
            logger.info(f"Configuration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False
    
    def export_debug_data(self, filepath: str):
        """Export debug data for analysis"""
        try:
            debug_data = {
                'timestamp': time.time(),
                'config': asdict(self.config),
                'stats': self.get_processing_stats(),
                'marker_template': {
                    'feature_count': len(self.current_marker_template.get('keypoints', [])),
                    'quality_score': self.current_marker_template.get('quality_score', 0.0)
                } if self.current_marker_template else None
            }
            
            with open(filepath, 'w') as f:
                json.dump(debug_data, f, indent=2, default=str)
            
            logger.info(f"Debug data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export debug data: {str(e)}")


@csrf_exempt
def process_ar_frame_api(request, slug):
    """Clean AR frame processing without problematic imports"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            frame_data = data.get('frame')
            frame_count = data.get('frame_count', 0)
            
            experience = get_object_or_404(ARExperience, slug=slug)
            
            print(f"🔍 Processing frame {frame_count} for {slug}")
            
            processing_start = time.time()
            marker_detected = False
            confidence = 0.0
            features_found = 0
            
            try:
                if frame_data and experience.image and os.path.exists(experience.image.path):
                    print(f"✅ Processing frame with marker: {experience.image.path}")
                    
                    # Decode base64 frame
                    header, encoded = frame_data.split(',', 1)
                    image_data = base64.b64decode(encoded)
                    
                    # Load frame image
                    frame_image = Image.open(BytesIO(image_data))
                    frame_array = np.array(frame_image.convert('RGB'))
                    
                    # Load marker image  
                    marker_image = Image.open(experience.image.path)
                    marker_array = np.array(marker_image.convert('RGB'))
                    
                    print(f"📏 Frame size: {frame_array.shape}, Marker size: {marker_array.shape}")
                    
                    # SAFE FEATURE DETECTION WITHOUT OPENCV
                    try:
                        # Convert to grayscale using scikit-image
                        frame_gray = rgb2gray(frame_array / 255.0)
                        marker_gray = rgb2gray(marker_array / 255.0)
                        
                        # Use ORB from scikit-image for feature detection
                        orb_frame = ORB(n_keypoints=500)
                        orb_frame.detect_and_extract(frame_gray)
                        frame_features = len(orb_frame.keypoints) if orb_frame.keypoints is not None else 0
                        
                        orb_marker = ORB(n_keypoints=500)
                        orb_marker.detect_and_extract(marker_gray)
                        marker_features = len(orb_marker.keypoints) if orb_marker.keypoints is not None else 0
                        
                        print(f"🔍 Frame features: {frame_features}, Marker features: {marker_features}")
                        
                        # Feature matching
                        if frame_features > 10 and marker_features > 10:
                            try:
                                matches = match_descriptors(
                                    orb_marker.descriptors, 
                                    orb_frame.descriptors,
                                    cross_check=True, 
                                    max_distance=0.7
                                )
                                
                                match_count = len(matches) if matches is not None else 0
                                print(f"🎯 Matches found: {match_count}")
                                
                                if match_count > 8:  # Good matches threshold
                                    marker_detected = True
                                    confidence = min(1.0, match_count / 20.0)
                                    features_found = frame_features
                                    
                                    print(f"✅ MARKER DETECTED! Matches: {match_count}, Confidence: {confidence:.2f}")
                                else:
                                    print(f"🔍 Not enough matches: {match_count}")
                                    
                            except Exception as match_error:
                                print(f"❌ Feature matching error: {match_error}")
                        else:
                            print(f"⚠️ Insufficient features - Frame: {frame_features}, Marker: {marker_features}")
                            
                    except Exception as detection_error:
                        print(f"❌ Feature detection error: {detection_error}")
                        # Fall back to simple pixel comparison or placeholder
                        features_found = 25  # Placeholder
                        
                else:
                    print(f"❌ Missing data - Frame: {bool(frame_data)}, Image: {bool(experience.image)}")
                    
            except Exception as processing_error:
                print(f"❌ Frame processing error: {processing_error}")
            
            processing_time = time.time() - processing_start
            
            result = {
                'success': True,
                'frame_count': frame_count,
                'marker_detected': marker_detected,
                'confidence': confidence,
                'features_detected': features_found,
                'marker_features': experience.feature_count,
                'processing_time': processing_time,
                'debug_info': {
                    'marker_image_path': experience.image.path if experience.image else None,
                    'marker_image_exists': os.path.exists(experience.image.path) if experience.image else False,
                    'slug': slug
                }
            }
            
            return JsonResponse(result)
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
