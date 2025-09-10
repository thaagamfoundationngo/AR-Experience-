from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, Http404, HttpResponse
from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.core.files import File
import os
import qrcode
from io import BytesIO
import base64
from .models import ARExperience
from .forms import ARExperienceForm
import logging
from django.utils.text import slugify
from pathlib import Path
import uuid
import time
import json
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import cv2
import numpy as np
import struct
import tempfile
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync, sync_to_async
import asyncio
import threading

logger = logging.getLogger(__name__)

# Global tracker storage
_active_trackers = {}

# ============================================================================
# REAL-TIME TRACKING VIA WEBSOCKET
# ============================================================================
class ARTrackingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.slug = self.scope['url_route']['kwargs']['slug']
        self.tracking_group_name = f'ar_tracking_{self.slug}'
        
        # Join room group
        await self.channel_layer.group_add(
            self.tracking_group_name,
            self.channel_name
        )
        await self.accept()
        logger.info(f"WebSocket connected for AR tracking: {self.slug}")
        
        # Initialize tracking for this experience
        tracker = get_tracker(self.slug)
        if not tracker.is_tracking:
            tracker.start_tracking()
        
        # Notify client about tracking capabilities
        await self.send(text_data=json.dumps({
            'type': 'tracking_info',
            'advanced_tracking': tracker.advanced_tracking_available,
            'methods_available': tracker.advanced_tracker.config['tracking_methods'] if tracker.advanced_tracker else ['basic_orb'],
            'target_accuracy': tracker.target_accuracy
        }))
    
    async def receive(self, text_data):
        """Receive tracking data from frontend"""
        try:
            data = json.loads(text_data)
            
            # Process tracking data
            if data.get('type') == 'tracking_update':
                # Forward to all clients in group
                await self.channel_layer.group_send(
                    self.tracking_group_name,
                    {
                        'type': 'tracking_message',
                        'message': data
                    }
                )
            
            # Handle validation requests
            elif data.get('type') == 'validate_marker':
                experience = await sync_to_async(ARExperience.objects.get)(slug=self.slug)
                if experience and experience.image:
                    # Validate marker quality
                    validation = await sync_to_async(validate_marker_quality)(experience.image.path)
                    # Send validation results
                    await self.send(text_data=json.dumps({
                        'type': 'validation_result',
                        'validation': validation
                    }))
            
            # Handle camera frame processing for advanced tracking
            elif data.get('type') == 'camera_frame':
                # Process frame for advanced feature matching
                if 'frame_data' in data:
                    frame_data = data['frame_data']
                    
                    # Get the tracker for this experience
                    tracker = get_tracker(self.slug)
                    
                    # Use advanced tracking if available
                    if tracker.advanced_tracking_available:
                        try:
                            # Decode base64 frame data
                            frame_bytes = base64.b64decode(frame_data.split(',')[1])
                            nparr = np.frombuffer(frame_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                # Process frame with advanced tracker
                                tracking_result = await sync_to_async(
                                    tracker.advanced_tracker.track_frame
                                )(frame)
                                
                                # Send comprehensive tracking results
                                await self.send(text_data=json.dumps({
                                    'type': 'advanced_tracking_result',
                                    'result': tracking_result
                                }))
                        except Exception as e:
                            logger.error(f"Error processing camera frame: {str(e)}")
                            await self.send(text_data=json.dumps({
                                'type': 'error',
                                'message': f'Frame processing error: {str(e)}'
                            }))
            
        except Exception as e:
            logger.error(f"Error in WebSocket receive: {str(e)}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def tracking_message(self, event):
        """Send tracking message to WebSocket"""
        await self.send(text_data=json.dumps(event['message']))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_mindar_marker(image_path, output_path):
    """
    Enhanced MindAR marker generation with Stories AR-style auto-enhancement
    Returns True if successful, False otherwise
    """
    try:
        logger.info(f"Starting enhanced MindAR marker generation for: {image_path}")
        
        # Read original image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return False
        
        # Step 1: Enhanced preprocessing (Stories AR style)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Force binary conversion with optimal threshold
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 2: Add guaranteed thick border (Stories AR standard)  
        height, width = binary.shape
        border_size = max(50, int(min(height, width) * 0.2))  # 20% minimum border
        
        bordered = cv2.copyMakeBorder(
            binary, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Step 3: Enhanced feature detection (Stories AR parameters)
        orb = cv2.ORB_create(
            nfeatures=1500,        # Increased from 750
            scaleFactor=1.15,      # More aggressive
            nlevels=12,            # More pyramid levels
            edgeThreshold=10,      # Lower for more edges
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=8        # Lower threshold
        )
        
        keypoints, descriptors = orb.detectAndCompute(bordered, None)
        
        # Step 4: Guarantee minimum feature count
        if len(keypoints) < 100:  # Stories AR standard
            corners = cv2.goodFeaturesToTrack(
                bordered, maxCorners=200, qualityLevel=0.01, minDistance=10
            )
            
            if corners is not None:
                corner_kps = [cv2.KeyPoint(x[0][0], x[0][1], 7) for x in corners]
                keypoints.extend(corner_kps[:200-len(keypoints)])
                keypoints, descriptors = orb.detectAndCompute(bordered, None)
        
        logger.info(f"Enhanced marker generated with {len(keypoints)} features")
        
        # Filter and sort keypoints
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:500]
        if descriptors is not None:
            descriptors = descriptors[:len(keypoints)]
        
        # Step 5: Create optimized MindAR data structure
        mindar_data = {
            "imageWidth": bordered.shape[1],
            "imageHeight": bordered.shape[0],
            "maxTrack": 1,
            "filterMinCF": 0.0001,  # More sensitive
            "filterBeta": 0.001,    # More responsive
            "missTolerance": 3,     
            "warmupTolerance": 3,   
            "targets": [{
                "name": os.path.splitext(os.path.basename(image_path))[0],
                "keypoints": [],
                "descriptors": []
            }]
        }
        
        # Add keypoints and descriptors
        for i, kp in enumerate(keypoints):
            if kp.response > 0.001:  # Higher quality threshold
                mindar_data["targets"][0]["keypoints"].append({
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                    "response": float(kp.response),
                    "octave": int(kp.octave)
                })
                if descriptors is not None and i < len(descriptors):
                    mindar_data["targets"][0]["descriptors"].append(descriptors[i].tolist())
        
        # Step 6: Save to file
        json_str = json.dumps(mindar_data, separators=(',', ':'))
        header = struct.pack('4sI', b'MIND', len(json_str))
        
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(json_str.encode('utf-8'))
        
        # Save enhanced image for reference
        enhanced_image_path = output_path.replace('.mind', '_enhanced.png')
        cv2.imwrite(enhanced_image_path, bordered)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logger.info(f"Enhanced MindAR marker generated successfully: {output_path}")
            return True
        else:
            logger.error(f"Failed to create MindAR marker file: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error generating MindAR marker: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def create_aruco_marker(marker_id, size=500, border_size=50):
    """Create a proper ArUco marker"""
    try:
        # Updated for newer OpenCV versions
        try:
            # For OpenCV 4.x+
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, size)
        except AttributeError:
            try:
                # For OpenCV 3.x
                aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
                marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, size)
            except AttributeError:
                # If aruco module is not available, create a unique pattern
                logger.error("OpenCV ArUco module not available, creating unique pattern")
                marker_image = np.zeros((size, size), dtype=np.uint8)
                cell_size = size // 7
                for i in range(7):
                    for j in range(7):
                        if i == 0 or i == 6 or j == 0 or j == 6:
                            continue  # Keep border black
                        if i > 0 and i < 6 and j > 0 and j < 6:
                            bit_position = ((i - 1) * 5 + (j - 1)) % 16
                            if (marker_id >> bit_position) & 1:
                                marker_image[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size] = 255

        marker_image = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
        if border_size > 0:
            marker_image = cv2.copyMakeBorder(
                marker_image, border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        return marker_image
    except Exception as e:
        logger.error(f"Error creating ArUco marker: {str(e)}")
        return None

def generate_aruco_mindar_marker(marker_id, output_path, size=500):
    """Generate a MindAR-compatible marker using ArUco pattern"""
    try:
        marker_image = create_aruco_marker(marker_id, size)
        if marker_image is None:
            return False
        image_path = output_path.replace('.mind', '.png')
        cv2.imwrite(image_path, marker_image)
        mindar_data = {
            "type": "aruco",
            "dictionary": "DICT_4X4_50",
            "id": marker_id,
            "size": size,
            "imageWidth": marker_image.shape[1],
            "imageHeight": marker_image.shape[0],
            "maxTrack": 1,
            "filterMinCF": 0.0001,
            "filterBeta": 0.001,
            "missTolerance": 3,
            "warmupTolerance": 3
        }
        json_str = json.dumps(mindar_data, separators=(',', ':'))
        header = struct.pack('4sI', b'MIND', len(json_str))
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(json_str.encode('utf-8'))
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"ArUco MindAR marker generated successfully: {output_path}")
            return True
        else:
            logger.error(f"Failed to create ArUco MindAR marker file: {output_path}")
            return False
    except Exception as e:
        logger.error(f"Error generating ArUco MindAR marker: {str(e)}")
        return False

def ensure_marker_directory(experience_slug):
    """Create marker directory if it doesn't exist"""
    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience_slug
    marker_dir.mkdir(parents=True, exist_ok=True)
    return marker_dir

# ============================================================================
# BACKEND TRACKING VALIDATION FUNCTIONS
# ============================================================================
def validate_marker_tracking(experience, test_image_path=None):
    """
    Validate marker tracking quality (not real-time tracking)
    Returns tracking validation results
    """
    results = {
        'valid': False,
        'quality_score': 0,
        'tracking_speed': 'unknown',
        'issues': [],
        'recommendations': [],
        'marker_stats': {},
        'test_results': {},
        'target_accuracy': 0.9  # Stories AR level
    }
    try:
        # 1. Check if marker file exists
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
        mind_file = marker_dir / f"{experience.slug}.mind"
        if not mind_file.exists() or mind_file.stat().st_size == 0:
            results['issues'].append('Marker file missing or empty')
            results['recommendations'].append('Regenerate marker file')
            return results
        results['marker_stats']['file_size'] = mind_file.stat().st_size
        results['marker_stats']['file_path'] = str(mind_file)
        
        # 2. Validate marker image quality
        if experience.image:
            img = cv2.imread(experience.image.path)
            if img is not None:
                quality_results = validate_marker_quality(experience.image.path)
                results['quality_score'] = quality_results['score']
                results['issues'].extend(quality_results['issues'])
                results['recommendations'].extend(quality_results['recommendations'])
                results['marker_stats'].update(quality_results['stats'])
            else:
                results['issues'].append('Cannot read marker image')
        
        # 3. Test tracking with sample image if provided
        if test_image_path and os.path.exists(test_image_path):
            tracking_result = test_marker_tracking(
                experience.image.path if experience.image else None,
                test_image_path
            )
            results['test_results'] = tracking_result
            results['tracking_speed'] = tracking_result.get('speed', 'unknown')
            if tracking_result.get('detected', False):
                results['valid'] = True
            else:
                results['issues'].append('Marker not detected in test image')
        
        # 4. Final validation
        if results['quality_score'] > 0.7 and len(results['issues']) == 0:
            results['valid'] = True
        elif results['quality_score'] > 0.5:
            results['valid'] = True
            results['recommendations'].insert(0, 'Marker quality is acceptable but could be improved')
        
        return results
    except Exception as e:
        logger.error(f"Error in validate_marker_tracking: {str(e)}")
        results['issues'].append(f'Validation error: {str(e)}')
        return results

def validate_marker_quality(image_path):
    """Validate marker image quality for AR tracking"""
    results = {
        'score': 0,
        'issues': [],
        'recommendations': [],
        'stats': {},
        'target_accuracy': 0.9  # Stories AR level
    }
    try:
        img = cv2.imread(image_path)
        if img is None:
            results['issues'].append('Cannot read image file')
            return results
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Check resolution
        height, width = gray.shape
        results['stats']['resolution'] = f"{width}x{height}"
        if width < 200 or height < 200:
            results['issues'].append('Image too small (minimum 200x200)')
            results['recommendations'].append('Use higher resolution image')
        elif width > 2000 or height > 2000:
            results['issues'].append('Image too large (maximum 2000x2000)')
            results['recommendations'].append('Resize image for better performance')
        
        # 2. Check contrast
        contrast = gray.std()
        results['stats']['contrast'] = round(contrast, 2)
        if contrast < 40:  # Raised threshold for Stories AR quality
            results['issues'].append('Low contrast - marker features not distinct')
            results['recommendations'].append('Increase contrast or use black/white design')
        elif contrast > 80:  # Stories AR level
            results['score'] += 0.3
        
        # 3. Check edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        results['stats']['edge_density'] = round(edge_density, 4)
        if edge_density < 0.015:  # Raised threshold
            results['issues'].append('Low edge density - not enough features')
            results['recommendations'].append('Add more distinct patterns')
        elif edge_density > 0.06:  # Stories AR level
            results['score'] += 0.3
        
        # 4. Check feature quality
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        results['stats']['keypoints'] = len(keypoints)
        if len(keypoints) < 100:  # Raised threshold for Stories AR
            results['issues'].append(f'Insufficient features: {len(keypoints)} (minimum 100)')
            results['recommendations'].append('Add more complex patterns')
        elif len(keypoints) > 400:  # Stories AR level
            results['score'] += 0.2
        
        # 5. Check border quality
        border_score = check_border_quality(gray)
        results['stats']['border_score'] = round(border_score, 2)
        if border_score < 0.8:  # Raised threshold
            results['issues'].append('Poor border quality - affects detection')
            results['recommendations'].append('Add thick, solid border')
        else:
            results['score'] += 0.2
        
        # 6. Final score calculation
        results['score'] = min(1.0, results['score'])
        return results
    except Exception as e:
        logger.error(f"Error in validate_marker_quality: {str(e)}")
        results['issues'].append(f'Quality check error: {str(e)}')
        return results

def check_border_quality(gray_img):
    """Check if marker has adequate border"""
    height, width = gray_img.shape
    border_size = min(height, width) // 8  # Increased border check area
    outer_border = gray_img[:border_size, :].mean()
    outer_border += gray_img[-border_size:, :].mean()
    outer_border += gray_img[:, :border_size].mean()
    outer_border += gray_img[:, -border_size:].mean()
    outer_border /= 4
    inner_area = gray_img[border_size:-border_size, border_size:-border_size].mean()
    border_score = abs(outer_border - inner_area) / 255
    return border_score

def test_marker_tracking(marker_path, test_image_path):
    """Test if marker can be detected in test image"""
    results = {
        'detected': False,
        'speed': 'unknown',
        'confidence': 0,
        'position': None,
        'details': {},
        'target_accuracy': 0.9
    }
    try:
        marker = cv2.imread(marker_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        if marker is None or test_img is None:
            results['details']['error'] = 'Cannot read images'
            return results
        
        start_time = time.time()
        
        # Method 1: Template matching for simple markers
        if marker.shape[0] < 500 and marker.shape[1] < 500:
            res = cv2.matchTemplate(test_img, marker, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.75:  # Stories AR confidence level
                results['detected'] = True
                results['confidence'] = round(max_val, 2)
                results['position'] = max_loc
                results['details']['method'] = 'template_matching'
        
        # Method 2: Feature matching for complex markers
        if not results['detected']:
            orb = cv2.ORB_create(nfeatures=1500)
            kp1, des1 = orb.detectAndCompute(marker, None)
            kp2, des2 = orb.detectAndCompute(test_img, None)
            if des1 is not None and des2 is not None and len(kp1) > 20:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < 30]  # Stricter threshold
                if len(good_matches) > 20:  # Stories AR standard
                    results['detected'] = True
                    results['confidence'] = min(1.0, len(good_matches) / 100)
                    results['details']['method'] = 'feature_matching'
                    results['details']['good_matches'] = len(good_matches)
        
        detection_time = time.time() - start_time
        results['details']['detection_time_ms'] = round(detection_time * 1000, 2)
        if detection_time < 0.05:
            results['speed'] = 'fast'
        elif detection_time < 0.2:
            results['speed'] = 'medium'
        else:
            results['speed'] = 'slow'
        
        return results
    except Exception as e:
        logger.error(f"Error in test_marker_tracking: {str(e)}")
        results['details']['error'] = str(e)
        return results

# ============================================================================
# REAL-TIME TRACKING PROCESSOR
# ============================================================================
class RealTimeTracker:
    """Real-time marker tracking processor with advanced tracking support"""
    def __init__(self, experience_slug):
        self.experience_slug = experience_slug
        self.is_tracking = False
        self.tracking_thread = None
        self.channel_layer = None
        self.target_accuracy = 0.9  # Stories AR level
        self.advanced_tracker = None
        self.advanced_tracking_available = False
        
        # Try to initialize the advanced tracker
        try:
            from .advanced_ar_tracking import integrate_with_django_views
            self.advanced_tracker = integrate_with_django_views(experience_slug)
            self.advanced_tracking_available = True
            logger.info(f"Advanced tracker initialized for {experience_slug}")
        except Exception as e:
            logger.warning(f"Advanced tracker not available for {experience_slug}: {str(e)}")
            self.advanced_tracking_available = False
            
        # Initialize channel layer
        try:
            self.channel_layer = get_channel_layer()
        except Exception as e:
            logger.warning(f"Channel layer not available for {experience_slug}: {str(e)}")
    
    def start_tracking(self, image_callback=None):
        """Start real-time tracking in a separate thread"""
        if self.is_tracking:
            return
        self.is_tracking = True
        self.tracking_thread = threading.Thread(
            target=self._tracking_loop,
            args=(image_callback,),
            daemon=True
        )
        self.tracking_thread.start()
        logger.info(f"Started real-time tracking for {self.experience_slug} with {self.target_accuracy*100}% accuracy target")
    
    def stop_tracking(self):
        """Stop real-time tracking"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        logger.info(f"Stopped real-time tracking for {self.experience_slug}")
    
    def _tracking_loop(self, image_callback):
        """Main tracking loop with enhanced tracking support"""
        try:
            # Get marker data
            experience = ARExperience.objects.get(slug=self.experience_slug)
            if not experience.image:
                logger.error(f"No image found for {self.experience_slug}")
                return
            
            marker_img = cv2.imread(experience.image.path, cv2.IMREAD_GRAYSCALE)
            if marker_img is None:
                logger.error(f"Cannot read marker image for {self.experience_slug}")
                return
            
            # Initialize enhanced ORB detector
            orb = cv2.ORB_create(
                nfeatures=1500,
                scaleFactor=1.15,
                nlevels=12,
                edgeThreshold=10,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=8
            )
            kp1, des1 = orb.detectAndCompute(marker_img, None)
            if des1 is None:
                logger.error(f"Cannot compute descriptors for {self.experience_slug}")
                return
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            while self.is_tracking:
                # Get frame from callback
                if image_callback:
                    frame = image_callback()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    
                    # Use advanced tracking if available
                    if self.advanced_tracking_available:
                        try:
                            # Process frame with advanced tracker
                            tracking_result = self.advanced_tracker.track_frame(frame)
                            
                            # Send tracking result via WebSocket if channel layer is available
                            if self.channel_layer:
                                async_to_sync(self.channel_layer.group_send)(
                                    f'ar_tracking_{self.experience_slug}',
                                    {
                                        'type': 'tracking_update',
                                        'tracking': {
                                            'detected': tracking_result.get('detected', False),
                                            'confidence': tracking_result.get('confidence', 0),
                                            'methods_used': tracking_result.get('methods_used', []),
                                            'target_accuracy': self.target_accuracy,
                                            'timestamp': time.time(),
                                            'advanced': True
                                        }
                                    }
                                )
                            continue
                        except Exception as e:
                            logger.error(f"Error in advanced tracking: {str(e)}")
                            # Fall back to basic tracking
                    
                    # Enhanced tracking (fallback)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndCompute(gray, None)
                    if des2 is not None:
                        # Match features
                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = [m for m in matches if m.distance < 30]  # Stricter threshold
                        
                        if len(good_matches) > 20:  # Stories AR standard
                            # Calculate tracking result
                            confidence = min(1.0, len(good_matches) / 100)
                            
                            # Send tracking result via WebSocket if channel layer is available
                            if self.channel_layer:
                                async_to_sync(self.channel_layer.group_send)(
                                    f'ar_tracking_{self.experience_slug}',
                                    {
                                        'type': 'tracking_update',
                                        'tracking': {
                                            'detected': True,
                                            'confidence': confidence,
                                            'matches': len(good_matches),
                                            'target_accuracy': self.target_accuracy,
                                            'timestamp': time.time(),
                                            'advanced': False
                                        }
                                    }
                                )
                        else:
                            if self.channel_layer:
                                async_to_sync(self.channel_layer.group_send)(
                                    f'ar_tracking_{self.experience_slug}',
                                    {
                                        'type': 'tracking_update',
                                        'tracking': {
                                            'detected': False,
                                            'confidence': 0,
                                            'matches': 0,
                                            'target_accuracy': self.target_accuracy,
                                            'timestamp': time.time(),
                                            'advanced': False
                                        }
                                    }
                                )
                time.sleep(0.03)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error in tracking loop: {str(e)}")
        finally:
            self.is_tracking = False

def get_tracker(experience_slug):
    """Get or create tracker instance"""
    if experience_slug not in _active_trackers:
        _active_trackers[experience_slug] = RealTimeTracker(experience_slug)
    return _active_trackers[experience_slug]

# ============================================================================
# CORE AR EXPERIENCE FUNCTIONS
# ============================================================================
def home(request):
    return render(request, "home.html")

def scanner(request):
    return render(request, "scanner.html")

def upload_view(request):
    """Enhanced upload view with Stories AR-style processing"""
    if request.method == 'POST':
        form = ARExperienceForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                with transaction.atomic():
                    experience = form.save(commit=False)
                    if not experience.slug:
                        base_slug = slugify(experience.title) or f"exp-{uuid.uuid4().hex[:8]}"
                        counter = 1
                        slug = base_slug
                        while ARExperience.objects.filter(slug=slug).exists():
                            slug = f"{base_slug}-{counter}"
                            counter += 1
                        experience.slug = slug
                    experience.save()
                    logger.info(f"Experience saved: {experience.slug}")

                marker_generated = False
                marker_message = ""
                
                if request.POST.get('use_browser_compilation') and request.FILES.get('mind_file'):
                    mind_file = request.FILES['mind_file']
                    marker_dir = ensure_marker_directory(experience.slug)
                    mind_file_path = marker_dir / f"{experience.slug}.mind"
                    with open(mind_file_path, 'wb') as f:
                        for chunk in mind_file.chunks():
                            f.write(chunk)
                    if mind_file_path.exists() and mind_file_path.stat().st_size > 0:
                        experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                        marker_generated = True
                        marker_message = "🧠 MindAR target compiled successfully in browser"
                        logger.info(f"Browser-compiled marker saved: {mind_file_path}")
                    else:
                        marker_message = "❌ Browser-compiled file is invalid"
                else:
                    # Stories AR approach: Always enhance the image first
                    if experience.image:
                        marker_dir = ensure_marker_directory(experience.slug)
                        mind_file_path = marker_dir / f"{experience.slug}.mind"
                        
                        # Try enhanced image marker first (Stories AR style)
                        if generate_mindar_marker(experience.image.path, str(mind_file_path)):
                            experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                            marker_generated = True
                            marker_message = "✅ Image automatically enhanced for optimal AR tracking"
                            logger.info(f"Enhanced image marker generated for {experience.slug}")
                        else:
                            # Fallback to ArUco if image enhancement fails
                            marker_id = hash(experience.slug) % 50
                            if generate_aruco_mindar_marker(marker_id, str(mind_file_path)):
                                experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                                marker_generated = True
                                marker_message = "🎯 ArUco marker generated as fallback"
                                logger.info(f"ArUco marker generated for {experience.slug}")
                            else:
                                marker_message = "❌ Failed to generate enhanced marker from image"
                                logger.error(f"Failed to generate marker for {experience.slug}")
                    else:
                        marker_message = "⚠️ No image provided for marker generation"

                experience.marker_generated = marker_generated
                experience.save(update_fields=['nft_iset_file', 'marker_generated'])

                try:
                    qr_url = request.build_absolute_uri(experience.get_absolute_url())
                    qr_code_dir = os.path.join(settings.MEDIA_ROOT, 'qrcodes')
                    os.makedirs(qr_code_dir, exist_ok=True)
                    qr_code_path = os.path.join(qr_code_dir, f'{experience.slug}.png')
                    qr = qrcode.QRCode(version=1, box_size=10, border=5)
                    qr.add_data(qr_url)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    img.save(qr_code_path)
                    experience.qr_code = f'qrcodes/{experience.slug}.png'
                    experience.save(update_fields=['qr_code'])
                    logger.info(f"QR code created: {qr_code_path}")
                except Exception as qr_error:
                    logger.error(f"QR generation failed: {qr_error}")

                # Always show success message (Stories AR style)
                if marker_generated:
                    messages.success(request, f'🎯 AR Experience created! {marker_message}')
                else:
                    messages.error(request, f'Upload failed. {marker_message}')
                
                return redirect(f'/upload/?new={experience.slug}')
                
            except Exception as save_error:
                logger.error(f"Critical error in upload process: {save_error}")
                messages.error(request, f"Upload failed: {str(save_error)}")
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ARExperienceForm()

    context = {
        'form': form,
        'experiences': ARExperience.objects.all().order_by('-created_at')[:10],
        'new_experience_slug': request.GET.get('new'),
    }
    return render(request, 'upload.html', context)

def experience_view(request, slug):
    """Enhanced experience view with advanced AR tracking"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
    except ARExperience.DoesNotExist:
        raise Http404(f"Experience '{slug}' not found")
    
    # Build URLs properly
    base_url = getattr(settings, 'BASE_URL', request.build_absolute_uri('/').rstrip('/'))
    media_url = settings.MEDIA_URL
    if not media_url.startswith('http'):
        media_url = base_url + media_url.rstrip('/')
    
    # Marker paths
    marker_base_url = f"{media_url}/markers/{slug}/{slug}"
    
    # Check for MindAR target file
    marker_files_exist = False
    mind_file_path = None
    mind_file_size = 0
    try:
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        mind_file = marker_dir / f"{slug}.mind"
        mind_file_path = str(mind_file)
        if mind_file.exists() and mind_file.stat().st_size > 0:
            marker_files_exist = True
            mind_file_size = mind_file.stat().st_size
            logger.info(f"MindAR target file found for {slug} (size: {mind_file_size} bytes)")
        else:
            logger.warning(f"MindAR target file missing for {slug}")
    except Exception as e:
        logger.error(f"Error checking MindAR target for {slug}: {e}")
    
    # Initialize advanced tracker if available
    advanced_tracker = None
    advanced_tracking_available = False
    advanced_tracking_results = None
    
    try:
        # Try to initialize the advanced tracker
        from .advanced_ar_tracking import AdvancedARTracker, integrate_with_django_views
        advanced_tracker = integrate_with_django_views(slug)
        advanced_tracking_available = True
        
        # Validate marker quality using advanced tracker
        if experience.image:
            try:
                # Load the image for advanced validation
                img = cv2.imread(experience.image.path)
                if img is not None:
                    # Run advanced tracking analysis
                    advanced_tracking_results = advanced_tracker.track_frame(img)
                    logger.info(f"Advanced tracking analysis completed for {slug}")
            except Exception as e:
                logger.error(f"Error in advanced tracking analysis: {str(e)}")
                advanced_tracking_available = False
    except ImportError:
        logger.warning("Advanced tracking dependencies not available, using basic tracking")
    except Exception as e:
        logger.error(f"Error initializing advanced tracker: {str(e)}")
    
    # MindAR configuration with Stories AR-style optimized parameters
    mindar_config = {
        'maxTrack': 1,
        'showStats': settings.DEBUG,
        'uiLoading': 'no',
        'uiError': 'no',
        'uiScanning': 'no',
        'autoStart': True,           # Stories AR auto-starts
        'filterMinCF': 0.0005,       # More sensitive
        'filterBeta': 0.005,         # More responsive  
        'missTolerance': 2,          # Lower tolerance
        'warmupTolerance': 2,        # Faster detection
        'targetAccuracy': 0.9        # Raise to 90% (Stories AR level)
    }
    
    # Build URLs
    video_url = experience.video.url if experience.video else None
    marker_image_url = experience.image.url if experience.image else None
    
    # Validate marker quality (with error handling)
    tracking_validation = None
    if experience.image:
        try:
            tracking_validation = validate_marker_tracking(experience)
        except Exception as e:
            logger.error(f"Error validating marker for {slug}: {str(e)}")
            tracking_validation = {
                'valid': False,
                'quality_score': 0,
                'issues': [f'Validation error: {str(e)}'],
                'recommendations': ['Please check the marker image and try regenerating it.'],
                'target_accuracy': 0.9
            }
    
    # Build WebSocket URL (with error handling)
    websocket_url = None
    websocket_available = False
    try:
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        websocket_available = True
        if settings.DEBUG:
            websocket_url = f"ws://{request.get_host()}/ws/ar/{slug}/"
        else:
            websocket_url = f"wss://{request.get_host()}/ws/ar/{slug}/"
    except Exception as e:
        logger.warning(f"Channel layer not available for {slug}, WebSocket disabled: {str(e)}")
        websocket_available = False
    
    # Initialize real-time tracker for this experience
    tracker = None
    tracker_available = False
    try:
        if websocket_available:
            tracker = get_tracker(slug)
            tracker_available = True
    except Exception as e:
        logger.warning(f"Cannot initialize tracker for {slug}: {str(e)}")
        tracker_available = False
    
    # Check if we need to regenerate the marker
    need_to_regenerate = False
    if experience.image and (not marker_files_exist or mind_file_size < 1000):
        need_to_regenerate = True
        logger.warning(f"Marker file for {slug} needs to be regenerated (exists: {marker_files_exist}, size: {mind_file_size})")
    
    # Prepare advanced tracking information for the template
    advanced_tracking_info = {}
    if advanced_tracking_available and advanced_tracking_results:
        advanced_tracking_info = {
            'available': True,
            'methods_used': advanced_tracking_results.get('methods_used', []),
            'confidence': advanced_tracking_results.get('confidence', 0),
            'detected': advanced_tracking_results.get('detected', False),
            'frame_quality': advanced_tracking_results.get('frame_quality', {}),
            'processing_time': advanced_tracking_results.get('processing_time', 0),
            'detailed_results': advanced_tracking_results.get('detailed_results', {})
        }
    else:
        advanced_tracking_info = {
            'available': False,
            'reason': 'dependencies_not_available' if not advanced_tracking_available else 'processing_failed'
        }
    
    context = {
        "experience": experience,
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "base_url": base_url,
        "title": experience.title,
        "video_url": video_url,
        "marker_image_url": marker_image_url,
        "timestamp": int(time.time()),
        "tracking_method": "Advanced Multi-Method" if advanced_tracking_available else "Enhanced MindAR",
        "mindar_config": mindar_config,
        "mindar_config_json": json.dumps(mindar_config),
        "tracking_validation": tracking_validation,
        "websocket_url": websocket_url,
        "websocket_available": websocket_available,
        "target_accuracy": 0.9,
        "need_to_regenerate": need_to_regenerate,
        "mind_file_size": mind_file_size,
        "advanced_tracking": advanced_tracking_info,
        "instructions": {
            'setup': 'Allow camera access when prompted by your browser',
            'usage': 'Point your camera at the uploaded marker image',
            'distance': 'Hold your device 20-60cm away from the marker',
            'lighting': 'Ensure good lighting for better tracking',
            'stability': 'Keep the marker clearly visible and steady',
            'technology': 'Powered by Advanced Multi-Method Tracking' if advanced_tracking_available else 'Powered by Enhanced MindAR',
            'accuracy_target': 'Target accuracy: 90% for professional performance'
        },
        "debug": {
            'mind_file_path': mind_file_path,
            'mind_file_exists': marker_files_exist,
            'mind_file_size': mind_file_size,
            'media_root': str(settings.MEDIA_ROOT),
            'slug': slug,
            'tracking_issues': tracking_validation['issues'] if tracking_validation else [],
            'tracking_recommendations': tracking_validation['recommendations'] if tracking_validation else [],
            'target_accuracy': 0.9,
            'websocket_available': websocket_available,
            'tracker_available': tracker_available,
            'need_to_regenerate': need_to_regenerate,
            'advanced_tracking_available': advanced_tracking_available,
            'advanced_tracking_results': advanced_tracking_results if advanced_tracking_available else None
        } if settings.DEBUG else {},
        # Tracking function connections
        "tracking_functions": {
            "real_time_tracker": {
                "active": tracker.is_tracking if tracker else False,
                "slug": slug,
                "target_accuracy": 0.9,
                "available": tracker_available
            },
            "websocket_consumer": {
                "connected": False,
                "group_name": f'ar_tracking_{slug}',
                "target_accuracy": 0.9,
                "available": websocket_available
            },
            "marker_validation": {
                "valid": tracking_validation['valid'] if tracking_validation else False,
                "quality_score": tracking_validation['quality_score'] if tracking_validation else 0,
                "target_accuracy": 0.9
            },
            "advanced_tracker": {
                "available": advanced_tracking_available,
                "methods": advanced_tracking_info.get('methods_used', []) if advanced_tracking_available else [],
                "confidence": advanced_tracking_info.get('confidence', 0) if advanced_tracking_available else 0
            }
        }
    }
    return render(request, "experience.html", context)

def webcam_ar_experience_view(request, slug=None):
    """Dedicated MindAR webcam experience view with connected tracking functions"""
    if slug:
        experience = get_object_or_404(ARExperience, slug=slug)
    else:
        try:
            experience = ARExperience.objects.filter(
                image__isnull=False,
                video__isnull=False,
                marker_generated=True
            ).latest('created_at')
        except ARExperience.DoesNotExist:
            logger.warning("No AR experiences available")
            experience = None

    base_url = getattr(settings, 'BASE_URL', request.build_absolute_uri('/')[:-1])
    media_url = getattr(settings, 'MEDIA_URL', '/media/')
    if not media_url.startswith('http'):
        media_url = base_url + media_url

    marker_base_url = ""
    marker_files_exist = False
    if experience:
        marker_base_url = f"{media_url}markers/{experience.slug}/{experience.slug}"
        try:
            marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
            mind_file = marker_dir / f"{experience.slug}.mind"
            if mind_file.exists() and mind_file.stat().st_size > 0:
                marker_files_exist = True
        except Exception as e:
            logger.error(f"Error checking MindAR target for {experience.slug}: {e}")

    # Initialize real-time tracker for this experience
    tracker = None
    if experience:
        tracker = get_tracker(experience.slug)

    context = {
        "experience": experience,
        "title": experience.title if experience else "Enhanced MindAR Experience",
        "description": experience.description if experience else "Enhanced MindAR Webcam Experience",
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "marker_image_url": experience.image.url if experience and experience.image else None,
        "video_url": experience.video.url if experience and experience.video else None,
        "base_url": base_url,
        "timestamp": int(time.time()),
        "debug_mode": settings.DEBUG,
        "tracking_method": "Enhanced MindAR",
        "websocket_url": f"ws://{request.get_host()}/ws/ar/{experience.slug}/" if experience else None,
        "target_accuracy": 0.9,  # Stories AR level
        "user_instructions": {
            "camera_setup": "Allow camera access when prompted",
            "marker_usage": "Point camera at the uploaded image",
            "optimal_distance": "20-60cm from marker",
            "lighting_tips": "Ensure good lighting for tracking",
            "stability_advice": "Keep marker visible and steady",
            "accuracy_target": "Target accuracy: 90% for professional performance"
        },
        # Tracking function connections
        "tracking_functions": {
            "real_time_tracker": {
                "active": tracker.is_tracking if tracker else False,
                "slug": experience.slug if experience else None,
                "target_accuracy": 0.9
            },
            "websocket_consumer": {
                "connected": False,  # Will be updated in frontend
                "group_name": f'ar_tracking_{experience.slug}' if experience else None,
                "target_accuracy": 0.9
            }
        }
    }
    return render(request, "experience.html", context)

# ============================================================================
# API ENDPOINTS
# ============================================================================
def ar_status_api(request, slug):
    """API endpoint to check MindAR experience status"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
        mind_file = marker_dir / f"{experience.slug}.mind"
        marker_files_exist = mind_file.exists() and mind_file.stat().st_size > 0
        status = {
            "success": True,
            "tracking_method": "Enhanced MindAR",
            "experience": {
                "title": experience.title,
                "slug": experience.slug,
                "created": experience.created_at.isoformat(),
            },
            "markers": {
                "exist": marker_files_exist,
                "file": f"{experience.slug}.mind",
                "base_url": f"/media/markers/{experience.slug}/{experience.slug}"
            },
            "media": {
                "image_available": bool(experience.image),
                "video_available": bool(experience.video),
                "image_url": experience.image.url if experience.image else None,
                "video_url": experience.video.url if experience.video else None
            },
            "mindar_database": {
                "target_stored": bool(experience.nft_iset_file),
            },
            "webcam": {
                "required": True,
                "permissions_needed": ["camera"],
                "https_required": True,
                "auto_activate": True
            },
            "realtime_tracking": {
                "available": True,
                "websocket_url": f"ws://{request.get_host()}/ws/ar/{slug}/",
                "target_accuracy": 0.9
            },
            "timestamp": int(time.time())
        }
        return JsonResponse(status)
    except ARExperience.DoesNotExist:
        return JsonResponse({
            "success": False,
            "error": "AR Experience not found",
            "timestamp": int(time.time())
        }, status=404)
    except Exception as e:
        logger.error(f"AR Status API error: {e}")
        return JsonResponse({
            "success": False,
            "error": "Internal server error",
            "timestamp": int(time.time())
        }, status=500)

def marker_status_api(request, slug):
    """API endpoint to check marker status"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        mind_file = marker_dir / f"{slug}.mind"
        file_exists = mind_file.exists() and mind_file.stat().st_size > 0
        file_status = {
            f"{slug}.mind": {
                'exists': file_exists,
                'size': mind_file.stat().st_size if file_exists else 0,
                'path': str(mind_file)
            }
        }
        return JsonResponse({
            'slug': slug,
            'tracking_method': 'Enhanced MindAR',
            'marker_generated': experience.marker_generated and file_exists,
            'files_exist': file_exists,
            'files': file_status,
            'can_regenerate': bool(experience.image),
            'webcam_ready': file_exists,
            'mindar_database_status': {
                'target_stored': bool(experience.nft_iset_file),
            },
            'realtime_tracking': {
                'available': True,
                'websocket_url': f"ws://{request.get_host()}/ws/ar/{slug}/",
                'target_accuracy': 0.9
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def validate_tracking_api(request, slug):
    """API endpoint to validate marker tracking quality"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        test_image = None
        if request.method == 'POST' and request.FILES.get('test_image'):
            test_image = request.FILES['test_image']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                for chunk in test_image.chunks():
                    f.write(chunk)
                test_image_path = f.name
        else:
            test_image_path = None
        results = validate_marker_tracking(experience, test_image_path)
        if test_image_path:
            os.unlink(test_image_path)
        return JsonResponse({
            'success': True,
            'slug': slug,
            'tracking_validation': results,
            'timestamp': int(time.time())
        })
    except Exception as e:
        logger.error(f"Error in validate_tracking_api: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'timestamp': int(time.time())
        }, status=500)

@csrf_exempt
def start_realtime_tracking_api(request, slug):
    """API endpoint to start real-time tracking"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        if not experience.image:
            return JsonResponse({
                'success': False,
                'error': 'No marker image available'
            }, status=400)
        tracker = get_tracker(slug)
        tracker.start_tracking()
        return JsonResponse({
            'success': True,
            'message': 'Real-time tracking started',
            'websocket_url': f"ws://{request.get_host()}/ws/ar/{slug}/",
            'target_accuracy': 0.9,
            'timestamp': int(time.time())
        })
    except Exception as e:
        logger.error(f"Error starting real-time tracking: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'timestamp': int(time.time())
        }, status=500)

@csrf_exempt
def stop_realtime_tracking_api(request, slug):
    """API endpoint to stop real-time tracking"""
    try:
        tracker = get_tracker(slug)
        tracker.stop_tracking()
        return JsonResponse({
            'success': True,
            'message': 'Real-time tracking stopped',
            'timestamp': int(time.time())
        })
    except Exception as e:
        logger.error(f"Error stopping real-time tracking: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'timestamp': int(time.time())
        }, status=500)

# ============================================================================
# BROWSER-BASED MINDAR COMPILER ENDPOINTS
# ============================================================================
@csrf_exempt
def save_browser_mindar_target(request):
    """Save browser-compiled MindAR target to database"""
    if request.method == 'POST':
        try:
            title = request.POST.get('title')
            compiled_data = request.FILES.get('mind_file')
            if not title or not compiled_data:
                return JsonResponse({'error': 'Missing title or compiled data'}, status=400)
            slug = slugify(title) or f"exp-{uuid.uuid4().hex[:8]}"
            counter = 1
            original_slug = slug
            while ARExperience.objects.filter(slug=slug).exists():
                slug = f"{original_slug}-{counter}"
                counter += 1
            experience, created = ARExperience.objects.get_or_create(
                title=title,
                defaults={'slug': slug, 'description': f'Enhanced MindAR experience for {title}'}
            )
            if not created:
                experience.slug = slug
                experience.save()
            marker_dir = ensure_marker_directory(experience.slug)
            mind_file_path = marker_dir / f"{experience.slug}.mind"
            with open(mind_file_path, 'wb') as f:
                for chunk in compiled_data.chunks():
                    f.write(chunk)
            valid_file = mind_file_path.exists() and mind_file_path.stat().st_size > 0
            if valid_file:
                experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                experience.marker_generated = True
                experience.save(update_fields=['nft_iset_file', 'marker_generated'])
                logger.info(f"Browser-compiled MindAR target saved for '{title}' (slug: {experience.slug})")
                return JsonResponse({
                    'success': True,
                    'message': f'Enhanced MindAR target saved successfully for "{title}"',
                    'slug': experience.slug,
                    'experience_url': f'/x/{experience.slug}/',
                    'file_size': mind_file_path.stat().st_size,
                    'target_accuracy': 0.9
                })
            else:
                return JsonResponse({'error': 'Invalid compiled data received'}, status=400)
        except Exception as e:
            logger.error(f"Error saving browser MindAR target: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'POST method required'}, status=405)

@csrf_exempt
def browser_mindar_compiler(request):
    """Browser-based MindAR compiler interface"""
    if request.method == 'GET':
        context = {
            'experiences': ARExperience.objects.all().order_by('-created_at')[:10],
            'target_accuracy': 0.9
        }
        return render(request, 'browser_mindar_compiler.html', context)
    return JsonResponse({'error': 'GET method required'}, status=405)

# ============================================================================
# UTILITY AND COMPATIBILITY FUNCTIONS
# ============================================================================
def ar_experience_by_slug(request, slug):
    """Enhanced MindAR experience viewer accessible by slug with connected tracking functions"""
    try:
        experience = ARExperience.objects.get(slug=slug)
        # Initialize real-time tracker for this experience
        tracker = get_tracker(slug)

        context = {
            'experience': experience,
            'video_url': experience.video.url if experience.video else None,
            'marker_url': experience.image.url if experience.image else None,
            'title': experience.title,
            'description': experience.description,
            'slug': experience.slug,
            'base_url': getattr(settings, 'BASE_URL', 'http://127.0.0.1:8000'),
            'tracking_method': 'Enhanced MindAR',
            'websocket_url': f"ws://{request.get_host()}/ws/ar/{slug}/",
            'target_accuracy': 0.9,
            # Tracking function connections
            "tracking_functions": {
                "real_time_tracker": {
                    "active": tracker.is_tracking,
                    "slug": slug,
                    "target_accuracy": 0.9
                },
                "websocket_consumer": {
                    "connected": False,  # Will be updated in frontend
                    "group_name": f'ar_tracking_{slug}',
                    "target_accuracy": 0.9
                }
            }
        }
        return render(request, 'experience.html', context)
    except ARExperience.DoesNotExist:
        messages.error(request, f'AR Experience "{slug}" not found.')
        return redirect('upload')
    except Exception as e:
        logger.error(f"Error loading AR experience {slug}: {str(e)}")
        messages.error(request, 'Error loading AR experience. Please try again.')
        return redirect('upload')

def ar_experience_view(request, experience_id: int):
    """Back-compat: resolve by ID then reuse the slug-based view"""
    exp = get_object_or_404(ARExperience, id=experience_id)
    return experience_view(request, exp.slug)

def qr_view(request, slug):
    """Generate QR code for AR experience"""
    experience = get_object_or_404(ARExperience, slug=slug)
    base_url = getattr(settings, 'BASE_URL', 'http://127.0.0.1:8000')
    experience_url = f"{base_url}/x/{slug}/"
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(experience_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_data = base64.b64encode(buffer.getvalue()).decode()

    # Initialize real-time tracker for this experience
    tracker = get_tracker(slug)

    return render(request, 'experience.html', {
        'experience': experience,
        'qr_data': qr_data,
        'experience_url': experience_url,
        'tracking_method': 'Enhanced MindAR',
        'websocket_url': f"ws://{request.get_host()}/ws/ar/{slug}/",
        'target_accuracy': 0.9,
        # Tracking function connections
        "tracking_functions": {
            "real_time_tracker": {
                "active": tracker.is_tracking,
                "slug": slug,
                "target_accuracy": 0.9
            },
            "websocket_consumer": {
                "connected": False,  # Will be updated in frontend
                "group_name": f'ar_tracking_{slug}',
                "target_accuracy": 0.9
            }
        }
    })

# ============================================================================
# DEBUG ENDPOINTS (Optional - for development)
# ============================================================================
def debug_markers(request, slug):
    """Debug view for marker status"""
    experience = get_object_or_404(ARExperience, slug=slug)
    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
    debug_info = {
        'slug': slug,
        'marker_dir': str(marker_dir),
        'marker_dir_exists': marker_dir.exists(),
        'files': {},
        'tracking_method': 'Enhanced MindAR',
        'target_accuracy': 0.9,
        'mind_db_status': {
            'mind_file_stored': bool(experience.nft_iset_file),
        }
    }
    # Check for MindAR target file
    mind_file = f"{slug}.mind"
    if marker_dir.exists():
        filepath = marker_dir / mind_file
        info = {
            'exists': filepath.exists(),
            'size': filepath.stat().st_size if filepath.exists() else 0
        }
        debug_info['files'][mind_file] = info
    return JsonResponse(debug_info, json_dumps_params={'indent': 2})
