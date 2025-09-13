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
from .advanced_ar_tracking import integrate_with_django_views

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
def create_trackable_marker(artistic_image_path, output_path=None):
    """
    Convert artistic images into AR-trackable markers by:
    - Adding high-contrast borders
    - Including tracking patterns in corners  
    - Making text opaque
    - Reducing glow effects
    - Adding geometric shapes for feature detection
    """
    import cv2
    import numpy as np
    import os
    
    # Read the artistic image
    img = cv2.imread(artistic_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {artistic_image_path}")
    
    height, width = img.shape[:2]
    
    # Step 1: Reduce glow effects and smooth gradients
    # Apply bilateral filter to reduce noise while preserving edges
    deglowed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Step 2: Enhance contrast moderately (not extreme)
    alpha = 1.4  # Contrast control
    beta = 20    # Brightness control  
    enhanced = cv2.convertScaleAbs(deglowed, alpha=alpha, beta=beta)
    
    # Step 3: Add thick black border for detection
    border_size = max(20, int(min(height, width) * 0.05))  # At least 20px border
    bordered = cv2.copyMakeBorder(
        enhanced, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # Update dimensions after border
    new_height, new_width = bordered.shape[:2]
    
    # Step 4: Add solid geometric tracking shapes in corners
    corner_size = int(min(height, width) * 0.08)  # 8% corner squares
    
    # White squares in corners for high contrast
    cv2.rectangle(bordered, 
                  (border_size, border_size), 
                  (border_size + corner_size, border_size + corner_size), 
                  (255, 255, 255), -1)  # Top-left
    
    cv2.rectangle(bordered, 
                  (new_width - border_size - corner_size, border_size), 
                  (new_width - border_size, border_size + corner_size), 
                  (255, 255, 255), -1)  # Top-right
    
    cv2.rectangle(bordered, 
                  (border_size, new_height - border_size - corner_size), 
                  (border_size + corner_size, new_height - border_size), 
                  (255, 255, 255), -1)  # Bottom-left
    
    cv2.rectangle(bordered, 
                  (new_width - border_size - corner_size, new_height - border_size - corner_size), 
                  (new_width - border_size, new_height - border_size), 
                  (255, 255, 255), -1)  # Bottom-right
    
    # Step 5: Add black squares inside white squares for better feature detection
    inner_size = corner_size // 3
    offset = (corner_size - inner_size) // 2
    
    cv2.rectangle(bordered, 
                  (border_size + offset, border_size + offset), 
                  (border_size + offset + inner_size, border_size + offset + inner_size), 
                  (0, 0, 0), -1)  # Top-left inner
    
    cv2.rectangle(bordered, 
                  (new_width - border_size - corner_size + offset, border_size + offset), 
                  (new_width - border_size - corner_size + offset + inner_size, border_size + offset + inner_size), 
                  (0, 0, 0), -1)  # Top-right inner
    
    cv2.rectangle(bordered, 
                  (border_size + offset, new_height - border_size - corner_size + offset), 
                  (border_size + offset + inner_size, new_height - border_size - corner_size + offset + inner_size), 
                  (0, 0, 0), -1)  # Bottom-left inner
    
    cv2.rectangle(bordered, 
                  (new_width - border_size - corner_size + offset, new_height - border_size - corner_size + offset), 
                  (new_width - border_size - corner_size + offset + inner_size, new_height - border_size - corner_size + offset + inner_size), 
                  (0, 0, 0), -1)  # Bottom-right inner
    
    # Step 6: Enhance text visibility by creating high-contrast text areas
    # Convert to grayscale for text detection
    gray = cv2.cvtColor(bordered, cv2.COLOR_BGR2GRAY)
    
    # Find bright text regions (assuming text is brighter than background)
    _, text_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Dilate to capture glow around text
    kernel = np.ones((3, 3), np.uint8)
    text_regions = cv2.dilate(text_mask, kernel, iterations=2)
    
    # Make text areas completely white (opaque)
    bordered[text_regions == 255] = [255, 255, 255]
    
    # Step 7: Add edge detection patterns along borders
    edge_thickness = max(2, border_size // 10)
    
    # Add white lines along inner border edges for additional tracking features
    cv2.rectangle(bordered, 
                  (border_size - edge_thickness, border_size - edge_thickness), 
                  (new_width - border_size + edge_thickness, new_height - border_size + edge_thickness), 
                  (255, 255, 255), edge_thickness)
    
    # Step 8: Apply final sharpening for crisp edges
    sharpening_kernel = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(bordered, -1, sharpening_kernel)
    
    # Ensure we don't over-sharpen
    final_image = cv2.addWeighted(bordered, 0.7, sharpened, 0.3, 0)
    
    # Save the processed image
    if output_path is None:
        base_name = os.path.splitext(artistic_image_path)[0]
        output_path = f"{base_name}_trackable.png"
    
    success = cv2.imwrite(output_path, final_image)
    
    if not success:
        raise ValueError(f"Failed to save trackable marker to: {output_path}")
    
    return output_path


# Enhanced version for your Django integration
def generate_mindar_marker_with_preprocessing(image_path, output_path):
    """
    Enhanced MindAR marker generation with automatic artistic image preprocessing
    """
    try:
        logger.info(f"Starting trackable marker generation for: {image_path}")
        
        # Step 1: Create trackable version of artistic image
        trackable_image_path = image_path.replace('.jpg', '_trackable.png').replace('.jpeg', '_trackable.png')
        
        # Convert artistic image to trackable format
        create_trackable_marker(image_path, trackable_image_path)
        
        # Step 2: Use the trackable image for MindAR marker generation
        img = cv2.imread(trackable_image_path)
        if img is None:
            logger.error(f"Could not read trackable image: {trackable_image_path}")
            return False
        
        # Stories AR gentle enhancement (from previous implementation)
        enhanced_color = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        
        # Use grayscale ONLY for feature detection
        gray = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)
        
        # Gentle CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # Stories AR feature detection
        orb = cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10
        )
        
        keypoints, descriptors = orb.detectAndCompute(enhanced_gray, None)
        
        # Ensure minimum features
        if len(keypoints) < 100:
            corners = cv2.goodFeaturesToTrack(
                enhanced_gray, 
                maxCorners=200, 
                qualityLevel=0.01, 
                minDistance=10
            )
            if corners is not None:
                corner_kps = [cv2.KeyPoint(x[0][0], x[0][1], 7) for x in corners]
                keypoints.extend(corner_kps[:200-len(keypoints)])
        
        logger.info(f"Trackable marker generated with {len(keypoints)} features")
        
        # Create MindAR data structure
        height, width = enhanced_color.shape[:2]
        
        mindar_data = {
            "imageWidth": width,
            "imageHeight": height,
            "maxTrack": 1,
            "filterMinCF": 0.0001,
            "filterBeta": 0.001,
            "missTolerance": 2,
            "warmupTolerance": 2,
            "targetAccuracy": 0.95,
            "targets": [{
                "name": os.path.splitext(os.path.basename(image_path))[0],
                "keypoints": [],
                "descriptors": []
            }]
        }
        
        # Add keypoints
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:500]
        for i, kp in enumerate(keypoints):
            if kp.response > 0.01:
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
        
        # Save MindAR file
        json_str = json.dumps(mindar_data, separators=(',', ':'))
        header = struct.pack('4sI', b'MIND', len(json_str))
        
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(json_str.encode('utf-8'))
        
        # Save the trackable image as reference
        trackable_reference_path = output_path.replace('.mind', '_trackable.png')
        cv2.imwrite(trackable_reference_path, enhanced_color)
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 100
        
    except Exception as e:
        logger.error(f"Error generating trackable MindAR marker: {str(e)}")
        return False


def generate_mindar_marker(image_path, output_path):
    """
    Stories AR-style marker generation - preserves color information for better tracking
    Returns True if successful, False otherwise
    """
    try:
        logger.info(f"Starting Stories AR-style marker generation for: {image_path}")
        
        # Read original image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return False
        
        # Stories AR approach: Preserve color information, gentle enhancement only
        
        # Step 1: Check if image is already high-contrast binary (problematic)
        gray_check = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        unique_values = len(np.unique(gray_check))
        if unique_values < 10:  # Likely high-contrast binary
            logger.warning(f"Input image appears to be high-contrast binary, this will cause tracking issues")
        
        # Step 2: Stories AR gentle enhancement (NO harsh binary conversion)
        # Use the original color image with minimal processing
        enhanced_color = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  # Gentle contrast boost
        
        # Optional: Very gentle sharpening (much milder than before)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])  # Gentler kernel
        sharpened_color = cv2.filter2D(enhanced_color, -1, kernel)
        
        # Step 3: Use grayscale ONLY for feature detection, not for the final marker
        gray = cv2.cvtColor(sharpened_color, cv2.COLOR_BGR2GRAY)
        
        # Apply gentle CLAHE (much milder than before)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # Reduced from 4.0
        enhanced_gray = clahe.apply(gray)
        
        # Step 4: Stories AR feature detection (using enhanced grayscale)
        orb = cv2.ORB_create(
            nfeatures=1000,        # Stories AR standard
            scaleFactor=1.2,       # Less aggressive
            nlevels=8,             # Fewer levels for stability
            edgeThreshold=10,      
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10       # Higher threshold for quality
        )
        
        keypoints, descriptors = orb.detectAndCompute(enhanced_gray, None)
        
        # Step 5: Ensure minimum feature count without destroying image
        if len(keypoints) < 100:  # Stories AR standard
            corners = cv2.goodFeaturesToTrack(
                enhanced_gray, 
                maxCorners=200, 
                qualityLevel=0.01, 
                minDistance=10
            )
            
            if corners is not None:
                corner_kps = [cv2.KeyPoint(x[0][0], x[0][1], 7) for x in corners]
                keypoints.extend(corner_kps[:200-len(keypoints)])
        
        logger.info(f"Stories AR marker generated with {len(keypoints)} features")
        
        # Filter and sort keypoints (Stories AR quality standards)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:500]
        if descriptors is not None:
            descriptors = descriptors[:len(keypoints)]
        
        # Step 6: Use the ENHANCED COLOR image dimensions (not binary)
        height, width = sharpened_color.shape[:2]
        
        # Step 7: Stories AR MindAR data structure (optimized parameters)
        mindar_data = {
            "imageWidth": width,
            "imageHeight": height,
            "maxTrack": 1,
            "filterMinCF": 0.0001,  # Stories AR sensitivity
            "filterBeta": 0.001,    # Stories AR responsiveness
            "missTolerance": 2,     # Stories AR tolerance (lower = faster)
            "warmupTolerance": 2,   # Stories AR warmup (faster detection)
            "targetAccuracy": 0.95, # Stories AR level
            "targets": [{
                "name": os.path.splitext(os.path.basename(image_path))[0],
                "keypoints": [],
                "descriptors": []
            }]
        }
        
        # Add high-quality keypoints only
        for i, kp in enumerate(keypoints):
            if kp.response > 0.01:  # Higher quality threshold (Stories AR)
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
        
        # Step 8: Save MindAR file
        json_str = json.dumps(mindar_data, separators=(',', ':'))
        header = struct.pack('4sI', b'MIND', len(json_str))
        
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(json_str.encode('utf-8'))
        
        # Step 9: Save the ENHANCED COLOR image as reference (NOT binary)
        enhanced_image_path = output_path.replace('.mind', '_enhanced.png')
        cv2.imwrite(enhanced_image_path, sharpened_color)  # Color image, not binary
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logger.info(f"Stories AR-style marker generated successfully: {output_path}")
            return True
        else:
            logger.error(f"Failed to create MindAR marker file: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error generating Stories AR marker: {str(e)}")
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

# ============================================================================
# AUTOMATED MARKER CREATION SYSTEM
# ============================================================================

def classify_image_type(image_path):
    """
    Automatically classify image type and determine optimal processing strategy
    """
    import cv2
    import numpy as np
    
    img = cv2.imread(image_path)
    if img is None:
        return "invalid"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate image characteristics
    unique_colors = len(np.unique(gray))
    contrast = gray.std()
    brightness = gray.mean()
    
    # Detect edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Classify image type
    if unique_colors < 10 and contrast > 80:
        return "high_contrast_binary"  # Your problematic black/white images
    elif edge_density > 0.15:
        return "complex_artistic"      # Rainbow bridge type images
    elif edge_density < 0.02:
        return "low_detail"           # Simple images, gradients
    elif 40 <= contrast <= 70 and edge_density > 0.05:
        return "architectural"        # Temple-type images (ideal)
    elif brightness > 200 or brightness < 50:
        return "extreme_lighting"     # Very bright/dark images
    elif contrast < 30:
        return "low_contrast"         # Washed out images
    else:
        return "standard"             # Regular photos

def get_processing_strategy(image_type):
    """
    Return optimal processing parameters for each image type
    """
    strategies = {
        "high_contrast_binary": {
            "needs_color_restoration": True,
            "border_size": 0.08,
            "corner_patterns": "large",
            "contrast_adjustment": 0.8,  # Reduce harsh contrast
            "blur_reduction": True
        },
        "complex_artistic": {
            "needs_simplification": True,
            "border_size": 0.06,
            "corner_patterns": "medium",
            "contrast_adjustment": 1.3,
            "add_tracking_elements": True
        },
        "architectural": {
            "minimal_processing": True,
            "border_size": 0.03,
            "corner_patterns": "small",
            "contrast_adjustment": 1.1,
            "preserve_details": True
        },
        "low_detail": {
            "needs_feature_addition": True,
            "border_size": 0.07,
            "corner_patterns": "large",
            "contrast_adjustment": 1.5,
            "add_geometric_shapes": True
        },
        "extreme_lighting": {
            "needs_normalization": True,
            "border_size": 0.05,
            "corner_patterns": "medium",
            "contrast_adjustment": 1.2,
            "brightness_correction": True
        },
        "low_contrast": {
            "needs_enhancement": True,
            "border_size": 0.05,
            "corner_patterns": "medium", 
            "contrast_adjustment": 1.4,
            "edge_enhancement": True
        },
        "standard": {
            "minimal_processing": True,
            "border_size": 0.04,
            "corner_patterns": "small",
            "contrast_adjustment": 1.2,
            "gentle_enhancement": True
        }
    }
    return strategies.get(image_type, strategies["standard"])

def create_automated_trackable_marker(image_path, output_path=None):
    """
    Automatically create trackable markers based on image type detection
    """
    import cv2
    import numpy as np
    import os
    
    # Step 1: Classify image type
    image_type = classify_image_type(image_path)
    strategy = get_processing_strategy(image_type)
    
    logger.info(f"Detected image type: {image_type}, applying strategy: {strategy}")
    
    # Step 2: Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Step 3: Apply type-specific preprocessing
    if strategy.get("needs_color_restoration"):
        # For high-contrast binary images - restore some color/gradient
        img = restore_gradient_information(img)
    
    elif strategy.get("needs_simplification"):
        # For complex artistic images - reduce visual noise
        img = simplify_artistic_image(img)
    
    elif strategy.get("needs_feature_addition"):
        # For low-detail images - add trackable features
        img = add_tracking_features(img)
    
    elif strategy.get("needs_normalization"):
        # For extreme lighting - normalize exposure
        img = normalize_lighting(img)
    
    elif strategy.get("needs_enhancement"):
        # For low contrast - enhance carefully
        img = enhance_contrast_safely(img)
    
    # Step 4: Apply universal improvements
    processed = apply_universal_improvements(img, strategy)
    
    # Step 5: Add tracking elements based on strategy
    final_marker = add_automated_tracking_elements(processed, strategy, width, height)
    
    # Step 6: Validate and save
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_auto_trackable.png"
    
    cv2.imwrite(output_path, final_marker)
    
    # Step 7: Quality assessment
    quality_score = assess_marker_quality(final_marker)
    logger.info(f"Generated marker quality score: {quality_score}")
    
    return output_path, quality_score, image_type

# Helper functions for specific image types
def restore_gradient_information(img):
    """Restore gradients to high-contrast binary images"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to create gradients
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Blend original with blurred version
    restored = cv2.addWeighted(gray, 0.7, blurred, 0.3, 0)
    
    # Convert back to color
    return cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)

def simplify_artistic_image(img):
    """Reduce visual noise in artistic images"""
    # Reduce color palette
    simplified = cv2.medianBlur(img, 5)
    
    # Apply bilateral filter to smooth gradients while preserving edges
    simplified = cv2.bilateralFilter(simplified, 15, 80, 80)
    
    return simplified

def add_tracking_features(img):
    """Add geometric patterns to low-detail images"""
    # This adds subtle geometric patterns that don't interfere with the original image
    overlay = np.zeros_like(img)
    height, width = img.shape[:2]
    
    # Add corner triangles (very subtle)
    triangle_size = min(width, height) // 20
    
    # Draw subtle triangular patterns in corners
    pts = np.array([[0, 0], [triangle_size, 0], [0, triangle_size]], np.int32)
    cv2.fillPoly(overlay, [pts], (30, 30, 30))  # Very dark gray
    
    # Add to other corners (transformed)
    pts2 = np.array([[width, 0], [width-triangle_size, 0], [width, triangle_size]], np.int32)
    cv2.fillPoly(overlay, [pts2], (30, 30, 30))
    
    pts3 = np.array([[0, height], [triangle_size, height], [0, height-triangle_size]], np.int32)
    cv2.fillPoly(overlay, [pts3], (30, 30, 30))
    
    pts4 = np.array([[width, height], [width-triangle_size, height], [width, height-triangle_size]], np.int32)
    cv2.fillPoly(overlay, [pts4], (30, 30, 30))
    
    return cv2.addWeighted(img, 0.95, overlay, 0.05, 0)

def normalize_lighting(img):
    """Normalize extreme lighting conditions"""
    # Convert to LAB color space for better exposure control
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge and convert back
    normalized = cv2.merge([l, a, b])
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

def enhance_contrast_safely(img):
    """Enhance contrast without over-processing"""
    # Use adaptive histogram equalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def apply_universal_improvements(img, strategy):
    """Apply improvements that work for all image types"""
    # Gentle sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Blend original with sharpened
    alpha = 0.3 if strategy.get("preserve_details") else 0.2
    improved = cv2.addWeighted(img, 1-alpha, sharpened, alpha, 0)
    
    # Contrast adjustment
    contrast_factor = strategy.get("contrast_adjustment", 1.0)
    final = cv2.convertScaleAbs(improved, alpha=contrast_factor, beta=5)
    
    return final

def add_automated_tracking_elements(img, strategy, original_width, original_height):
    """Add borders and tracking elements based on strategy"""
    border_size = int(min(original_width, original_height) * strategy.get("border_size", 0.05))
    
    # Add border
    bordered = cv2.copyMakeBorder(
        img, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    # Add corner patterns based on strategy
    pattern_type = strategy.get("corner_patterns", "medium")
    corner_size = {
        "small": int(min(original_width, original_height) * 0.06),
        "medium": int(min(original_width, original_height) * 0.08),
        "large": int(min(original_width, original_height) * 0.10)
    }.get(pattern_type, int(min(original_width, original_height) * 0.08))
    
    # Draw corner patterns
    new_height, new_width = bordered.shape[:2]
    
    # White squares with black centers
    positions = [
        (border_size, border_size),  # Top-left
        (new_width - border_size - corner_size, border_size),  # Top-right
        (border_size, new_height - border_size - corner_size),  # Bottom-left
        (new_width - border_size - corner_size, new_height - border_size - corner_size)  # Bottom-right
    ]
    
    for x, y in positions:
        # White square
        cv2.rectangle(bordered, (x, y), (x + corner_size, y + corner_size), (255, 255, 255), -1)
        # Black center
        center_size = corner_size // 3
        center_offset = (corner_size - center_size) // 2
        cv2.rectangle(bordered, 
                     (x + center_offset, y + center_offset), 
                     (x + center_offset + center_size, y + center_offset + center_size), 
                     (0, 0, 0), -1)
    
    return bordered

def assess_marker_quality(marker_image):
    """Assess the quality of the generated marker"""
    gray = cv2.cvtColor(marker_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    contrast = gray.std()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Feature detection test
    orb = cv2.ORB_create(nfeatures=500)
    keypoints = orb.detect(gray, None)
    feature_count = len(keypoints)
    
    # Calculate quality score (0-1)
    contrast_score = min(1.0, contrast / 80.0)
    edge_score = min(1.0, edge_density / 0.08)
    feature_score = min(1.0, feature_count / 200.0)
    
    overall_score = (contrast_score + edge_score + feature_score) / 3
    
    return overall_score

def generate_mindar_marker_automated(image_path, output_path):
    """
    Enhanced MindAR marker generation with automated image type detection
    """
    try:
        logger.info(f"Starting automated marker generation for: {image_path}")
        
        # Step 1: Create automated trackable marker
        trackable_path, quality_score, image_type = create_automated_trackable_marker(image_path)
        
        logger.info(f"Image type: {image_type}, Quality score: {quality_score:.2f}")
        
        # Step 2: Use trackable image for MindAR generation
        img = cv2.imread(trackable_path)
        if img is None:
            logger.error(f"Could not read processed image: {trackable_path}")
            return False
        
        # Step 3: Generate MindAR data (same as before)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Feature detection with parameters based on image type
        if image_type in ["architectural", "standard"]:
            # Use standard parameters for good images
            orb_features = 800
        elif image_type in ["complex_artistic", "low_detail"]:
            # Use more features for challenging images
            orb_features = 1200
        else:
            # Conservative approach for problematic images
            orb_features = 600
        
        orb = cv2.ORB_create(
            nfeatures=orb_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10
        )
        
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Ensure adequate features
        if len(keypoints) < 50:
            # Add corner features if insufficient
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.01, minDistance=10)
            if corners is not None:
                corner_kps = [cv2.KeyPoint(x[0][0], x[0][1], 7) for x in corners]
                keypoints.extend(corner_kps[:300-len(keypoints)])
        
        # Create MindAR data
        height, width = img.shape[:2]
        
        mindar_data = {
            "imageWidth": width,
            "imageHeight": height,
            "maxTrack": 1,
            "filterMinCF": 0.0001,
            "filterBeta": 0.001,
            "missTolerance": 2,
            "warmupTolerance": 2,
            "targetAccuracy": 0.95,
            "imageType": image_type,  # Store for debugging
            "qualityScore": quality_score,
            "targets": [{
                "name": os.path.splitext(os.path.basename(image_path))[0],
                "keypoints": [],
                "descriptors": []
            }]
        }
        
        # Add keypoints
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:500]
        for i, kp in enumerate(keypoints):
            if kp.response > 0.01:
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
        
        # Save MindAR file
        json_str = json.dumps(mindar_data, separators=(',', ':'))
        header = struct.pack('4sI', b'MIND', len(json_str))
        
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(json_str.encode('utf-8'))
        
        logger.info(f"Automated marker generated: {len(keypoints)} features, type: {image_type}")
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 100
        
    except Exception as e:
        logger.error(f"Error in automated marker generation: {str(e)}")
        return False

def batch_process_markers(image_directory, output_directory):
    """
    Process multiple images automatically
    """
    import os
    from pathlib import Path
    
    results = []
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for image_file in Path(image_directory).iterdir():
        if image_file.suffix.lower() in supported_formats:
            try:
                output_path = Path(output_directory) / f"{image_file.stem}_marker.mind"
                success = generate_mindar_marker_automated(str(image_file), str(output_path))
                
                results.append({
                    'input': str(image_file),
                    'output': str(output_path),
                    'success': success,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                results.append({
                    'input': str(image_file),
                    'error': str(e),
                    'success': False
                })
    
    return results

def validate_marker_quality(image_path):
    """Validate marker image quality for AR tracking"""
    results = {
        'score': 0.0,
        'issues': [],
        'recommendations': [],
        'stats': {},
        'target_accuracy': 0.9,  # Stories AR level
        'quality_level': 'poor',  # poor, fair, good, excellent
        'valid': False
    }
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            results['issues'].append('Cannot read image file')
            return results
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Score components
        score_components = {
            'resolution': 0,
            'contrast': 0,
            'edge_density': 0,
            'features': 0,
            'border': 0,
            'symmetry': 0  # Added for better tracking
        }
        
        # 1. Resolution Check (0-0.2 points)
        results['stats']['resolution'] = f"{width}x{height}"
        if width < 200 or height < 200:
            results['issues'].append('Image too small (minimum 200x200)')
            results['recommendations'].append('Use higher resolution image (recommended: 512x512)')
        elif width > 2048 or height > 2048:
            results['issues'].append('Image too large (maximum 2048x2048)')
            results['recommendations'].append('Resize image for better performance')
        else:
            # Optimal resolution scoring
            if 400 <= width <= 1024 and 400 <= height <= 1024:
                score_components['resolution'] = 0.2
            else:
                score_components['resolution'] = 0.1
        
        # 2. Contrast Analysis (0-0.25 points)
        contrast = gray.std()
        mean_brightness = gray.mean()
        results['stats']['contrast'] = round(contrast, 2)
        results['stats']['brightness'] = round(mean_brightness, 2)
        
        if contrast < 30:
            results['issues'].append('Very low contrast - marker features not distinct')
            results['recommendations'].append('Increase contrast or use high-contrast design')
        elif contrast < 50:
            results['issues'].append('Low contrast - may affect tracking reliability')
            results['recommendations'].append('Consider increasing contrast for better tracking')
            score_components['contrast'] = 0.1
        elif contrast > 80:
            score_components['contrast'] = 0.25
        else:
            score_components['contrast'] = 0.15
            
        # Check brightness range
        if mean_brightness < 50 or mean_brightness > 200:
            results['issues'].append('Extreme brightness levels may affect tracking')
            results['recommendations'].append('Adjust brightness to mid-range (100-150)')
        
        # 3. Edge Density (0-0.2 points)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        results['stats']['edge_density'] = round(edge_density, 4)
        
        if edge_density < 0.01:
            results['issues'].append('Very low edge density - insufficient features')
            results['recommendations'].append('Add more distinct patterns and shapes')
        elif edge_density < 0.03:
            results['issues'].append('Low edge density - may affect tracking')
            results['recommendations'].append('Consider adding more geometric patterns')
            score_components['edge_density'] = 0.1
        elif edge_density > 0.15:
            results['issues'].append('Very high edge density - may cause confusion')
            results['recommendations'].append('Simplify some patterns')
            score_components['edge_density'] = 0.1
        elif 0.05 <= edge_density <= 0.12:
            score_components['edge_density'] = 0.2
        else:
            score_components['edge_density'] = 0.15
        
        # 4. Feature Quality (0-0.25 points)
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        feature_count = len(keypoints)
        results['stats']['keypoints'] = feature_count
        
        if feature_count < 50:
            results['issues'].append(f'Very few features: {feature_count} (minimum 100 recommended)')
            results['recommendations'].append('Add more complex patterns and textures')
        elif feature_count < 100:
            results['issues'].append(f'Low feature count: {feature_count} (recommend 200+)')
            results['recommendations'].append('Add more detailed patterns')
            score_components['features'] = 0.1
        elif feature_count > 500:
            score_components['features'] = 0.25
        elif feature_count > 200:
            score_components['features'] = 0.2
        else:
            score_components['features'] = 0.15
            
        # Feature distribution check
        if keypoints:
            kp_coords = np.array([kp.pt for kp in keypoints])
            x_std, y_std = kp_coords.std(axis=0)
            feature_distribution = min(x_std, y_std) / max(width, height)
            results['stats']['feature_distribution'] = round(feature_distribution, 3)
            
            if feature_distribution < 0.1:
                results['issues'].append('Features clustered in small area')
                results['recommendations'].append('Distribute patterns across entire image')
        
        # 5. Border Quality (0-0.15 points)
        border_score = check_border_quality(gray)
        results['stats']['border_score'] = round(border_score, 2)
        
        if border_score < 0.5:
            results['issues'].append('Poor border definition - affects detection')
            results['recommendations'].append('Add thick, solid border around marker')
        elif border_score < 0.7:
            results['issues'].append('Weak border - may affect detection reliability')
            results['recommendations'].append('Strengthen border contrast')
            score_components['border'] = 0.05
        elif border_score > 0.9:
            score_components['border'] = 0.15
        else:
            score_components['border'] = 0.1
        
        # 6. Symmetry Check (0-0.15 points)
        symmetry_score = check_symmetry(gray)
        results['stats']['symmetry'] = round(symmetry_score, 2)
        
        if symmetry_score > 0.95:
            results['issues'].append('Image too symmetric - may cause tracking ambiguity')
            results['recommendations'].append('Add some asymmetric elements')
            score_components['symmetry'] = 0.05
        elif 0.3 <= symmetry_score <= 0.8:
            score_components['symmetry'] = 0.15
        else:
            score_components['symmetry'] = 0.1
        
        # Calculate final score
        results['score'] = sum(score_components.values())
        results['score_breakdown'] = score_components
        
        # Determine quality level
        if results['score'] >= 0.9:
            results['quality_level'] = 'excellent'
            results['valid'] = True
        elif results['score'] >= 0.7:
            results['quality_level'] = 'good' 
            results['valid'] = True
        elif results['score'] >= 0.5:
            results['quality_level'] = 'fair'
            results['valid'] = True
        else:
            results['quality_level'] = 'poor'
            results['valid'] = False
            
        # Add general recommendations based on quality level
        if results['quality_level'] == 'poor':
            results['recommendations'].append('Consider using a different image with better contrast and features')
        elif results['quality_level'] == 'fair':
            results['recommendations'].append('Image should work but may have tracking issues in poor lighting')
        
        return results
        
    except Exception as e:
        logger.error(f"Error in validate_marker_quality: {str(e)}")
        results['issues'].append(f'Quality check error: {str(e)}')
        return results


def check_border_quality(gray):
    """Check the quality of the image border for AR tracking"""
    try:
        height, width = gray.shape
        border_width = min(width, height) // 20  # 5% border
        
        # Sample border pixels
        top_border = gray[:border_width, :]
        bottom_border = gray[-border_width:, :]
        left_border = gray[:, :border_width]
        right_border = gray[:, -border_width:]
        
        # Calculate border contrast
        center = gray[height//4:3*height//4, width//4:3*width//4]
        
        border_mean = np.mean([
            top_border.mean(), bottom_border.mean(),
            left_border.mean(), right_border.mean()
        ])
        center_mean = center.mean()
        
        border_contrast = abs(border_mean - center_mean) / 255.0
        
        # Check border consistency
        border_std = np.std([
            top_border.mean(), bottom_border.mean(),
            left_border.mean(), right_border.mean()
        ]) / 255.0
        
        # Combine metrics
        score = border_contrast * (1 - border_std)
        return min(1.0, score)
        
    except Exception:
        return 0.0


def check_symmetry(gray):
    """Check image symmetry to avoid tracking ambiguity"""
    try:
        height, width = gray.shape
        
        # Horizontal symmetry
        left_half = gray[:, :width//2]
        right_half = np.fliplr(gray[:, width//2:])
        
        # Resize to match if different sizes
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        horizontal_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        
        # Vertical symmetry
        top_half = gray[:height//2, :]
        bottom_half = np.flipud(gray[height//2:, :])
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        vertical_diff = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float))) / 255.0
        
        # Return average asymmetry (lower = more symmetric)
        return (horizontal_diff + vertical_diff) / 2
        
    except Exception:
        return 0.5  # Neutral score if error

def validate_marker_quality_stories_ar(image_path):
    """Enhanced validation optimized for Stories AR-level live tracking performance"""
    results = validate_marker_quality(image_path)  # Your existing function
    
    # Add Stories AR specific checks
    img = cv2.imread(image_path)
    if img is None:
        results['issues'].insert(0, 'Cannot read image file for validation')
        results['valid'] = False
        return results
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    unique_values = len(np.unique(gray))
    
    # **Critical Issue #1: High-contrast binary detection**
    if unique_values < 10:
        results['issues'].insert(0, 'Image is high-contrast binary - CRITICAL for live tracking')
        results['recommendations'].insert(0, 'Replace with original color image for Stories AR performance')
        results['score'] = min(0.2, results['score'])  # Severe penalty
        results['quality_level'] = 'critical'
        results['valid'] = False
    
    # **Critical Issue #2: Feature density for live tracking** 
    orb = cv2.ORB_create(nfeatures=2000)  # Stories AR standard
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    feature_count = len(keypoints)
    
    if feature_count < 200:  # Stories AR minimum
        results['issues'].insert(0, f'Insufficient features for live tracking: {feature_count} (need 200+)')
        results['recommendations'].insert(0, 'Add more detailed patterns, textures, and geometric shapes')
        results['score'] = min(0.4, results['score'])
        results['valid'] = False
    elif feature_count < 400:  # Stories AR preferred
        results['issues'].append(f'Low feature density for stable live tracking: {feature_count}')
        results['recommendations'].append('Increase visual complexity for better tracking stability')
        results['score'] *= 0.8
    
    # **Critical Issue #3: Feature distribution for angular tracking**
    if keypoints:
        kp_coords = np.array([kp.pt for kp in keypoints])
        x_coords, y_coords = kp_coords[:, 0], kp_coords[:, 1]
        
        # Check corner coverage (Stories AR requirement)
        corner_regions = [
            (x_coords < width * 0.3) & (y_coords < height * 0.3),  # Top-left
            (x_coords > width * 0.7) & (y_coords < height * 0.3),  # Top-right  
            (x_coords < width * 0.3) & (y_coords > height * 0.7),  # Bottom-left
            (x_coords > width * 0.7) & (y_coords > height * 0.7),  # Bottom-right
        ]
        
        corner_coverage = sum(np.any(region) for region in corner_regions)
        if corner_coverage < 3:
            results['issues'].insert(0, f'Poor corner coverage: {corner_coverage}/4 corners have features')
            results['recommendations'].insert(0, 'Add tracking elements in all four corners')
            results['score'] *= 0.6
            results['valid'] = False
    
    # **Issue #4: Contrast assessment for live tracking**
    contrast = gray.std()
    mean_brightness = gray.mean()
    
    if contrast < 60:  # Stories AR requires higher contrast
        results['issues'].append(f'Low contrast for live tracking: {contrast:.1f} (need 60+)')
        results['recommendations'].append('Increase contrast between elements')
        results['score'] *= 0.7
    
    # **Issue #5: Edge density optimization**
    edges = cv2.Canny(gray, 30, 100)  # More sensitive edge detection
    edge_density = np.sum(edges > 0) / edges.size
    
    if edge_density < 0.05:  # Stories AR minimum
        results['issues'].append(f'Low edge density for live tracking: {edge_density:.3f}')
        results['recommendations'].append('Add more geometric patterns and sharp edges')
        results['score'] *= 0.8
    elif edge_density > 0.20:  # Too busy for stable tracking
        results['issues'].append(f'Excessive edge density may cause tracking confusion: {edge_density:.3f}')
        results['recommendations'].append('Simplify some patterns while maintaining key features')
        results['score'] *= 0.9
    
    # **Issue #6: Symmetry detection (tracking ambiguity)**
    # Horizontal symmetry check
    left_half = gray[:, :width//2]
    right_half = np.fliplr(gray[:, width//2:])
    min_width = min(left_half.shape[1], right_half.shape[1])
    
    horizontal_symmetry = 1.0 - np.mean(np.abs(
        left_half[:, :min_width].astype(float) - right_half[:, :min_width].astype(float)
    )) / 255.0
    
    if horizontal_symmetry > 0.8:
        results['issues'].append('High horizontal symmetry may cause tracking orientation issues')
        results['recommendations'].append('Add asymmetric elements to break symmetry')
        results['score'] *= 0.8
    
    # **Issue #7: Size validation for mobile tracking**
    if width < 512 or height < 512:
        results['issues'].append(f'Small image size: {width}x{height} (recommend 512x512+)')
        results['recommendations'].append('Use higher resolution for better mobile tracking')
        results['score'] *= 0.9
    elif width > 2048 or height > 2048:
        results['issues'].append(f'Large image size may impact performance: {width}x{height}')
        results['recommendations'].append('Resize to 1024x1024 for optimal mobile performance')
    
    # **Issue #8: Repetitive pattern detection**
    # Simple repetition check using autocorrelation
    try:
        # Check for repetitive patterns that confuse tracking
        roi = gray[height//4:3*height//4, width//4:3*width//4]  # Center region
        
        # Template matching with shifted version to detect repetition
        shifted = roi[10:, 10:]  # Slightly shifted
        if shifted.shape[0] > 50 and shifted.shape[1] > 50:
            match_result = cv2.matchTemplate(
                roi[:-10, :-10], shifted, cv2.TM_CCOEFF_NORMED
            )
            max_correlation = np.max(match_result)
            
            if max_correlation > 0.8:
                results['issues'].append(f'Repetitive patterns detected: {max_correlation:.2f} similarity')
                results['recommendations'].append('Reduce repetitive elements that confuse tracking')
                results['score'] *= 0.7
    except Exception:
        pass  # Skip if correlation check fails
    
    # **Stories AR Performance Score Adjustment**
    # Apply Stories AR specific scoring criteria
    if results['score'] >= 0.85 and feature_count >= 400 and corner_coverage >= 3:
        results['quality_level'] = 'stories_ar_excellent'
        results['live_tracking_ready'] = True
    elif results['score'] >= 0.7 and feature_count >= 200:
        results['quality_level'] = 'stories_ar_good'  
        results['live_tracking_ready'] = True
    elif results['score'] >= 0.5:
        results['quality_level'] = 'stories_ar_fair'
        results['live_tracking_ready'] = False
        results['recommendations'].insert(0, 'Image needs improvement for reliable live tracking')
    else:
        results['quality_level'] = 'stories_ar_poor'
        results['live_tracking_ready'] = False
        results['recommendations'].insert(0, 'Image unsuitable for Stories AR-level tracking')
    
    # **Add Stories AR specific metrics**
    results['stories_ar_metrics'] = {
        'feature_count': feature_count,
        'corner_coverage': corner_coverage,
        'edge_density': round(edge_density, 4),
        'contrast_score': round(contrast, 2),
        'symmetry_score': round(horizontal_symmetry, 2),
        'unique_values': unique_values,
        'live_tracking_ready': results.get('live_tracking_ready', False),
        'recommended_for_mobile': width <= 1024 and height <= 1024,
        'target_fps': 60 if results.get('live_tracking_ready', False) else 30
    }
    
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
    """Real-time marker tracking processor optimized for Stories AR-level performance"""
    def __init__(self, experience_slug):
        self.experience_slug = experience_slug
        self.is_tracking = False
        self.tracking_thread = None
        self.channel_layer = None
        self.target_accuracy = 0.97  # **Stories AR level (increased)**
        self.advanced_tracker = None
        self.advanced_tracking_available = False
        
        # **NEW: Stories AR Performance Settings**
        self.tracking_fps = 60           # Target 60 FPS for live tracking
        self.feature_threshold = 25      # Minimum matches for detection (increased)
        self.quality_threshold = 0.02    # Higher quality requirement
        self.stability_frames = 3        # Frames needed for stable tracking
        self.live_tracking_optimized = False
        
        # **NEW: Check marker quality for optimization**
        try:
            experience = ARExperience.objects.get(slug=experience_slug)
            if experience.image:
                validation = validate_marker_quality_stories_ar(experience.image.path)
                self.live_tracking_optimized = validation.get('live_tracking_ready', False)
                self.target_accuracy = 0.98 if self.live_tracking_optimized else 0.95
                
                # **Adjust parameters based on marker quality**
                if self.live_tracking_optimized:
                    self.tracking_fps = 60
                    self.feature_threshold = 20    # More sensitive
                    self.quality_threshold = 0.015 # More sensitive
                    self.stability_frames = 2      # Faster response
                    logger.info(f"Live tracking optimization enabled for {experience_slug}")
                else:
                    self.tracking_fps = 30
                    self.feature_threshold = 30    # More stable
                    self.quality_threshold = 0.025 # More stable
                    self.stability_frames = 5      # More stable
                    logger.info(f"Conservative tracking settings for {experience_slug}")
        except Exception as e:
            logger.warning(f"Could not optimize tracker for {experience_slug}: {str(e)}")
            self.live_tracking_optimized = False
            self.target_accuracy = 0.95
        
        # Try to initialize the advanced tracker (unchanged)
        try:
            from .advanced_ar_tracking import integrate_with_django_views
            self.advanced_tracker = integrate_with_django_views(experience_slug)
            self.advanced_tracking_available = True
            logger.info(f"Advanced tracker initialized for {experience_slug}")
        except Exception as e:
            logger.warning(f"Advanced tracker not available for {experience_slug}: {str(e)}")
            self.advanced_tracking_available = False
            
        # Initialize channel layer (unchanged)
        try:
            self.channel_layer = get_channel_layer()
        except Exception as e:
            logger.warning(f"Channel layer not available for {experience_slug}: {str(e)}")
    
    def start_tracking(self, image_callback=None):
        """Start real-time tracking with Stories AR optimization"""
        if self.is_tracking:
            return
        self.is_tracking = True
        self.tracking_thread = threading.Thread(
            target=self._tracking_loop,
            args=(image_callback,),
            daemon=True
        )
        self.tracking_thread.start()
        
        optimization_status = "Live Tracking" if self.live_tracking_optimized else "Enhanced"
        logger.info(f"Started {optimization_status} tracking for {self.experience_slug} - Target: {self.target_accuracy*100}% @ {self.tracking_fps}fps")
    
    def stop_tracking(self):
        """Stop real-time tracking"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        logger.info(f"Stopped real-time tracking for {self.experience_slug}")
    
    def _tracking_loop(self, image_callback):
        """Enhanced tracking loop with Stories AR optimization"""
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
            
            # **NEW: Stories AR Optimized ORB Detector**
            if self.live_tracking_optimized:
                # Ultra-sensitive settings for live tracking
                orb = cv2.ORB_create(
                    nfeatures=2000,              # High feature count
                    scaleFactor=1.1,             # Fine scale steps
                    nlevels=12,                  # More pyramid levels
                    edgeThreshold=3,             # Ultra-sensitive edges
                    firstLevel=0,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=21,                # Smaller patches for precision
                    fastThreshold=3              # Ultra-sensitive features
                )
            else:
                # Conservative settings for stable tracking
                orb = cv2.ORB_create(
                    nfeatures=1500,              # Moderate feature count
                    scaleFactor=1.15,            # Moderate scale steps
                    nlevels=10,                  # Fewer levels for stability
                    edgeThreshold=8,             # Less sensitive edges
                    firstLevel=0,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=31,                # Larger patches for stability
                    fastThreshold=8              # Less sensitive features
                )
            
            kp1, des1 = orb.detectAndCompute(marker_img, None)
            if des1 is None:
                logger.error(f"Cannot compute descriptors for {self.experience_slug}")
                return
            
            # **NEW: Enhanced Matcher for Stories AR**
            if self.live_tracking_optimized:
                # More aggressive matching for live tracking
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                distance_threshold = 25  # More sensitive
            else:
                # Conservative matching for stability
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                distance_threshold = 30  # More stable
            
            # **NEW: Tracking state management**
            consecutive_detections = 0
            consecutive_losses = 0
            stable_tracking = False
            
            # Calculate frame delay for target FPS
            frame_delay = 1.0 / self.tracking_fps
            
            while self.is_tracking:
                # Get frame from callback
                if image_callback:
                    frame = image_callback()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    
                    # Use advanced tracking if available (unchanged logic)
                    if self.advanced_tracking_available:
                        try:
                            tracking_result = self.advanced_tracker.track_frame(frame)
                            
                            # Send advanced tracking result
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
                                            'fps': self.tracking_fps,
                                            'optimized': self.live_tracking_optimized,
                                            'timestamp': time.time(),
                                            'advanced': True
                                        }
                                    }
                                )
                            continue
                        except Exception as e:
                            logger.error(f"Error in advanced tracking: {str(e)}")
                    
                    # **ENHANCED: Stories AR Optimized Tracking**
                    start_time = time.time()
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndCompute(gray, None)
                    
                    detected = False
                    confidence = 0.0
                    match_count = 0
                    
                    if des2 is not None:
                        # Match features with optimized threshold
                        matches = bf.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = [m for m in matches if m.distance < distance_threshold]
                        match_count = len(good_matches)
                        
                        # **NEW: Stories AR Detection Logic**
                        if match_count >= self.feature_threshold:
                            confidence = min(1.0, match_count / 150)  # Normalized confidence
                            
                            # **Quality filtering for live tracking**
                            if self.live_tracking_optimized:
                                # Additional quality check for live tracking
                                high_quality_matches = [m for m in good_matches if m.distance < 20]
                                if len(high_quality_matches) >= self.feature_threshold * 0.6:
                                    detected = True
                                    consecutive_detections += 1
                                    consecutive_losses = 0
                            else:
                                # Standard detection for stable tracking
                                detected = True
                                consecutive_detections += 1
                                consecutive_losses = 0
                        else:
                            consecutive_losses += 1
                            consecutive_detections = 0
                    else:
                        consecutive_losses += 1
                        consecutive_detections = 0
                    
                    # **NEW: Stability Management**
                    if consecutive_detections >= self.stability_frames:
                        stable_tracking = True
                    elif consecutive_losses >= self.stability_frames:
                        stable_tracking = False
                    
                    # **Calculate performance metrics**
                    processing_time = (time.time() - start_time) * 1000  # ms
                    actual_fps = 1.0 / max(processing_time / 1000, frame_delay)
                    
                    # Send enhanced tracking result
                    if self.channel_layer:
                        async_to_sync(self.channel_layer.group_send)(
                            f'ar_tracking_{self.experience_slug}',
                            {
                                'type': 'tracking_update',
                                'tracking': {
                                    'detected': detected,
                                    'stable': stable_tracking,
                                    'confidence': confidence,
                                    'matches': match_count,
                                    'target_accuracy': self.target_accuracy,
                                    'target_fps': self.tracking_fps,
                                    'actual_fps': round(actual_fps, 1),
                                    'processing_time_ms': round(processing_time, 2),
                                    'optimized': self.live_tracking_optimized,
                                    'consecutive_detections': consecutive_detections,
                                    'consecutive_losses': consecutive_losses,
                                    'timestamp': time.time(),
                                    'advanced': False
                                }
                            }
                        )
                
                # **Dynamic frame rate control**
                time.sleep(frame_delay)
                
        except Exception as e:
            logger.error(f"Error in Stories AR tracking loop: {str(e)}")
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
    """Enhanced upload view with Stories AR live tracking optimization"""
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
                    # **NEW: Stories AR Live Tracking Optimized Pipeline**
                    if experience.image:
                        marker_dir = ensure_marker_directory(experience.slug)
                        mind_file_path = marker_dir / f"{experience.slug}.mind"
                        
                        # **Step 1: ALWAYS validate with Stories AR criteria first**
                        try:
                            validation_results = validate_marker_quality_stories_ar(experience.image.path)
                            logger.info(f"Stories AR validation for {experience.slug}: {validation_results['quality_level']}")
                            
                            # **Priority 1: Live tracking ready images**
                            if validation_results.get('live_tracking_ready', False):
                                logger.info(f"Image already optimized for live tracking: {experience.slug}")
                                if generate_mindar_marker(experience.image.path, str(mind_file_path)):
                                    marker_generated = True
                                    score = validation_results['score']
                                    features = validation_results['stories_ar_metrics']['feature_count']
                                    marker_message = f"🚀 Live tracking ready! Score: {score:.2f}, Features: {features}"
                                    logger.info(f"Stories AR optimized marker generated for {experience.slug}")
                                
                            # **Priority 2: Good images that need enhancement**
                            elif validation_results['quality_level'] in ['stories_ar_good', 'stories_ar_fair']:
                                logger.info(f"Enhancing image for live tracking: {experience.slug}")
                                if generate_mindar_marker_with_preprocessing(experience.image.path, str(mind_file_path)):
                                    marker_generated = True
                                    marker_message = f"🔧 Enhanced for live tracking (was {validation_results['quality_level']})"
                                    logger.info(f"Enhanced marker generated for {experience.slug}")
                                
                            # **Priority 3: Poor images need automated processing**
                            elif validation_results['quality_level'] in ['stories_ar_poor', 'critical']:
                                logger.warning(f"Poor quality image, trying automated processing: {experience.slug}")
                                
                                # Try automated approach for difficult images
                                if generate_mindar_marker_automated(experience.image.path, str(mind_file_path)):
                                    marker_generated = True
                                    image_type = classify_image_type(experience.image.path)
                                    marker_message = f"⚠️ Automated processing applied ({image_type}) - may have tracking issues"
                                    logger.info(f"Automated marker generated for poor quality image: {experience.slug}")
                                
                                # Last resort: try basic Stories AR method
                                elif generate_mindar_marker(experience.image.path, str(mind_file_path)):
                                    marker_generated = True
                                    marker_message = "📱 Basic Stories AR processing (limited tracking performance)"
                                    logger.info(f"Basic Stories AR marker generated: {experience.slug}")
                                
                            # **Fallback chain if validation-based approach fails**
                            if not marker_generated:
                                logger.warning(f"Validation-based approach failed for {experience.slug}, trying fallback chain")
                                
                                # Fallback 1: Manual preprocessing
                                if generate_mindar_marker_with_preprocessing(experience.image.path, str(mind_file_path)):
                                    marker_generated = True
                                    marker_message = "🛠️ Manual preprocessing applied (fallback)"
                                    logger.info(f"Manual preprocessed marker generated: {experience.slug}")
                                
                                # Fallback 2: Original Stories AR method
                                elif generate_mindar_marker(experience.image.path, str(mind_file_path)):
                                    marker_generated = True
                                    marker_message = "📱 Stories AR methodology applied (fallback)"
                                    logger.info(f"Stories AR fallback marker generated: {experience.slug}")
                                
                                # Fallback 3: ArUco emergency fallback
                                else:
                                    logger.error(f"All image processing failed for {experience.slug}, generating ArUco fallback")
                                    marker_id = hash(experience.slug) % 50
                                    if generate_aruco_mindar_marker(marker_id, str(mind_file_path)):
                                        marker_generated = True
                                        marker_message = "🎯 ArUco fallback generated (basic tracking only)"
                                        logger.info(f"ArUco emergency fallback generated: {experience.slug}")
                                    else:
                                        marker_message = "❌ All marker generation methods failed"
                                        logger.error(f"Complete failure for {experience.slug}")
                            
                            # **Add validation info to success message**
                            if marker_generated and validation_results.get('live_tracking_ready'):
                                stories_metrics = validation_results.get('stories_ar_metrics', {})
                                fps_target = stories_metrics.get('target_fps', 30)
                                marker_message += f" | Target FPS: {fps_target}"
                                
                        except Exception as validation_error:
                            logger.error(f"Stories AR validation failed for {experience.slug}: {str(validation_error)}")
                            
                            # **Emergency fallback when validation fails**
                            logger.warning(f"Falling back to original automated pipeline for {experience.slug}")
                            
                            if generate_mindar_marker_automated(experience.image.path, str(mind_file_path)):
                                marker_generated = True
                                marker_message = "🔧 Emergency automated processing (validation failed)"
                            else:
                                # Final emergency fallback
                                marker_id = hash(experience.slug) % 50
                                if generate_aruco_mindar_marker(marker_id, str(mind_file_path)):
                                    marker_generated = True
                                    marker_message = "🔧 Emergency ArUco marker generated"
                                else:
                                    marker_message = f"❌ Critical error: {str(validation_error)}"
                    
                    else:
                        marker_message = "⚠️ No image provided for marker generation"

                experience.marker_generated = marker_generated
                experience.save(update_fields=['nft_iset_file', 'marker_generated'])

                # QR Code generation (unchanged)
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

                # **Enhanced success/error messages with Stories AR feedback**
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
        from .advanced_ar_tracking import AdvancedARTracker, integrate_with_django_views
        advanced_tracker = integrate_with_django_views(slug)
        advanced_tracking_available = True
        # ... advanced tracking code
    except ImportError:
        logger.warning("Advanced tracking dependencies not available, using basic tracking")
        advanced_tracking_available = False
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
    
    # **NEW: Stories AR Validation and Dynamic Configuration**
    stories_ar_validation = None
    stories_ar_metrics = {}
    live_tracking_ready = False
    
    if experience.image:
        try:
            stories_ar_validation = validate_marker_quality_stories_ar(experience.image.path)
            stories_ar_metrics = stories_ar_validation.get('stories_ar_metrics', {})
            live_tracking_ready = stories_ar_validation.get('live_tracking_ready', False)
            logger.info(f"Stories AR validation for {slug}: {stories_ar_validation['quality_level']}")
        except Exception as e:
            logger.warning(f"Stories AR validation failed for {slug}: {str(e)}")
    
    # **UPDATED: Dynamic MindAR Configuration Based on Stories AR Validation**
    mindar_config = {
        'maxTrack': 1,
        'showStats': settings.DEBUG,  # Keep existing debug functionality
        'uiLoading': 'no',
        'uiError': 'no', 
        'uiScanning': 'no',
        'autoStart': True,
        # **Default enhanced parameters (existing functionality preserved)**
        'filterMinCF': 0.0001,      
        'filterBeta': 0.001,        
        'missTolerance': 1,         
        'warmupTolerance': 1,       
        'targetAccuracy': 0.95      
    }
    
    # **NEW: Apply Stories AR Optimizations Based on Validation**
    if stories_ar_validation:
        if live_tracking_ready:
            # Ultra-optimized settings for live tracking ready markers
            mindar_config.update({
                'filterMinCF': 0.00005,      # Ultra-sensitive
                'filterBeta': 0.0005,        # Ultra-responsive
                'missTolerance': 0,          # Instant detection
                'warmupTolerance': 0,        # No warmup delay
                'targetAccuracy': 0.97,      # Highest precision
                'maxFPS': stories_ar_metrics.get('target_fps', 60)
            })
            logger.info(f"Applied live tracking optimization for {slug}")
            
        elif stories_ar_validation['quality_level'] in ['stories_ar_poor', 'critical']:
            # Conservative settings for poor quality markers
            mindar_config.update({
                'filterMinCF': 0.0002,       # Less sensitive for stability
                'filterBeta': 0.002,         # Slower but smoother
                'missTolerance': 2,          # Allow brief losses
                'warmupTolerance': 2,        # Brief warmup for stability
                'targetAccuracy': 0.93,      # Lower precision for stability
                'maxFPS': 30                 # Conservative FPS
            })
            logger.info(f"Applied conservative tracking for lower quality marker: {slug}")
    
    # Build URLs (unchanged)
    video_url = experience.video.url if experience.video else None
    marker_image_url = experience.image.url if experience.image else None
    
    # Validate marker quality (with error handling) - unchanged
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
    
    # Build WebSocket URL (with error handling) - unchanged
    websocket_url = None
    websocket_available = False
    try:
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        websocket_available = True
        protocol = 'wss' if request.is_secure() else 'ws'
        websocket_url = f"{protocol}://{request.get_host()}/ws/ar/{slug}/"
    except Exception as e:
        logger.warning(f"Channel layer not available for {slug}, WebSocket disabled: {str(e)}")
        websocket_available = False
    
    # Initialize real-time tracker for this experience - unchanged
    tracker = None
    tracker_available = False
    try:
        if websocket_available:
            tracker = get_tracker(slug)
            tracker_available = True
    except Exception as e:
        logger.warning(f"Cannot initialize tracker for {slug}: {str(e)}")
        tracker_available = False
    
    # Check if we need to regenerate the marker - unchanged
    need_to_regenerate = False
    if experience.image and (not marker_files_exist or mind_file_size < 1000):
        need_to_regenerate = True
        logger.warning(f"Marker file for {slug} needs to be regenerated (exists: {marker_files_exist}, size: {mind_file_size})")
    
    # Prepare advanced tracking information for the template - unchanged
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
    
    # **UPDATED: Enhanced Instructions Based on Stories AR Optimization**
    instructions = {
        'setup': 'Allow camera access when prompted by your browser',
        'usage': 'Point your camera at the uploaded marker image',
        'distance': '15-50cm away from marker for optimal tracking' if live_tracking_ready else 'Hold your device 20-60cm away from the marker',
        'lighting': 'Ensure bright, even lighting for 60fps performance' if live_tracking_ready else 'Ensure good lighting for better tracking',
        'stability': 'Smooth movements - ultra-sensitive tracking active' if live_tracking_ready else 'Keep the marker clearly visible and steady',
        'technology': 'Powered by Stories AR Live Tracking' if live_tracking_ready else ('Powered by Advanced Multi-Method Tracking' if advanced_tracking_available else 'Powered by Enhanced MindAR'),
        'accuracy_target': f'Target accuracy: {int(mindar_config["targetAccuracy"]*100)}% for professional performance'
    }
    
    # **NEW: Add FPS target to instructions if available**
    if 'maxFPS' in mindar_config:
        instructions['fps_target'] = f'Target FPS: {mindar_config["maxFPS"]} for real-time experience'
    
    context = {
        "experience": experience,
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "base_url": base_url,
        "title": experience.title,
        "video_url": video_url,
        "marker_image_url": marker_image_url,
        "timestamp": int(time.time()),
        "tracking_method": "Stories AR Live Tracking" if live_tracking_ready else ("Advanced Multi-Method" if advanced_tracking_available else "Enhanced MindAR"),
        "mindar_config": mindar_config,
        "mindar_config_json": json.dumps(mindar_config),
        "tracking_validation": tracking_validation,
        "websocket_url": websocket_url,
        "websocket_available": websocket_available,
        "target_accuracy": mindar_config['targetAccuracy'],  # **Updated to use dynamic value**
        "need_to_regenerate": need_to_regenerate,
        "mind_file_size": mind_file_size,
        "advanced_tracking": advanced_tracking_info,
        "instructions": instructions,  # **Updated instructions**
        
        # **NEW: Stories AR Context**
        "stories_ar_optimization": {
            "enabled": stories_ar_validation is not None,
            "live_tracking_ready": live_tracking_ready,
            "quality_level": stories_ar_validation.get('quality_level') if stories_ar_validation else None,
            "metrics": stories_ar_metrics,
            "ultra_sensitive": mindar_config.get('filterMinCF', 0) <= 0.0001,
            "instant_detection": mindar_config.get('warmupTolerance', 1) == 0,
            "target_fps": mindar_config.get('maxFPS', 30)
        },
        
        "debug": {
            'mind_file_path': mind_file_path,
            'mind_file_exists': marker_files_exist,
            'mind_file_size': mind_file_size,
            'media_root': str(settings.MEDIA_ROOT),
            'slug': slug,
            'tracking_issues': tracking_validation['issues'] if tracking_validation else [],
            'tracking_recommendations': tracking_validation['recommendations'] if tracking_validation else [],
            'target_accuracy': mindar_config['targetAccuracy'],  # **Updated**
            'websocket_available': websocket_available,
            'tracker_available': tracker_available,
            'need_to_regenerate': need_to_regenerate,
            'advanced_tracking_available': advanced_tracking_available,
            'advanced_tracking_results': advanced_tracking_results if advanced_tracking_available else None,
            # **NEW: Stories AR debug info**
            'stories_ar_validation': stories_ar_validation if settings.DEBUG else None,
            'live_tracking_ready': live_tracking_ready,
            'mindar_config_optimized': live_tracking_ready
        } if settings.DEBUG else {},
        
        # Tracking function connections - unchanged structure, updated values
        "tracking_functions": {
            "real_time_tracker": {
                "active": tracker.is_tracking if tracker else False,
                "slug": slug,
                "target_accuracy": mindar_config['targetAccuracy'],  # **Updated**
                "available": tracker_available
            },
            "websocket_consumer": {
                "connected": False,
                "group_name": f'ar_tracking_{slug}',
                "target_accuracy": mindar_config['targetAccuracy'],  # **Updated**
                "available": websocket_available
            },
            "marker_validation": {
                "valid": tracking_validation['valid'] if tracking_validation else False,
                "quality_score": tracking_validation['quality_score'] if tracking_validation else 0,
                "target_accuracy": mindar_config['targetAccuracy']  # **Updated**
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

def generate_optimal_marker_for_tracking(experience, image_path, mind_file_path):
    """Generate marker optimized for Stories AR-level live tracking"""
    
    # Step 1: Always try Stories AR method FIRST
    if generate_mindar_marker(image_path, str(mind_file_path)):
        return True, "Stories AR optimized marker"
    
    # Step 2: Enhanced preprocessing for difficult images
    if generate_mindar_marker_with_preprocessing(image_path, str(mind_file_path)):
        return True, "Enhanced preprocessing applied"
    
    # Step 3: Automated as last resort
    return False, "Failed to generate trackable marker"
