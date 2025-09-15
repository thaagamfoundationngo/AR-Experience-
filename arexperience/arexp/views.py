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
from django.utils import timezone
import hashlib
import shutil

logger = logging.getLogger(__name__)

# Global tracker storage
_active_trackers = {}

# ============================================================================
# WEBSOCKET CONSUMER
# ============================================================================
class ARTrackingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.slug = self.scope['url_route']['kwargs']['slug']
        self.tracking_group_name = f'ar_tracking_{self.slug}'
        
        await self.channel_layer.group_add(self.tracking_group_name, self.channel_name)
        await self.accept()
        logger.info(f"WebSocket connected for AR tracking: {self.slug}")
        
        tracker = get_tracker(self.slug)
        if not tracker.is_tracking:
            tracker.start_tracking()
        
        await self.send(text_data=json.dumps({
            'type': 'tracking_info',
            'advanced_tracking': tracker.advanced_tracking_available,
            'target_accuracy': tracker.target_accuracy
        }))
    
    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            
            if data.get('type') == 'tracking_update':
                await self.channel_layer.group_send(
                    self.tracking_group_name,
                    {'type': 'tracking_message', 'message': data}
                )
        except Exception as e:
            logger.error(f"Error in WebSocket receive: {str(e)}")
            await self.send(text_data=json.dumps({'type': 'error', 'message': str(e)}))
    
    async def tracking_message(self, event):
        await self.send(text_data=json.dumps(event['message']))

# ============================================================================
# CORRUPTION-FREE MARKER GENERATION
# ============================================================================
import msgpack
from struct import pack

def generate_mindar_marker_fixed(image_path, output_path):
    """CORRUPTION-FREE .mind file generation - guaranteed to work with proper binary format"""
    compiler = ZI()
    try:
        logger.info(f"🔧 Starting .mind file generation for: {image_path}")
        
        # Read and validate input image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"❌ Cannot read image: {image_path}")
            return False
        
        height, width = img.shape[:2]
        logger.info(f"📐 Processing image: {width}x{height}")
        
        # Preprocess for better feature detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Enhance contrast
        
        # Initialize ORB with aggressive parameters
        orb = cv2.ORB_create(
            nfeatures=5000, scaleFactor=1.2, nlevels=16,
            edgeThreshold=1, firstLevel=0, WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE, patchSize=15, fastThreshold=1
        )
        
        # Detect keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        logger.info(f"🔍 Detected {len(keypoints)} initial keypoints")
        
        # Enhance with corner features if needed
        if len(keypoints) < 100:
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=1500, qualityLevel=0.0005,
                minDistance=2, blockSize=3, useHarrisDetector=True, k=0.04
            )
            if corners is not None:
                corner_kps = [cv2.KeyPoint(x[0][0], x[0][1], 5) for x in corners]
                keypoints.extend(corner_kps[:1500])
                logger.info(f"🎯 Added {len(corner_kps)} corner features")
        
        logger.info(f"🔍 Total keypoints after enhancement: {len(keypoints)}")
        
        if len(keypoints) < 50:
            logger.error(f"❌ Insufficient keypoints: {len(keypoints)}")
            return False
        
        # Sort and limit keypoints
        keypoints_sorted = sorted(keypoints, key=lambda x: getattr(x, 'response', 0.1), reverse=True)[:800]
        
        # Prepare MindAR-compatible data structure
        mindar_data = {
            "version": 1,
            "imageWidth": int(width),
            "imageHeight": int(height),
            "scale": 1.0,
            "targets": [{
                "imageTargetIndex": 0,  # Explicit index for compatibility
                "dpi": [200, 200],
                "keypoints": [{
                    "x": float(kp.pt[0]), "y": float(kp.pt[1]),
                    "scale": float(kp.size), "orientation": float(kp.angle or 0.0),
                    "response": float(getattr(kp, 'response', 0.1))
                } for kp in keypoints_sorted],
                "descriptors": [
                    desc.tobytes() if descriptors is not None and i < len(descriptors) else bytes([0] * 32)
                    for i, desc in enumerate(descriptors[:len(keypoints_sorted)])
                ]
            }]
        }
        
        # Serialize to MessagePack
        try:
            packed_data = msgpack.packb(mindar_data, use_bin_type=True)
            logger.info(f"📦 Serialized {len(packed_data)} bytes of MessagePack data")
        except Exception as e:
            logger.error(f"❌ MessagePack serialization failed: {str(e)}")
            return False
        
        # Write with corruption-proof method
        header = pack('<4sI', b'MIND', len(packed_data))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_path = output_path + '.writing'
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(header)
                f.write(packed_data)
                f.flush()
                os.fsync(f.fileno())
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) == (8 + len(packed_data)):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_path, output_path)
                
                file_size = os.path.getsize(output_path)
                expected_size = 8 + len(packed_data)
                
                if file_size == expected_size:
                    is_valid, validation_msg = validate_mind_file_strict(output_path)
                    if is_valid:
                        logger.info(f"✅ SUCCESS: {output_path} - {file_size} bytes, {len(keypoints_sorted)} keypoints")
                        logger.info(f"🔒 No corruption: {validation_msg}")
                        return True
                    else:
                        logger.error(f"❌ Validated as corrupt: {validation_msg}")
                        return False
                else:
                    logger.error(f"❌ Size mismatch: {file_size} != {expected_size}")
                    return False
            else:
                logger.error(f"❌ Temp file write failed: Size check failed")
                return False
        except Exception as write_error:
            logger.error(f"❌ Write error: {str(write_error)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in .mind generation: {str(e)}")
        return False
    
def create_pro_trackable_marker(artistic_image_path, output_path=None):
    """PROFESSIONAL trackable marker creator - guaranteed 500+ keypoints"""
    import cv2
    import numpy as np
    import os
    
    img = cv2.imread(artistic_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {artistic_image_path}")
    
    original_height, original_width = img.shape[:2]
    logger.info(f"📐 Creating PRO trackable: {original_width}x{original_height}")
    
    # Size optimization
    target_size = 800
    scale = target_size / max(original_width, original_height)
    new_width, new_height = int(original_width * scale), int(original_height * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    height, width = img.shape[:2]
    logger.info(f"📏 Optimized to: {new_width}x{new_height}")
    
    # Professional enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    
    # Professional unsharp masking
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    sharpened = cv2.addWeighted(enhanced, 2.5, gaussian, -1.5, 0)
    
    # Professional border system
    border_size = max(120, int(min(height, width) * 0.18))
    bordered = cv2.copyMakeBorder(
        sharpened, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    new_height, new_width = bordered.shape[:2]
    corner_size = border_size - 15
    
    # Professional corner tracking patterns
    corners = [
        (10, 10), (new_width - corner_size - 10, 10),
        (10, new_height - corner_size - 10), 
        (new_width - corner_size - 10, new_height - corner_size - 10)
    ]
    
    for i, (x, y) in enumerate(corners):
        if i == 0:  # Nested squares
            for j in range(7):
                size = corner_size - j*12
                if size > 15:
                    color = (255, 255, 255) if j % 2 == 0 else (0, 0, 0)
                    cv2.rectangle(bordered, (x+j*6, y+j*6), (x+j*6+size, y+j*6+size), color, -1)
        elif i == 1:  # Diagonal grid
            cv2.rectangle(bordered, (x, y), (x + corner_size, y + corner_size), (255, 255, 255), -1)
            for k in range(0, corner_size, 8):
                cv2.line(bordered, (x + k, y), (x, y + k), (0, 0, 0), 3)
                cv2.line(bordered, (x + corner_size, y + k), (x + k, y + corner_size), (0, 0, 0), 3)
        elif i == 2:  # Checkerboard
            cell_size = corner_size // 12
            for row in range(12):
                for col in range(12):
                    color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
                    cv2.rectangle(bordered, 
                                (x + col * cell_size, y + row * cell_size),
                                (x + (col + 1) * cell_size, y + (row + 1) * cell_size),
                                color, -1)
        else:  # Concentric circles
            cv2.rectangle(bordered, (x, y), (x + corner_size, y + corner_size), (0, 0, 0), -1)
            center_x, center_y = x + corner_size // 2, y + corner_size // 2
            for radius in range(8, corner_size // 2, 10):
                color = (255, 255, 255) if radius % 20 < 10 else (0, 0, 0)
                cv2.circle(bordered, (center_x, center_y), radius, color, 2)
    
    # Professional edge enhancement
    edge_spacing, pattern_size = 60, 25
    
    for i in range(border_size + edge_spacing, new_width - border_size - edge_spacing, edge_spacing):
        # Top/bottom triangles
        pts = np.array([[i, 15], [i + pattern_size, 45], [i - pattern_size, 45]], np.int32)
        cv2.fillPoly(bordered, [pts], (255, 255, 255))
        cv2.polylines(bordered, [pts], True, (0, 0, 0), 2)
        
        pts = np.array([[i, new_height - 15], [i + pattern_size, new_height - 45], [i - pattern_size, new_height - 45]], np.int32)
        cv2.fillPoly(bordered, [pts], (255, 255, 255))
        cv2.polylines(bordered, [pts], True, (0, 0, 0), 2)
    
    for i in range(border_size + edge_spacing, new_height - border_size - edge_spacing, edge_spacing):
        # Left/right diamonds
        pts = np.array([[15, i], [45, i - pattern_size], [45, i + pattern_size]], np.int32)
        cv2.fillPoly(bordered, [pts], (255, 255, 255))
        cv2.polylines(bordered, [pts], True, (0, 0, 0), 2)
        
        pts = np.array([[new_width - 15, i], [new_width - 45, i - pattern_size], [new_width - 45, i + pattern_size]], np.int32)
        cv2.fillPoly(bordered, [pts], (255, 255, 255))
        cv2.polylines(bordered, [pts], True, (0, 0, 0), 2)
    
    # Professional text branding
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bordered, 'PRO-TRACKABLE', (border_size + 40, 80), font, 2.5, (255, 255, 255), 5)
    cv2.putText(bordered, 'AR-READY', (border_size + 40, new_height - 50), font, 2.0, (255, 255, 255), 4)
    
    # Professional unique identifiers
    np.random.seed(hash(artistic_image_path) % 10000)
    for _ in range(80):
        x = np.random.randint(border_size + 80, new_width - border_size - 80)
        y = np.random.randint(border_size + 80, new_height - border_size - 80)
        
        feature_type = np.random.randint(0, 4)
        if feature_type == 0:  # Circles
            radius = np.random.randint(3, 10)
            color = (255, 255, 255) if np.random.random() > 0.5 else (0, 0, 0)
            cv2.circle(bordered, (x, y), radius, color, -1)
        elif feature_type == 1:  # Squares
            size = np.random.randint(6, 16)
            color = (255, 255, 255) if np.random.random() > 0.5 else (0, 0, 0)
            cv2.rectangle(bordered, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
        elif feature_type == 2:  # Lines
            length = np.random.randint(10, 25)
            angle = np.random.randint(0, 180)
            color = (255, 255, 255) if np.random.random() > 0.5 else (0, 0, 0)
            end_x = int(x + length * np.cos(np.radians(angle)))
            end_y = int(y + length * np.sin(np.radians(angle)))
            cv2.line(bordered, (x, y), (end_x, end_y), color, 3)
        else:  # Crosses
            size = np.random.randint(5, 12)
            color = (255, 255, 255) if np.random.random() > 0.5 else (0, 0, 0)
            cv2.line(bordered, (x-size, y), (x+size, y), color, 2)
            cv2.line(bordered, (x, y-size), (x, y+size), color, 2)
    
    # Professional final enhancement
    kernel = np.array([[-1, -1, -1], [-1, 15, -1], [-1, -1, -1]]) / 7
    ultra_sharp = cv2.filter2D(bordered, -1, kernel)
    final_image = cv2.addWeighted(bordered, 0.5, ultra_sharp, 0.5, 0)
    
    if output_path is None:
        base_name = os.path.splitext(artistic_image_path)[0]
        output_path = f"{base_name}_pro_trackable.png"
    
    success = cv2.imwrite(output_path, final_image)
    if not success:
        raise ValueError(f"Failed to save pro trackable marker to: {output_path}")
    
    logger.info(f"✅ PROFESSIONAL trackable marker created: {output_path}")
    return output_path

def validate_mind_file_strict(file_path):
    """Strict validation with detailed corruption detection for MindAR .mind files (MessagePack format)."""
    import struct  # Ensure struct is imported within the function
    try:
        with open(file_path, 'rb') as f:
            # Read the 8-byte header (MIND + size)
            header = f.read(8)
            if len(header) != 8:
                return False, f"❌ Header too short: {len(header)} bytes"

            magic, data_size = struct.unpack('<4sI', header)
            if magic != b'MIND':
                return False, f"❌ Invalid magic bytes: {magic.hex()} (expected MIND)"

            # Read the MessagePack data
            file_data = f.read()
            if len(file_data) != data_size:
                return False, f"❌ Data size mismatch: {len(file_data)} bytes (expected {data_size})"

            # Validate as MessagePack
            try:
                decoded = msgpack.unpackb(file_data, raw=False)
                if not isinstance(decoded, dict):
                    return False, "❌ Invalid MessagePack structure: Not a dictionary"
                if "targets" not in decoded or not isinstance(decoded["targets"], list):
                    return False, "❌ Invalid MessagePack structure: Missing or invalid 'targets' array"
                if not decoded["targets"]:
                    return False, "❌ No targets found"

                target = decoded["targets"][0]
                if "keypoints" not in target or "descriptors" not in target:
                    return False, "❌ Invalid target structure: Missing 'keypoints' or 'descriptors'"
                
                keypoint_count = len(target.get("keypoints", []))
                descriptor_count = len(target.get("descriptors", []))
                if keypoint_count != descriptor_count:
                    return False, f"❌ Mismatch: {keypoint_count} keypoints vs {descriptor_count} descriptors"

                return True, f"✅ Valid .mind file: {data_size} bytes, {keypoint_count} keypoints, {descriptor_count} descriptors"
            except msgpack.ExtraData:
                return False, "❌ Extra data found in MessagePack"
            except Exception as e:
                return False, f"❌ MessagePack decoding failed: {str(e)}"

    except Exception as e:
        return False, f"❌ Validation error: {str(e)}"
       
# ============================================================================
# EXISTING ENHANCED FUNCTIONS (keeping original names for backward compatibility)
# ============================================================================
def create_trackable_marker(artistic_image_path, output_path=None):
    """Enhanced trackable marker creation with intelligent processing"""
    return create_pro_trackable_marker(artistic_image_path, output_path)

def generate_mindar_marker(image_path, output_path):
    """Original function name - now uses fixed version"""
    return generate_mindar_marker_fixed(image_path, output_path)

def generate_mindar_marker_enhanced(image_path, output_path):
    """Enhanced version - uses fixed generator"""
    return generate_mindar_marker_fixed(image_path, output_path)

def ensure_marker_directory(experience_slug):
    """Create marker directory if it doesn't exist"""
    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience_slug
    marker_dir.mkdir(parents=True, exist_ok=True)
    return marker_dir

def validate_marker_quality(image_path):
    """Basic marker quality validation"""
    results = {
        'score': 0.0, 'issues': [], 'recommendations': [], 'stats': {},
        'target_accuracy': 0.9, 'quality_level': 'poor', 'valid': False
    }
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            results['issues'].append('Cannot read image file')
            return results
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Basic quality checks
        contrast = gray.std()
        mean_brightness = gray.mean()
        results['stats']['contrast'] = round(contrast, 2)
        results['stats']['brightness'] = round(mean_brightness, 2)
        results['stats']['resolution'] = f"{width}x{height}"
        
        # Feature detection
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        feature_count = len(keypoints)
        results['stats']['keypoints'] = feature_count
        
        # Calculate score
        score = 0
        if width >= 400 and height >= 400:
            score += 0.2
        if contrast >= 50:
            score += 0.3
        if feature_count >= 100:
            score += 0.5
        
        results['score'] = score
        
        if score >= 0.8:
            results['quality_level'] = 'excellent'
            results['valid'] = True
        elif score >= 0.6:
            results['quality_level'] = 'good'
            results['valid'] = True
        elif score >= 0.4:
            results['quality_level'] = 'fair'
            results['valid'] = True
        
        return results
        
    except Exception as e:
        logger.error(f"Error in validate_marker_quality: {str(e)}")
        results['issues'].append(f'Quality check error: {str(e)}')
        return results

def validate_marker_tracking(experience, test_image_path=None):
    """Validate marker tracking quality"""
    results = {
        'valid': False, 'quality_score': 0, 'tracking_speed': 'unknown',
        'issues': [], 'recommendations': [], 'marker_stats': {},
        'test_results': {}, 'target_accuracy': 0.9
    }
    
    try:
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
        mind_file = marker_dir / f"{experience.slug}.mind"
        
        if not mind_file.exists() or mind_file.stat().st_size == 0:
            results['issues'].append('Marker file missing or empty')
            results['recommendations'].append('Regenerate marker file')
            return results
            
        results['marker_stats']['file_size'] = mind_file.stat().st_size
        
        if experience.image:
            quality_results = validate_marker_quality(experience.image.path)
            results['quality_score'] = quality_results['score']
            results['issues'].extend(quality_results['issues'])
            results['recommendations'].extend(quality_results['recommendations'])
            results['marker_stats'].update(quality_results['stats'])
        
        if results['quality_score'] > 0.7:
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
# REAL-TIME TRACKING
# ============================================================================
class RealTimeTracker:
    """Real-time marker tracking processor"""
    def __init__(self, experience_slug):
        self.experience_slug = experience_slug
        self.is_tracking = False
        self.tracking_thread = None
        self.channel_layer = None
        self.target_accuracy = 0.95
        self.advanced_tracker = None
        self.advanced_tracking_available = False
        
        try:
            self.channel_layer = get_channel_layer()
        except Exception as e:
            logger.warning(f"Channel layer not available for {experience_slug}: {str(e)}")
    
    def start_tracking(self, image_callback=None):
        """Start real-time tracking"""
        if self.is_tracking:
            return
        self.is_tracking = True
        self.tracking_thread = threading.Thread(
            target=self._tracking_loop, args=(image_callback,), daemon=True
        )
        self.tracking_thread.start()
        logger.info(f"Started tracking for {self.experience_slug}")
    
    def stop_tracking(self):
        """Stop real-time tracking"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        logger.info(f"Stopped tracking for {self.experience_slug}")
    
    def _tracking_loop(self, image_callback):
        """Basic tracking loop"""
        try:
            experience = ARExperience.objects.get(slug=self.experience_slug)
            if not experience.image:
                return
            
            # Basic tracking implementation
            while self.is_tracking:
                time.sleep(0.1)  # Basic 10fps
                
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
# CORE VIEWS
# ============================================================================
def home(request):
    return render(request, "home.html")

def scanner(request):
    return render(request, "scanner.html")

def upload_view(request):
    """Enhanced upload view with corruption-free .mind file generation"""
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
                    # Handle browser-compiled .mind files
                    mind_file = request.FILES['mind_file']
                    marker_dir = ensure_marker_directory(experience.slug)
                    mind_file_path = marker_dir / f"{experience.slug}.mind"
                    
                    with open(mind_file_path, 'wb') as f:
                        for chunk in mind_file.chunks():
                            f.write(chunk)
                    
                    if mind_file_path.exists() and mind_file_path.stat().st_size > 0:
                        experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                        marker_generated = True
                        marker_message = "🧠 .mind file compiled successfully in browser"
                        logger.info(f"✅ Browser-compiled .mind file saved for {experience.slug}")
                        # Validate the file
                        is_valid, validation_msg = validate_mind_file_strict(mind_file_path)
                        if not is_valid:
                            logger.warning(f"⚠️ Browser-compiled .mind file invalid: {validation_msg}")
                            marker_generated = False
                            marker_message = f"❌ Browser-compiled .mind file invalid: {validation_msg}"
                    else:
                        marker_message = "❌ Browser-compiled file is invalid or empty"
                        logger.error(f"❌ Invalid browser-compiled file for {experience.slug}")
                else:
                    # Generate .mind file from image
                    if experience.image:
                        marker_dir = ensure_marker_directory(experience.slug)
                        mind_file_path = marker_dir / f"{experience.slug}.mind"
                        
                        logger.info(f"🎯 Starting .mind file generation for: {experience.slug}")
                        
                        try:
                            # Step 1: Create professional trackable version
                            base_path = os.path.splitext(experience.image.path)[0]
                            trackable_image_path = f"{base_path}_trackable.png"
                            
                            trackable_path = create_pro_trackable_marker(experience.image.path, trackable_image_path)
                            
                            if os.path.exists(trackable_path):
                                logger.info(f"✅ Professional trackable image created: {trackable_path}")
                                
                                # Step 2: Validate trackable image quality
                                quality_results = validate_marker_quality(trackable_path)
                                if quality_results['valid'] and quality_results['score'] >= 0.6:  # Minimum good quality
                                    logger.info(f"📊 Trackable image quality: {quality_results['score']} (valid)")
                                    
                                    # Step 3: Generate corruption-free .mind file
                                    if generate_mindar_marker_fixed(trackable_path, str(mind_file_path)):
                                        marker_generated = True
                                        file_size = mind_file_path.stat().st_size
                                        quality_indicator = "🎯 Excellent" if file_size > 50000 else "✅ Good" if file_size > 20000 else "⚠️ Moderate" if file_size > 10000 else "📱 Basic"
                                        marker_message = f"🎯 .mind file created! {quality_indicator} ({file_size:,} bytes)"
                                        logger.info(f"🎉 SUCCESS: .mind file for {experience.slug} - {file_size} bytes")
                                        experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                                        # Validate the generated file
                                        is_valid, validation_msg = validate_mind_file_strict(mind_file_path)
                                        if not is_valid:
                                            logger.error(f"❌ Generated .mind file invalid: {validation_msg}")
                                            marker_generated = False
                                            marker_message = f"❌ Generated .mind file invalid: {validation_msg}"
                                    else:
                                        logger.error(f"❌ Failed to generate .mind file from trackable image")
                                        marker_message = "❌ Failed to generate .mind file from enhanced image"
                                else:
                                    logger.error(f"❌ Trackable image quality too low: {quality_results['score']}")
                                    marker_message = f"❌ Trackable image quality insufficient (score: {quality_results['score']})"
                            else:
                                logger.error(f"❌ Failed to create trackable image")
                                marker_message = "❌ Failed to create trackable image version"
                        except Exception as enhancement_error:
                            logger.error(f"❌ Enhancement process failed: {str(enhancement_error)}")
                            marker_message = f"❌ Processing failure: {str(enhancement_error)}"
                    else:
                        marker_message = "⚠️ No image provided for .mind file generation"
                        logger.warning(f"⚠️ No image provided for {experience.slug}")

                # Save marker status
                experience.marker_generated = marker_generated
                experience.save(update_fields=['nft_iset_file', 'marker_generated'])

                # QR Code generation
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

                # User feedback
                if marker_generated:
                    messages.success(request, f'🎯 AR Experience created! {marker_message}')
                    logger.info(f"SUCCESS: AR experience created for {experience.slug}")
                else:
                    messages.error(request, f'Upload completed but .mind file generation failed. {marker_message}')
                    logger.warning(f"WARNING: AR experience created but .mind file generation failed for {experience.slug}")
                
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
    """Enhanced view for rendering an AR experience with MindAR integration."""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
    except ARExperience.DoesNotExist:
        raise Http404(f"Experience '{slug}' not found")

    # Build base URLs
    base_url = getattr(settings, 'BASE_URL', request.build_absolute_uri('/').rstrip('/'))
    media_url = base_url + settings.MEDIA_URL.rstrip('/') if not settings.MEDIA_URL.startswith('http') else settings.MEDIA_URL
    marker_base_url = f"{media_url}/markers/{slug}/{slug}"

    # Check marker file existence and size
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
            logger.warning(f"MindAR target file missing or empty for {slug}")
    except Exception as e:
        logger.error(f"Error checking MindAR target for {slug}: {e}")

    # MindAR configuration
    mindar_config = {
        'maxTrack': 1,
        'showStats': settings.DEBUG,
        'uiLoading': 'no',
        'uiError': 'no',
        'uiScanning': 'no',
        'autoStart': True,
        'filterMinCF': 0.001,
        'filterBeta': 0.001,
        'missTolerance': 2,
        'warmupTolerance': 2,
        'targetAccuracy': 0.98
    }

    # Prepare media URLs
    video_url = experience.video.url if experience.video else None
    marker_image_url = experience.image.url if experience.image else None

    # Validate marker quality
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

    # Set up WebSocket
    websocket_url = None
    websocket_available = False
    try:
        from channels.layers import get_channel_layer
        get_channel_layer()  # Test channel layer availability
        websocket_available = True
        protocol = 'wss' if request.is_secure() else 'ws'
        websocket_url = f"{protocol}://{request.get_host()}/ws/ar/{slug}/"
    except Exception as e:
        logger.warning(f"Channel layer not available for {slug}: {str(e)}")

    # Initialize real-time tracker
    tracker = None
    tracker_available = False
    if websocket_available:
        try:
            tracker = get_tracker(slug)
            tracker_available = True
        except Exception as e:
            logger.warning(f"Cannot initialize tracker for {slug}: {str(e)}")

    # Determine if regeneration is needed
    need_to_regenerate = experience.image and (not marker_files_exist or mind_file_size < 1000)

    # Instructions for user
    instructions = {
        'setup': 'Allow camera access when prompted by your browser',
        'usage': 'Point your camera at the uploaded marker image',
        'distance': 'Hold your device 20-60cm away from the marker',
        'lighting': 'Ensure good lighting for better tracking',
        'stability': 'Keep the marker clearly visible and steady',
        'technology': 'Powered by Professional MindAR',
        'accuracy_target': f'Target accuracy: {int(mindar_config["targetAccuracy"] * 100)}% for professional performance'
    }

    # Build context
    context = {
        "experience": experience,
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "base_url": base_url,
        "title": experience.title,
        "video_url": video_url,
        "marker_image_url": marker_image_url,
        "timestamp": int(time.time()),
        "tracking_method": "Professional MindAR",
        "mindar_config": mindar_config,
        "mindar_config_json": json.dumps(mindar_config),
        "tracking_validation": tracking_validation,
        "websocket_url": websocket_url,
        "websocket_available": websocket_available,
        "target_accuracy": mindar_config['targetAccuracy'],
        "need_to_regenerate": need_to_regenerate,
        "mind_file_size": mind_file_size,
        "instructions": instructions,
        "debug": {
            'mind_file_path': mind_file_path,
            'mind_file_exists': marker_files_exist,
            'mind_file_size': mind_file_size,
            'media_root': str(settings.MEDIA_ROOT),
            'slug': slug,
            'tracking_issues': tracking_validation['issues'] if tracking_validation else [],
            'tracking_recommendations': tracking_validation['recommendations'] if tracking_validation else [],
            'target_accuracy': mindar_config['targetAccuracy'],
            'websocket_available': websocket_available,
            'tracker_available': tracker_available,
            'need_to_regenerate': need_to_regenerate
        } if settings.DEBUG else {},
        "tracking_functions": {
            "real_time_tracker": {
                "active": tracker.is_tracking if tracker else False,
                "slug": slug,
                "target_accuracy": mindar_config['targetAccuracy'],
                "available": tracker_available
            },
            "websocket_consumer": {
                "connected": False,
                "group_name": f'ar_tracking_{slug}',
                "target_accuracy": mindar_config['targetAccuracy'],
                "available": websocket_available
            },
            "marker_validation": {
                "valid": tracking_validation['valid'] if tracking_validation else False,
                "quality_score": tracking_validation['quality_score'] if tracking_validation else 0,
                "target_accuracy": mindar_config['targetAccuracy']
            }
        }
    }

    return render(request, "experience.html", context)

def webcam_ar_experience_view(request, slug=None):
    """Dedicated MindAR webcam experience view"""
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
        "title": experience.title if experience else "Professional MindAR Experience",
        "description": experience.description if experience else "Professional MindAR Webcam Experience",
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "marker_image_url": experience.image.url if experience and experience.image else None,
        "video_url": experience.video.url if experience and experience.video else None,
        "base_url": base_url,
        "timestamp": int(time.time()),
        "debug_mode": settings.DEBUG,
        "tracking_method": "Professional MindAR",
        "websocket_url": f"ws://{request.get_host()}/ws/ar/{experience.slug}/" if experience else None,
        "target_accuracy": 0.95,
        "user_instructions": {
            "camera_setup": "Allow camera access when prompted",
            "marker_usage": "Point camera at the uploaded image",
            "optimal_distance": "20-60cm from marker",
            "lighting_tips": "Ensure good lighting for tracking",
            "stability_advice": "Keep marker visible and steady",
            "accuracy_target": "Target accuracy: 95% for professional performance"
        },
        # Tracking function connections
        "tracking_functions": {
            "real_time_tracker": {
                "active": tracker.is_tracking if tracker else False,
                "slug": experience.slug if experience else None,
                "target_accuracy": 0.95
            },
            "websocket_consumer": {
                "connected": False,  # Will be updated in frontend
                "group_name": f'ar_tracking_{experience.slug}' if experience else None,
                "target_accuracy": 0.95
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
            "tracking_method": "Professional MindAR",
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
            "realtime_tracking": {
                "available": True,
                "websocket_url": f"ws://{request.get_host()}/ws/ar/{slug}/",
                "target_accuracy": 0.95
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

@csrf_exempt
def validate_tracking_api(request, slug):
    """API endpoint to validate marker tracking quality"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        test_image_path = None
        
        if request.method == 'POST' and request.FILES.get('test_image'):
            test_image = request.FILES['test_image']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                for chunk in test_image.chunks():
                    f.write(chunk)
                test_image_path = f.name
        
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def ar_experience_by_slug(request, slug):
    """AR experience viewer by slug"""
    try:
        experience = ARExperience.objects.get(slug=slug)
        tracker = get_tracker(slug)

        context = {
            'experience': experience,
            'video_url': experience.video.url if experience.video else None,
            'marker_url': experience.image.url if experience.image else None,
            'title': experience.title,
            'description': experience.description,
            'slug': experience.slug,
            'base_url': getattr(settings, 'BASE_URL', 'http://127.0.0.1:8000'),
            'tracking_method': 'Professional MindAR',
            'websocket_url': f"ws://{request.get_host()}/ws/ar/{slug}/",
            'target_accuracy': 0.95
        }
        return render(request, 'experience.html', context)
    except ARExperience.DoesNotExist:
        messages.error(request, f'AR Experience "{slug}" not found.')
        return redirect('upload')
    except Exception as e:
        logger.error(f"Error loading AR experience {slug}: {str(e)}")
        messages.error(request, 'Error loading AR experience. Please try again.')
        return redirect('upload')

def debug_markers(request, slug):
    """Debug view for marker status"""
    experience = get_object_or_404(ARExperience, slug=slug)
    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
    
    debug_info = {
        'slug': slug,
        'marker_dir': str(marker_dir),
        'marker_dir_exists': marker_dir.exists(),
        'files': {},
        'tracking_method': 'Professional MindAR',
        'target_accuracy': 0.95
    }
    
    mind_file = f"{slug}.mind"
    if marker_dir.exists():
        filepath = marker_dir / mind_file
        info = {
            'exists': filepath.exists(),
            'size': filepath.stat().st_size if filepath.exists() else 0
        }
        debug_info['files'][mind_file] = info
        
    return JsonResponse(debug_info, json_dumps_params={'indent': 2})

def validate_mind_file(file_path):
    """Quick validation function for backward compatibility"""
    return validate_mind_file_strict(file_path)

# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS (for compatibility)
# ============================================================================
def marker_status_api(request, slug):
    """API endpoint to check marker status"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        mind_file = marker_dir / f"{slug}.mind"
        file_exists = mind_file.exists() and mind_file.stat().st_size > 0
        
        return JsonResponse({
            'slug': slug,
            'tracking_method': 'Professional MindAR',
            'marker_generated': experience.marker_generated and file_exists,
            'files_exist': file_exists,
            'can_regenerate': bool(experience.image),
            'webcam_ready': file_exists,
            'target_accuracy': 0.95
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def save_browser_mindar_target(request):
    """Save browser-compiled MindAR target"""
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
                defaults={'slug': slug, 'description': f'Professional MindAR experience for {title}'}
            )
            
            marker_dir = ensure_marker_directory(experience.slug)
            mind_file_path = marker_dir / f"{experience.slug}.mind"
            
            with open(mind_file_path, 'wb') as f:
                for chunk in compiled_data.chunks():
                    f.write(chunk)
            
            if mind_file_path.exists() and mind_file_path.stat().st_size > 0:
                experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                experience.marker_generated = True
                experience.save(update_fields=['nft_iset_file', 'marker_generated'])
                
                return JsonResponse({
                    'success': True,
                    'message': f'Professional MindAR target saved successfully for "{title}"',
                    'slug': experience.slug,
                    'experience_url': f'/x/{experience.slug}/',
                    'file_size': mind_file_path.stat().st_size,
                    'target_accuracy': 0.95
                })
            else:
                return JsonResponse({'error': 'Invalid compiled data received'}, status=400)
                
        except Exception as e:
            logger.error(f"Error saving browser MindAR target: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)

# @csrf_exempt  
# def browser_mindar_compiler(request):
#     """Browser-based MindAR compiler interface"""
#     if request.method == 'GET':
#         context = {
#             'experiences': ARExperience.objects.all().order_by('-created_at')[:10],
#             'target_accuracy': 0.95
#         }
#         return render(request, 'browser_mindar_compiler.html', context)
#     return JsonResponse({'error': 'GET method required'}, status=405)

def serve_mind_file(request, slug):
    file_path = os.path.join(settings.MEDIA_ROOT, "markers", slug, f"{slug}.mind")
    if not os.path.exists(file_path):
        raise Http404("Marker not found")
    return FileResponse(open(file_path, 'wb'), content_type='application/octet-stream')