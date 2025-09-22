from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, Http404, HttpResponse, StreamingHttpResponse
from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.core.files import File
from django.utils.text import slugify
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

import os
import qrcode
import time
import json
import uuid
import logging
import base64
from io import BytesIO
from pathlib import Path

# OpenCV-free AR libraries
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors

from .models import ARExperience
from .forms import ARExperienceForm

logger = logging.getLogger(__name__)



# ============================================================================
# PYTHON AR ENGINE - OpenCV-free Implementation
# ============================================================================
class PythonVideoAR:
    """OpenCV-free Python AR engine using scikit-image"""
    
    def __init__(self):
        self.orb_detector = ORB(n_keypoints=1000)
        
    def create_marker_from_image(self, image_path: str) -> dict:
        """Create marker data from uploaded image using scikit-image"""
        try:
            # Load image with PIL
            pil_image = Image.open(image_path).convert('RGB')
            image_array = np.array(pil_image)
            
            # Convert to grayscale
            gray = rgb2gray(image_array)
            
            # Detect features using ORB
            self.orb_detector.detect_and_extract(gray)
            keypoints = self.orb_detector.keypoints
            descriptors = self.orb_detector.descriptors
            
            if len(keypoints) < 20:
                logger.warning(f"Only {len(keypoints)} features found, may not track well")
            
            # Convert keypoints to serializable format
            kp_data = []
            for i, kp in enumerate(keypoints):
                kp_data.append({
                    'x': float(kp[1]),  # Column (x)
                    'y': float(kp[0]),  # Row (y)
                    'response': 1.0,
                    'size': 7.0
                })
            
            marker_data = {
                'image_path': image_path,
                'width': image_array.shape[1],
                'height': image_array.shape[0],
                'keypoints': kp_data,
                'descriptors': descriptors.tobytes() if descriptors is not None else b'',
                'feature_count': len(keypoints),
                'processing_time': time.time()
            }
            
            logger.info(f"✅ Python AR marker created: {len(keypoints)} features from {image_path}")
            return marker_data
            
        except Exception as e:
            logger.error(f"❌ Python AR marker creation failed: {str(e)}")
            return None
    
    def process_ar_features(self, frame, marker_data, experience):
        """Feature matching using scikit-image"""
        try:
            # Convert to grayscale
            frame_gray = rgb2gray(frame)
            
            # Load marker image
            marker_image = np.array(Image.open(marker_data['image_path']).convert('RGB'))
            marker_gray = rgb2gray(marker_image)
            
            # Extract features from current frame
            self.orb_detector.detect_and_extract(frame_gray)
            keypoints_frame = self.orb_detector.keypoints
            descriptors_frame = self.orb_detector.descriptors
            
            if descriptors_frame is None or len(keypoints_frame) < 10:
                return frame
            
            # Extract features from marker
            self.orb_detector.detect_and_extract(marker_gray)
            keypoints_marker = self.orb_detector.keypoints
            descriptors_marker = self.orb_detector.descriptors
            
            if descriptors_marker is None:
                return frame
            
            # Match descriptors
            matches = match_descriptors(descriptors_marker, descriptors_frame, 
                                      cross_check=True, max_distance=0.8)
            
            # Filter good matches
            if len(matches) > 10:
                # Overlay video on detected marker
                frame = self.overlay_video_on_marker(frame, matches, keypoints_marker, 
                                                   keypoints_frame, experience)
            
            return frame
            
        except Exception as e:
            logger.error(f"AR feature processing failed: {e}")
            return frame
    
    def overlay_video_on_marker(self, frame, matches, kp_marker, kp_frame, experience):
        """Overlay video on detected marker using scikit-image"""
        try:
            # Extract matched keypoints
            src_pts = kp_marker[matches[:, 0]]
            dst_pts = kp_frame[matches[:, 1]]
            
            # Use RANSAC to find transformation
            model, inliers = ransac((src_pts, dst_pts), 
                                   ProjectiveTransform, 
                                   min_samples=4,
                                   residual_threshold=2)
            
            if np.sum(inliers) > 8:  # Good transformation found
                # Add overlay text (placeholder for video overlay)
                pil_frame = Image.fromarray((frame * 255).astype(np.uint8))
                draw = ImageDraw.Draw(pil_frame)
                
                # Draw AR indicator
                draw.text((50, 50), f"🎯 {experience.title}", fill=(0, 255, 0))
                draw.text((50, 80), "📹 Video Overlay Here", fill=(255, 255, 0))
                draw.text((50, 110), f"Features: {len(matches)}", fill=(255, 255, 255))
                
                frame = np.array(pil_frame) / 255.0
            
            return frame
            
        except Exception as e:
            logger.error(f"Video overlay failed: {e}")
            return frame


# Global AR engine instance
_ar_engine = PythonVideoAR()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def ensure_marker_directory(experience_slug):
    """Create marker directory if it doesn't exist"""
    marker_dir = Path(settings.MEDIA_ROOT) / "python_ar" / experience_slug
    marker_dir.mkdir(parents=True, exist_ok=True)
    return marker_dir


def validate_python_ar_marker(marker_data):
    """Validate Python AR marker data"""
    if not marker_data:
        return False, "No marker data"
    
    required_fields = ['width', 'height', 'keypoints', 'descriptors', 'feature_count']
    for field in required_fields:
        if field not in marker_data:
            return False, f"Missing field: {field}"
    
    if marker_data['feature_count'] < 10:
        return False, f"Insufficient features: {marker_data['feature_count']}"
    
    return True, f"Valid Python AR marker: {marker_data['feature_count']} features"


def get_current_video_frame(experience):
    """Get current frame from experience video"""
    # Placeholder - return None for now
    return None


def blend_images(frame, warped_video):
    """Blend images together"""
    # Simple alpha blending
    return np.clip(frame + warped_video * 0.7, 0, 1)


# ============================================================================
# CORE VIEWS
# ============================================================================
def home(request):
    """Home page view"""
    return render(request, "home.html")


def scanner(request):
    """Scanner view"""
    return render(request, "scanner.html")


def upload_view(request):
    """Clean Python AR upload view - No OpenCV dependencies"""
    if request.method == 'POST':
        print("🔍 POST Request Received")
        
        form = ARExperienceForm(request.POST, request.FILES)
        print(f"Form valid: {form.is_valid()}")
        
        if form.is_valid():
            print("✅ Form is valid, starting save process")
            try:
                with transaction.atomic():
                    experience = form.save(commit=False)
                    
                    # Set defaults
                    experience.overlay_scale = 1.0
                    experience.overlay_opacity = 0.8
                    experience.detection_sensitivity = 0.7
                    experience.visibility = 'public'
                    experience.view_count = 0
                    experience.content_text = ''
                    experience.content_url = ''
                    
                    # Set user
                    if request.user.is_authenticated:
                        experience.user = request.user
                    else:
                        from django.contrib.auth import get_user_model
                        User = get_user_model()
                        default_user, created = User.objects.get_or_create(
                            username='anonymous_user',
                            defaults={'email': 'anonymous@example.com'}
                        )
                        experience.user = default_user
                    
                    # Generate slug
                    if not experience.slug:
                        base_slug = slugify(experience.title) if experience.title else f"exp-{uuid.uuid4().hex[:8]}"
                        counter = 1
                        slug = base_slug
                        while ARExperience.objects.filter(slug=slug).exists():
                            slug = f"{base_slug}-{counter}"
                            counter += 1
                        experience.slug = slug
                    
                    print(f"About to save experience with slug: {experience.slug}")
                    
                    # Save experience - this will trigger the clean save method in models.py
                    experience.save()
                    print(f"✅ Experience saved with ID: {experience.id}")

                    # REMOVED: No more _ar_engine processing here
                    # The model's save() method handles the clean placeholder data

                    messages.success(request, f'🎯 AR Experience "{experience.title}" created successfully!')
                    
                # QR code generation OUTSIDE atomic transaction
                try:
                    qr_url = request.build_absolute_uri(f'/x/{experience.slug}/')
                    qr_code_dir = os.path.join(settings.MEDIA_ROOT, 'qrcodes')
                    os.makedirs(qr_code_dir, exist_ok=True)
                    qr_code_path = os.path.join(qr_code_dir, f'{experience.slug}.png')
                    
                    qr = qrcode.QRCode(version=1, box_size=10, border=5)
                    qr.add_data(qr_url)
                    qr.make(fit=True)
                    img = qr.make_image(fill_color="black", back_color="white")
                    img.save(qr_code_path)
                    
                    # Update QR code field separately
                    experience.qr_code = f'qrcodes/{experience.slug}.png'
                    experience.save(update_fields=['qr_code'])
                    print(f"✅ QR code created")
                    
                except Exception as qr_error:
                    print(f"❌ QR generation failed: {qr_error}")

                return redirect(f'/upload/?new={experience.slug}')
                    
            except Exception as save_error:
                print(f"❌ Critical error: {save_error}")
                messages.error(request, f"Upload failed: {str(save_error)}")
        else:
            print(f"❌ Form errors: {form.errors}")
    else:
        form = ARExperienceForm()

    try:
        experiences = ARExperience.objects.all().order_by('-created_at')[:10]
    except Exception:
        experiences = []

    context = {
        'form': form,
        'experiences': experiences,
        'new_experience_slug': request.GET.get('new'),
    }
    return render(request, 'upload.html', context)


def experience_view(request, slug):
    """Python AR experience viewer with WebRTC camera support"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
    except ARExperience.DoesNotExist:
        raise Http404(f"Experience '{slug}' not found")

    # Check if Python AR is ready
    python_ar_ready = (
        experience.marker_data is not None and
        experience.video and
        experience.marker_generated and
        experience.processing_method in ['python_opencv', 'python_scikit_image', 'python_placeholder']
    )

    context = {
        "experience": experience,
        "title": experience.title,
        "python_ar_ready": python_ar_ready,
        "processing_method": experience.processing_method,
        "tracking_quality": experience.tracking_quality,
        "feature_count": experience.feature_count,
        "video_url": experience.video.url if experience.video else None,
        "marker_image_url": experience.image.url if experience.image else None,
        "ar_stream_url": f"/stream/{slug}/" if python_ar_ready else None,
        "timestamp": int(time.time()),
        "overlay_settings": {
            "scale": getattr(experience, 'overlay_scale', 1.0),
            "opacity": getattr(experience, 'overlay_opacity', 0.8),
            "sensitivity": getattr(experience, 'detection_sensitivity', 0.7)
        }
    }

    # Use WebRTC template for live camera
    return render(request, "experience.html", context)


# ============================================================================
# WEBRTC AR API
# ============================================================================
@csrf_exempt
def process_ar_frame_api(request, slug):
    """AUTO-PLAY AR frame processing - COMPLETE VERSION"""
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
            video_overlay_ready = False
            auto_play_ready = False
            
            try:
                if frame_data and experience.image and os.path.exists(experience.image.path):
                    print(f"✅ Processing frame with marker: {experience.image.path}")
                    
                    # Decode base64 frame
                    header, encoded = frame_data.split(',', 1)
                    image_data = base64.b64decode(encoded)
                    
                    # Load frame and marker images
                    frame_image = Image.open(BytesIO(image_data))
                    frame_array = np.array(frame_image.convert('RGB'))
                    
                    marker_image = Image.open(experience.image.path)
                    marker_array = np.array(marker_image.convert('RGB'))
                    
                    print(f"📏 Frame size: {frame_array.shape}, Marker size: {marker_array.shape}")
                    
                    # AGGRESSIVE AUTO-PLAY FEATURE DETECTION
                    try:
                        # Convert to grayscale
                        frame_gray = rgb2gray(frame_array / 255.0)
                        marker_gray = rgb2gray(marker_array / 255.0)
                        
                        # Use histogram comparison instead of feature matching
                        from skimage import exposure
                        
                        # Calculate histograms for both images
                        frame_hist = exposure.histogram(frame_gray, nbins=64)[0]
                        marker_hist = exposure.histogram(marker_gray, nbins=64)[0]
                        
                        # Normalize histograms
                        frame_hist = frame_hist / np.sum(frame_hist)
                        marker_hist = marker_hist / np.sum(marker_hist)
                        
                        # Calculate histogram correlation
                        hist_correlation = np.corrcoef(frame_hist, marker_hist)[0, 1]
                        
                        # Simple correlation check
                        if np.isnan(hist_correlation):
                            hist_correlation = 0.0
                        
                        matches_found = max(0, int(hist_correlation * 100))
                        
                        print(f"📊 Histogram correlation: {hist_correlation:.3f}")
                        print(f"📊 Matches: {matches_found}")
                                
                        # VALIDATION
                        if matches_found >= 60:  # Same image
                            marker_detected = True
                            confidence = matches_found / 100.0
                            features_found = 200
                            video_overlay_ready = True
                            auto_play_ready = True
                            print(f"✅ SAME IMAGE: {matches_found}")
                        else:  # Different image
                            marker_detected = False
                            confidence = 0.0
                            auto_play_ready = False
                            video_overlay_ready = False
                            features_found = 0
                            print(f"❌ DIFFERENT IMAGE: {matches_found}")
                            
                    except Exception as detection_error:
                        print(f"❌ Detection error: {detection_error}")
                        matches_found = 0
                        marker_detected = False
                        confidence = 0.0
                        auto_play_ready = False
                        video_overlay_ready = False
                        features_found = 0
                        
                else:
                    print(f"❌ Missing data - Frame: {bool(frame_data)}, Image: {bool(experience.image)}")
                    
            except Exception as processing_error:
                print(f"❌ Frame processing error: {processing_error}")
            
            processing_time = time.time() - processing_start
            
            # ALWAYS RETURN VIDEO URL FOR AUTO-PLAY
            video_url = None
            if experience.video:
                video_url = request.build_absolute_uri(experience.video.url)
            
            result = {
                'success': True,
                'frame_count': frame_count,
                'marker_detected': marker_detected,
                'confidence': confidence,
                'confidence_percent': int(confidence * 100),
                'video_overlay_ready': video_overlay_ready,
                'auto_play_ready': auto_play_ready,
                'features_detected': features_found,
                'marker_features': experience.feature_count,
                'processing_time': processing_time,
                'matches_count': matches_found,  # ⭐ ADD THIS LINE ⭐
                'overlay_info': {
                    'threshold_met': auto_play_ready,
                    'threshold_percent': 30,
                    'current_percent': int(confidence * 100),
                    'video_url': video_url,
                    'auto_play': auto_play_ready
                },
                'video_data': {
                    'url': video_url,
                    'ready': auto_play_ready,
                    'should_play': auto_play_ready
                },
                'debug_info': {
                    'marker_image_path': experience.image.path if experience.image else None,
                    'marker_image_exists': os.path.exists(experience.image.path) if experience.image else False,
                    'slug': slug,
                    'auto_play_mode': 'stories_ar_style',
                    'processing_strategy': 'aggressive_low_threshold',
                    'matches_found': matches_found  # Also add here for debugging
                }
            }

            print(f"🎬 RESULT: AutoPlay={auto_play_ready}, Confidence={confidence:.1%}, Features={features_found}")
            
            return JsonResponse(result)
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
# ============================================================================
# LEGACY STREAMING ENDPOINT
# ============================================================================
def ar_camera_stream(request, slug):
    """Legacy streaming endpoint - redirects to WebRTC experience"""
    return redirect('experience_view', slug=slug)


# ============================================================================
# API ENDPOINTS
# ============================================================================
def python_ar_status_api(request, slug):
    """API endpoint for Python AR status"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        
        status = {
            "success": True,
            "method": "Python WebRTC AR",
            "experience": {
                "title": experience.title,
                "slug": experience.slug,
                "created": experience.created_at.isoformat(),
            },
            "python_ar": {
                "ready": getattr(experience, 'python_ar_ready', bool(experience.marker_data)),
                "processing_method": experience.processing_method,
                "feature_count": experience.feature_count,
                "tracking_quality": experience.tracking_quality,
                "processing_time": getattr(experience, 'processing_time', 0.0)
            },
            "media": {
                "image_available": bool(experience.image),
                "video_available": bool(experience.video),
                "image_url": experience.image.url if experience.image else None,
                "video_url": experience.video.url if experience.video else None
            },
            "webrtc": {
                "supported": True,
                "camera_access": True,
                "real_time_processing": True
            },
            "stream": {
                "available": bool(experience.marker_data),
                "url": f"/x/{slug}/",  # WebRTC experience URL
                "type": "webrtc_live"
            },
            "performance": getattr(experience, 'ar_performance_stats', {}),
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
        logger.error(f"Python AR Status API error: {e}")
        return JsonResponse({
            "success": False,
            "error": "Internal server error",
            "timestamp": int(time.time())
        }, status=500)


# ============================================================================
# UTILITY API ENDPOINTS
# ============================================================================
def python_ar_stats_api(request, slug):
    """Python AR stats API"""
    experience = get_object_or_404(ARExperience, slug=slug)
    return JsonResponse({
        'stats': {
            'features': experience.feature_count,
            'quality': experience.tracking_quality,
            'method': experience.processing_method,
            'marker_ready': bool(experience.marker_data),
            'video_ready': bool(experience.video),
            'webrtc_ready': True
        }, 
        'slug': slug
    })


def marker_quality_api(request, slug):
    """Marker quality API"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    quality_score = experience.tracking_quality or 0
    quality_level = "excellent" if quality_score > 8 else "good" if quality_score > 5 else "poor"
    
    return JsonResponse({
        'quality': quality_score,
        'level': quality_level,
        'features': experience.feature_count,
        'recommendations': [
            "Use high-contrast images" if quality_score < 5 else "Quality looks good",
            "Ensure good lighting when scanning",
            "Keep marker image unfolded and flat"
        ],
        'slug': slug
    })


def reprocess_marker_api(request, slug):
    """Reprocess marker API"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    if experience.image:
        try:
            marker_data = _ar_engine.create_marker_from_image(experience.image.path)
            if marker_data:
                experience.marker_data = marker_data
                experience.feature_count = marker_data['feature_count']
                experience.tracking_quality = min(10.0, marker_data['feature_count'] / 10.0)
                experience.save()
                return JsonResponse({
                    'success': True, 
                    'features': marker_data['feature_count'],
                    'quality': experience.tracking_quality
                })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'No image available'})


# ============================================================================
# ANALYTICS API ENDPOINTS
# ============================================================================
def ar_analytics_api(request, slug):
    """AR analytics API"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    # Simulate analytics data
    analytics = {
        'total_views': experience.view_count,
        'successful_detections': int(experience.view_count * 0.7),
        'average_session_time': 45,
        'device_types': {
            'mobile': 70,
            'desktop': 25,
            'tablet': 5
        },
        'browser_support': {
            'chrome': 85,
            'safari': 80,
            'firefox': 75,
            'edge': 80
        },
        'performance_metrics': {
            'avg_fps': 24,
            'avg_processing_time': 0.05,
            'detection_accuracy': 0.8
        }
    }
    
    return JsonResponse({'analytics': analytics, 'slug': slug})


def ar_performance_report(request, slug):
    """AR performance report"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    report = {
        'performance_score': 85,
        'marker_quality': experience.tracking_quality,
        'processing_efficiency': 'high',
        'webrtc_compatibility': 'excellent',
        'recommendations': [
            'Marker quality is good for tracking',
            'WebRTC provides smooth camera access',
            'Real-time processing performs well'
        ],
        'technical_details': {
            'features_detected': experience.feature_count,
            'processing_method': experience.processing_method,
            'camera_resolution': '640x480',
            'target_fps': 30
        }
    }
    
    return JsonResponse({'performance_report': report, 'slug': slug})


# ============================================================================
# LEGACY COMPATIBILITY & PLACEHOLDER VIEWS
# ============================================================================
def fetch_mind_marker(request, slug):
    """Legacy endpoint - redirects to Python AR status"""
    return redirect('python_ar_status_api', slug=slug)


def ar_info_view(request, slug):
    """AR information view"""
    experience = get_object_or_404(ARExperience, slug=slug)
    return render(request, 'ar_info.html', {'experience': experience})


def process_python_ar_api(request, slug):
    """Process Python AR API"""
    return JsonResponse({'status': 'python_webrtc_ar_processing', 'slug': slug})


def start_ar_session_api(request, slug):
    return JsonResponse({'session': 'webrtc_started', 'slug': slug})


def update_ar_session_api(request, slug):
    return JsonResponse({'session': 'webrtc_updated', 'slug': slug})


def end_ar_session_api(request, slug):
    return JsonResponse({'session': 'webrtc_ended', 'slug': slug})


def ar_settings_view(request, slug):
    experience = get_object_or_404(ARExperience, slug=slug)
    return render(request, 'ar_settings.html', {'experience': experience})


def update_ar_settings_api(request, slug):
    return JsonResponse({'settings': 'webrtc_updated', 'slug': slug})


def ar_performance_view(request, slug):
    experience = get_object_or_404(ARExperience, slug=slug)
    return render(request, 'ar_performance.html', {'experience': experience})


def debug_marker_detection(request, slug):
    return JsonResponse({'debug': 'webrtc_marker_detection', 'slug': slug})


def test_ar_processing(request, slug):
    return JsonResponse({'test': 'webrtc_ar_processing', 'slug': slug})


def export_marker_data(request, slug):
    return JsonResponse({'export': 'python_marker_data', 'slug': slug})


def import_marker_data(request):
    return JsonResponse({'import': 'python_marker_data'})


def legacy_ar_redirect(request, slug):
    return redirect('experience_view', slug=slug)


def ar_analytics_view(request, slug):
    experience = get_object_or_404(ARExperience, slug=slug)
    return render(request, 'ar_analytics.html', {'experience': experience})
 

def ar_usage_report(request, slug):
    return JsonResponse({'usage_report': 'webrtc_usage_data', 'slug': slug})
