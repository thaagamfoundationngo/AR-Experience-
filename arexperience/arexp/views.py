from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, Http404
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

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def generate_mindar_marker(image_path, output_path):
    """
    Generate MindAR marker using Python-based implementation
    Returns True if successful, False otherwise
    """
    try:
        logger.info(f"Starting MindAR marker generation for: {image_path}")
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints using ORB (similar to what MindAR uses)
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if len(keypoints) < 50:
            logger.error(f"Not enough keypoints detected in image: {len(keypoints)}")
            return False
        
        # Create a simple MindAR-like data structure
        mindar_data = {
            "imageWidth": img.shape[1],
            "imageHeight": img.shape[0],
            "maxTrack": 1,
            "filterMinCF": 0.001,
            "filterBeta": 0.001,
            "missTolerance": 5,
            "warmupTolerance": 5,
            "targets": [{
                "name": os.path.splitext(os.path.basename(image_path))[0],
                "keypoints": [],
                "descriptors": []
            }]
        }
        
        # Add keypoints and descriptors to the data structure
        for kp, desc in zip(keypoints, descriptors):
            mindar_data["targets"][0]["keypoints"].append({
                "x": float(kp.pt[0]),
                "y": float(kp.pt[1]),
                "size": float(kp.size),
                "angle": float(kp.angle),
                "response": float(kp.response),
                "octave": int(kp.octave)
            })
            
            # Convert descriptor to list
            mindar_data["targets"][0]["descriptors"].append(desc.tolist())
        
        # Convert to binary format similar to MindAR
        json_str = json.dumps(mindar_data)
        
        # Create a simple binary header
        header = struct.pack('4sI', b'MIND', len(json_str))
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(json_str.encode('utf-8'))
        
        # Verify file was created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"MindAR marker generated successfully: {output_path}")
            return True
        else:
            logger.error(f"Failed to create MindAR marker file: {output_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error generating MindAR marker: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def ensure_marker_directory(experience_slug):
    """Create marker directory if it doesn't exist"""
    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience_slug
    marker_dir.mkdir(parents=True, exist_ok=True)
    return marker_dir

# ============================================================================
# CORE AR EXPERIENCE FUNCTIONS
# ============================================================================
def home(request):
    return render(request, "home.html")

def scanner(request):
    return render(request, "scanner.html")

def upload_view(request):
    """Enhanced upload view with server-side MindAR marker generation"""
    if request.method == 'POST':
        form = ARExperienceForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                with transaction.atomic():
                    experience = form.save(commit=False)
                    
                    # Generate unique slug
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
                
                # Handle marker generation
                marker_generated = False
                marker_message = ""
                
                # Priority 1: Browser-compiled MindAR data
                if request.POST.get('use_browser_compilation') and request.FILES.get('mind_file'):
                    mind_file = request.FILES['mind_file']
                    marker_dir = ensure_marker_directory(experience.slug)
                    mind_file_path = marker_dir / f"{experience.slug}.mind"
                    
                    # Save browser-compiled file
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
                
                # Priority 2: Server-side generation if image exists
                elif experience.image:
                    marker_dir = ensure_marker_directory(experience.slug)
                    mind_file_path = marker_dir / f"{experience.slug}.mind"
                    
                    if generate_mindar_marker(experience.image.path, str(mind_file_path)):
                        experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                        marker_generated = True
                        marker_message = "🧠 MindAR target generated successfully on server"
                        logger.info(f"Server-generated marker: {mind_file_path}")
                    else:
                        marker_message = "❌ Failed to generate MindAR target on server"
                        logger.error(f"Failed to generate marker for {experience.slug}")
                
                else:
                    marker_message = "⚠️ No image provided for marker generation"
                
                # Update experience with marker status
                experience.marker_generated = marker_generated
                experience.save(update_fields=['nft_iset_file', 'marker_generated'])
                
                # Generate QR Code
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
                
                # Set appropriate success message
                if marker_generated:
                    messages.success(request, f'🎯 AR Experience created! {marker_message}')
                else:
                    messages.warning(request, f'AR Experience created. {marker_message}')
                    
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
    """Enhanced experience view with MindAR capabilities"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
    except ARExperience.DoesNotExist:
        raise Http404(f"Experience '{slug}' not found")
    
    # Build base URL properly
    base_url = getattr(settings, 'BASE_URL', request.build_absolute_uri('/').rstrip('/'))
    
    # Build media URL
    media_url = settings.MEDIA_URL
    if not media_url.startswith('http'):
        media_url = base_url + media_url.rstrip('/')
    
    # Marker paths
    marker_base_url = f"{media_url}/markers/{slug}/{slug}"
    
    # Check for MindAR target file
    marker_files_exist = False
    mind_file_path = None
    
    try:
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        mind_file = marker_dir / f"{slug}.mind"
        mind_file_path = str(mind_file)
        
        if mind_file.exists() and mind_file.stat().st_size > 0:
            marker_files_exist = True
            logger.info(f"MindAR target file found for {slug}")
        else:
            logger.warning(f"MindAR target file missing for {slug}")
            
    except Exception as e:
        logger.error(f"Error checking MindAR target for {slug}: {e}")
    
    # MindAR configuration
    mindar_config = {
        'maxTrack': 1,
        'showStats': settings.DEBUG,
        'uiLoading': 'no',
        'uiError': 'no', 
        'uiScanning': 'no',
        'autoStart': False,
        'filterMinCF': 0.0001,
        'filterBeta': 0.001,
        'missTolerance': 5,
        'warmupTolerance': 2
    }
    
    # Build URLs
    video_url = experience.video.url if experience.video else None
    marker_image_url = experience.image.url if experience.image else None
    
    context = {
        "experience": experience,
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "base_url": base_url,
        "title": experience.title,
        "video_url": video_url,
        "marker_image_url": marker_image_url,
        "timestamp": int(time.time()),
        "tracking_method": "MindAR",
        "mindar_config": mindar_config,
        "mindar_config_json": json.dumps(mindar_config),
        "instructions": {
            'setup': 'Allow camera access when prompted by your browser',
            'usage': 'Point your camera at the uploaded marker image',
            'distance': 'Hold your device 20-60cm away from the marker',
            'lighting': 'Ensure good lighting for better tracking',
            'stability': 'Keep the marker clearly visible and steady',
            'technology': 'Powered by MindAR for superior tracking'
        },
        "debug": {
            'mind_file_path': mind_file_path,
            'media_root': str(settings.MEDIA_ROOT),
            'slug': slug,
        } if settings.DEBUG else {}
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
    
    context = {
        "experience": experience,
        "title": experience.title if experience else "MindAR Experience",
        "description": experience.description if experience else "MindAR Webcam Experience",
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "marker_image_url": experience.image.url if experience and experience.image else None,
        "video_url": experience.video.url if experience and experience.video else None,
        "base_url": base_url,
        "timestamp": int(time.time()),
        "debug_mode": settings.DEBUG,
        "tracking_method": "MindAR",
        "user_instructions": {
            "camera_setup": "Allow camera access when prompted",
            "marker_usage": "Point camera at the uploaded image",
            "optimal_distance": "20-60cm from marker",
            "lighting_tips": "Ensure good lighting for tracking",
            "stability_advice": "Keep marker visible and steady"
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
            "tracking_method": "MindAR",
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
            'tracking_method': 'MindAR',
            'marker_generated': experience.marker_generated and file_exists,
            'files_exist': file_exists,
            'files': file_status,
            'can_regenerate': bool(experience.image),
            'webcam_ready': file_exists,
            'mindar_database_status': {
                'target_stored': bool(experience.nft_iset_file),
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

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
            
            # Generate slug from title
            slug = slugify(title) or f"exp-{uuid.uuid4().hex[:8]}"
            counter = 1
            original_slug = slug
            while ARExperience.objects.filter(slug=slug).exists():
                slug = f"{original_slug}-{counter}"
                counter += 1
            
            # Create or get the AR experience
            experience, created = ARExperience.objects.get_or_create(
                title=title,
                defaults={'slug': slug, 'description': f'MindAR experience for {title}'}
            )
            
            if not created:
                experience.slug = slug
                experience.save()
            
            # Save the compiled .mind file
            marker_dir = ensure_marker_directory(experience.slug)
            mind_file_path = marker_dir / f"{experience.slug}.mind"
            
            # Write compiled data to file
            with open(mind_file_path, 'wb') as f:
                for chunk in compiled_data.chunks():
                    f.write(chunk)
            
            # Verify file was saved correctly
            valid_file = mind_file_path.exists() and mind_file_path.stat().st_size > 0
            
            if valid_file:
                # Update the AR experience with file path
                experience.nft_iset_file = str(mind_file_path.relative_to(Path(settings.MEDIA_ROOT)))
                experience.marker_generated = True
                experience.save(update_fields=['nft_iset_file', 'marker_generated'])
                
                logger.info(f"Browser-compiled MindAR target saved for '{title}' (slug: {experience.slug})")
                
                return JsonResponse({
                    'success': True, 
                    'message': f'MindAR target saved successfully for "{title}"',
                    'slug': experience.slug,
                    'experience_url': f'/x/{experience.slug}/',
                    'file_size': mind_file_path.stat().st_size
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
            'experiences': ARExperience.objects.all().order_by('-created_at')[:10]
        }
        return render(request, 'browser_mindar_compiler.html', context)
    
    return JsonResponse({'error': 'GET method required'}, status=405)

# ============================================================================
# UTILITY AND COMPATIBILITY FUNCTIONS
# ============================================================================
def ar_experience_by_slug(request, slug):
    """MindAR experience viewer accessible by slug"""
    try:
        experience = ARExperience.objects.get(slug=slug)
        context = {
            'experience': experience,
            'video_url': experience.video.url if experience.video else None,
            'marker_url': experience.image.url if experience.image else None,
            'title': experience.title,
            'description': experience.description,
            'slug': experience.slug,
            'base_url': getattr(settings, 'BASE_URL', 'http://127.0.0.1:8000'),
            'tracking_method': 'MindAR',
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
    
    return render(request, 'experience.html', {
        'experience': experience,
        'qr_data': qr_data,
        'experience_url': experience_url,
        'tracking_method': 'MindAR',
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
        'tracking_method': 'MindAR',
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
    
    return JsonResponse(debug_info, indent=2)