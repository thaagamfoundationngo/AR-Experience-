from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
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
import shutil
import subprocess
from PIL import Image, ImageEnhance
import time
import json

logger = logging.getLogger(__name__)

def optimize_image_for_markers(image_path: str, max_size: tuple = (512, 512), quality: int = 85) -> str:
    """Use original uploaded image as-is for NFT marker generation."""
    try:
        img_path = Path(image_path)
        
        if not img_path.exists():
            logger.error(f"Image file not found: {img_path}")
            return str(img_path)
        
        logger.info(f"Using original image for NFT marker generation: {img_path.name}")
        return str(img_path)
        
    except Exception as e:
        logger.error(f"Error in optimize_image_for_markers: {e}")
        return image_path

def check_node_environment() -> tuple[bool, str]:
    """Check if Node.js environment is properly set up."""
    try:
        node_cmd = "node.exe" if os.name == "nt" else "node"
        result = subprocess.run([node_cmd, "--version"], capture_output=True, text=True, timeout=None)
        if result.returncode != 0:
            return False, "Node.js not found or not working"
        
        node_version = result.stdout.strip()
        logger.info(f"Node.js version: {node_version}")
        
        pkg_root = Path(settings.BASE_DIR) / "node_modules" / "@webarkit" / "nft-marker-creator-app"
        script = pkg_root / "src" / "NFTMarkerCreator.js"
        
        if not script.exists():
            return False, f"NFTMarkerCreator.js not found at {script}"
        
        package_json = pkg_root / "package.json"
        if not package_json.exists():
            return False, "Package configuration not found"
        
        return True, f"Environment OK - Node.js {node_version}"
        
    except Exception as e:
        return False, f"Environment check failed: {e}"

def ensure_named(file_path: str, expected_name: str) -> str:
    """Ensures a file has the expected name. If not, renames it."""
    file_path = Path(file_path)
    expected_path = file_path.parent / expected_name
    
    if file_path.name != expected_name:
        try:
            shutil.move(str(file_path), str(expected_path))
            print(f"[arjs] Renamed {file_path.name} to {expected_name}")
            return str(expected_path)
        except Exception as e:
            print(f"[arjs] Failed to rename {file_path.name}: {e}")
            return str(file_path)
    
    return str(file_path)

def train_arjs_marker(image_path: str, out_dir: str, slug: str) -> tuple[bool, dict]:
    """
    Enhanced AR.js marker training WITHOUT timeout constraints.
    Returns (success, file_paths) where file_paths contains paths to generated files.
    """
    try:
        env_ok, env_msg = check_node_environment()
        if not env_ok:
            logger.error(f"Environment check failed: {env_msg}")
            return False, {}
        
        logger.info(f"Starting marker generation for {slug} WITHOUT timeout")
        
        optimized_image = optimize_image_for_markers(image_path)
        
        img = Path(optimized_image).resolve()
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        pkg_root = Path(settings.BASE_DIR) / "node_modules" / "@webarkit" / "nft-marker-creator-app"
        script = pkg_root / "src" / "NFTMarkerCreator.js"
        script_dir = script.parent

        timestamp = int(time.time())
        temp_img = script_dir / f"temp_{slug}_{timestamp}_{img.name}"
        
        generated_files = {}
        
        try:
            shutil.copy2(img, temp_img)
            logger.info(f"Copied image to: {temp_img}")
            
            if optimized_image != image_path:
                try:
                    os.remove(optimized_image)
                except:
                    pass
            
            node = "node.exe" if os.name == "nt" else "node"
            cmd = [node, str(script), "-i", temp_img.name]
            
            env = os.environ.copy()
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("NODE_OPTIONS", "--max-old-space-size=4096")
            
            logger.info(f"Running command: {' '.join(cmd)}")
            start_time = time.time()
            
            # Execute WITHOUT timeout
            proc = subprocess.run(
                cmd,
                cwd=str(script_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=None,  # NO TIMEOUT
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Command completed in {execution_time:.2f} seconds")
            
            if proc.stdout:
                logger.info(f"stdout: {proc.stdout[:500]}...")
            if proc.stderr:
                logger.warning(f"stderr: {proc.stderr[:500]}...")
            
            if proc.returncode != 0:
                logger.error(f"Node process failed with code {proc.returncode}")
                return False, {}
            
            gen_dir = script_dir / "output"
            if not gen_dir.exists():
                logger.error(f"Output directory not found: {gen_dir}")
                return False, {}
            
            produced = {".iset": None, ".fset": None, ".fset3": None}
            for p in gen_dir.glob("*"):
                if p.suffix in produced and produced[p.suffix] is None:
                    produced[p.suffix] = p
                    logger.info(f"Found generated file: {p}")
            
            copied_files = []
            for ext in [".iset", ".fset", ".fset3"]:
                src = produced.get(ext)
                if not src or not src.exists():
                    logger.error(f"Missing generated {ext} file")
                    continue
                    
                dest = out / f"{slug}{ext}"
                try:
                    shutil.copy2(src, dest)
                    copied_files.append(str(dest))
                    generated_files[ext] = str(dest)
                    logger.info(f"Copied {src} -> {dest} ({dest.stat().st_size} bytes)")
                except Exception as e:
                    logger.error(f"Copy error: {src} -> {dest}, {e}")
                    return False, {}
            
            try:
                for file in gen_dir.glob("*"):
                    file.unlink()
                logger.info("Cleaned up generated files")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
            
            success = len(copied_files) >= 3
            return success, generated_files
            
        finally:
            if temp_img.exists():
                try:
                    temp_img.unlink()
                    logger.info(f"Cleaned up temp image: {temp_img}")
                except Exception as e:
                    logger.warning(f"Temp cleanup warning: {e}")
    
    except Exception as e:
        logger.error(f"Exception in train_arjs_marker: {e}")
        return False, {}

def build_pattern_marker(image_path, slug, media_root):
    """
    Build pattern marker WITHOUT timeout constraints and return file paths.
    Returns tuple (success, file_paths) for database storage.
    """
    try:
        img = Path(image_path)
        out_dir = Path(media_root) / "markers" / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        if not img.exists():
            logger.error(f"[marker] Image file not found: {img}")
            return False, {}

        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True, timeout=None)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("[marker] Node.js is not installed or not in PATH")
            return False, {}

        logger.info(f"[marker] Starting NFT marker generation for slug: {slug} WITHOUT timeout")
        logger.info(f"[marker] Image: {img} ({img.stat().st_size} bytes)")
        logger.info(f"[marker] Output directory: {out_dir}")
        
        success, file_paths = train_arjs_marker(str(img), str(out_dir), slug)

        if success:
            logger.info(f"[marker] Successfully generated NFT markers for {slug}")
            return True, file_paths
        else:
            logger.warning(f"[marker] Failed to generate NFT markers for {slug}")
            return False, {}

    except Exception as e:
        logger.error(f"[marker] Error in build_pattern_marker: {str(e)}")
        return False, {}

def save_nft_files_to_database(experience: ARExperience, file_paths: dict):
    """Save generated NFT files to database fields (hidden from frontend)."""
    try:
        # Use the model's update_nft_files method instead of direct field assignment
        iset_path = file_paths.get('.iset')
        fset_path = file_paths.get('.fset')
        fset3_path = file_paths.get('.fset3')
        
        success = experience.update_nft_files(iset_path, fset_path, fset3_path)
        
        if success:
            logger.info(f"NFT files saved to database for {experience.slug}")
            return True
        else:
            logger.error(f"Failed to save NFT files to database for {experience.slug}")
            return False
        
    except Exception as e:
        logger.error(f"Error saving NFT files to database: {e}")
        return False    


def update_nft_files(self, iset_path=None, fset_path=None, fset3_path=None):
    """Update NFT file paths in database safely"""
    import logging
    logger = logging.getLogger(__name__)
    
    update_fields = []
    
    # Convert to string for comparison
    current_iset = str(self.nft_iset_file) if self.nft_iset_file else None
    current_fset = str(self.nft_fset_file) if self.nft_fset_file else None
    current_fset3 = str(self.nft_fset3_file) if self.nft_fset3_file else None
    
    # Update iset file
    if iset_path is not None and current_iset != iset_path:
        self.nft_iset_file = iset_path
        update_fields.append('nft_iset_file')
    
    # Update fset file
    if fset_path is not None and current_fset != fset_path:
        self.nft_fset_file = fset_path
        update_fields.append('nft_fset_file')
    
    # Update fset3 file
    if fset3_path is not None and current_fset3 != fset3_path:
        self.nft_fset3_file = fset3_path
        update_fields.append('nft_fset3_file')
    
    if update_fields:
        try:
            self.save(update_fields=update_fields)
            return True
        except Exception as e:
            logger.error(f"Error updating NFT files: {e}")
            return False
    
    return True  # No changes needed



   
def home(request):
    return render(request, "home.html")

def scanner(request):
    return render(request, "scanner.html")

def debug_markers(request, slug):
    experience = get_object_or_404(ARExperience, slug=slug)

    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
    debug_info = {
        'slug': slug,
        'marker_dir': str(marker_dir),
        'marker_dir_exists': marker_dir.exists(),
        'files': {},
        'nft_db_status': {
            'iset': bool(experience.nft_iset_file),
            'fset': bool(experience.nft_fset_file),
            'fset3': bool(experience.nft_fset3_file),
        }
    }

    required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]

    if marker_dir.exists():
        for filename in required_files:
            filepath = marker_dir / filename
            info = {
                'exists': filepath.exists(),
                'size': filepath.stat().st_size if filepath.exists() else 0
            }
            if filepath.exists():
                try:
                    with open(filepath, 'rb') as f:
                        info['content_preview'] = 'Binary OK'
                except:
                    info['content_preview'] = 'Binary file or read error'
            debug_info['files'][filename] = info

    return JsonResponse(debug_info, indent=2)

def upload_view(request):
    """Enhanced upload view with NFT file database storage and no timeouts"""
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

                marker_message = "No image provided"
                try:
                    if experience.image:
                        logger.info(f"Starting marker generation for {experience.slug} WITHOUT timeout")
                        
                        success, nft_file_paths = build_pattern_marker(
                            image_path=experience.image.path,
                            slug=experience.slug,
                            media_root=settings.MEDIA_ROOT
                        )
                        
                        if success and nft_file_paths:
                            logger.info(f"Marker generation successful for {experience.slug}")
                            
                            db_save_success = save_nft_files_to_database(experience, nft_file_paths)
                            
                            experience.marker_generated = True
                            marker_message = "Markers generated and saved successfully"
                            
                            if not db_save_success:
                                marker_message += " (Warning: Database save partially failed)"
                        else:
                            logger.warning(f"Marker generation failed for {experience.slug}")
                            experience.marker_generated = False
                            marker_message = "Marker generation failed"
                    else:
                        logger.info(f"No image provided for {experience.slug}")
                        experience.marker_generated = False
                        
                    experience.save(update_fields=["marker_generated"])

                except Exception as pattern_error:
                    logger.error(f"Pattern setup failed for {experience.slug}: {pattern_error}")
                    experience.marker_generated = False
                    experience.save(update_fields=["marker_generated"])
                    marker_message = f"Generation error: {pattern_error}"

                # QR Code Generation
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
                    logger.info(f"QR code saved at {qr_code_path}")
                    
                except Exception as qr_error:
                    logger.error(f"QR generation failed: {qr_error}")
                    messages.error(request, f"QR code generation failed: {qr_error}")

                if experience.marker_generated:
                    messages.success(request, f'AR Experience created successfully! {marker_message}')
                else:
                    messages.warning(request, f'AR Experience created, but marker generation failed: {marker_message}')
                    
                return redirect(f'/upload/?new={experience.slug}')

            except Exception as save_error:
                logger.error(f"Critical error in upload process: {save_error}")
                messages.error(request, f"Upload failed: {str(save_error)}")

        else:
            logger.error("Form validation failed.")
            for field, errors in form.errors.items():
                logger.error(f"  {field}: {errors}")
            messages.error(request, 'Please correct the errors below.')

    form = ARExperienceForm() if request.method == 'GET' else form
    
    context = {
        'form': form,
        'experiences': ARExperience.objects.all().order_by('-created_at'),
        'new_experience_slug': request.GET.get('new'),
        'marker_generated': True,
    }
    
    return render(request, 'upload.html', context)

def experience_view(request, slug):
    """Enhanced experience view with webcam activation capabilities"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    base_url = getattr(settings, 'BASE_URL', request.build_absolute_uri('/')[:-1])
    media_url = getattr(settings, 'MEDIA_URL', '/media/')
    if not media_url.startswith('http'):
        media_url = base_url + media_url
    
    marker_base_url = f"{media_url}markers/{slug}/{slug}"
    
    marker_files_exist = False
    try:
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]
        marker_files_exist = all(
            (marker_dir / file).exists() for file in required_files
        )
        
        if not marker_files_exist:
            logger.warning(f"⚠️ Some marker files missing for {slug}, regenerating...")
            success, nft_file_paths = build_pattern_marker(
                image_path=experience.image.path,
                slug=experience.slug,
                media_root=settings.MEDIA_ROOT
            )
            
            if success:
                save_nft_files_to_database(experience, nft_file_paths)
                logger.info(f"✅ Successfully regenerated marker files for {slug}")
                marker_files_exist = all(
                    (marker_dir / file).exists() for file in required_files
                )
            else:
                logger.error(f"❌ Failed to regenerate marker files for {slug}")
                marker_files_exist = False
            
    except Exception as e:
        logger.error(f"❌ Error checking marker files for {slug}: {e}")
        marker_files_exist = False
    
    webcam_config = {
        'enabled': True,
        'source_type': 'webcam',
        'device_id': None,
        'facing_mode': 'environment',
        'resolution': {
            'width': 640,
            'height': 480
        },
        'fps': 30,
        'auto_focus': True,
        'torch': False,
        'permissions_required': ['camera'],
        'https_required': True,
        'fallback_message': 'Camera access is required for AR experience'
    }
    
    feature_detection_config = {
        'tracking_method': 'best',
        'detection_mode': 'mono',
        'max_detection_rate': 60,
        'canvas_width': 640,
        'canvas_height': 480,
        'smoothing': {
            'enabled': True,
            'count': 10,
            'tolerance': 0.01,
            'threshold': 5
        }
    }
    
    ar_overlay_config = {
        'video_opacity': 0.95,
        'video_scale': {
            'width': 1.6,
            'height': 0.9
        },
        'position_offset': [0, 0, 0.01],
        'rotation_offset': [-90, 0, 0],
        'auto_play': True,
        'loop_video': True,
        'muted': True,
        'preload': 'metadata'
    }
    
    browser_requirements = {
        'webrtc_support': True,
        'webgl_support': True,
        'https_required': True,
        'modern_browser_required': True,
        'supported_browsers': ['Chrome', 'Firefox', 'Safari', 'Edge'],
        'minimum_versions': {
            'Chrome': '60+',
            'Firefox': '55+',
            'Safari': '11+',
            'Edge': '79+'
        }
    }
    
    context = {
        "experience": experience,
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "base_url": base_url,
        "title": experience.title,
        "video_url": experience.video.url if experience.video else None,
        "marker_image_url": experience.image.url if experience.image else None,
        "timestamp": int(time.time()),
        
        "webcam_enabled": True,
        "webcam_config": json.dumps(webcam_config),
        "feature_detection_config": json.dumps(feature_detection_config),
        "ar_overlay_config": json.dumps(ar_overlay_config),
        "browser_requirements": json.dumps(browser_requirements),
        
        "ui_config": json.dumps({
            'show_debug_info': settings.DEBUG,
            'show_fps_counter': True,
            'show_feature_count': True,
            'show_tracking_status': True,
            'enable_manual_controls': True
        }),
        
        "instructions": {
            'setup': 'Allow camera access when prompted by your browser',
            'usage': 'Point your camera at the uploaded marker image',
            'distance': 'Hold your device 20-60cm away from the marker',
            'lighting': 'Ensure good lighting for better tracking',
            'stability': 'Keep the marker clearly visible and steady'
        }
    }
    
    logger.info(f"🎥 Webcam AR experience loaded for {slug} (Markers: {'✅' if marker_files_exist else '❌'})")
    
    return render(request, "experience.html", context)

# Keep all other existing functions with their original names...

def webcam_ar_experience_view(request, slug=None):
    """Dedicated webcam AR experience view with full activation capabilities"""
    
    if slug:
        experience = get_object_or_404(ARExperience, slug=slug)
    else:
        try:
            experience = ARExperience.objects.filter(
                image__isnull=False,
                video__isnull=False
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
            required_files = [
                f"{experience.slug}.iset",
                f"{experience.slug}.fset",
                f"{experience.slug}.fset3"
            ]
            
            marker_files_exist = all(
                (marker_dir / file).exists() for file in required_files
            )
            
            if not marker_files_exist:
                logger.info(f"🔄 Regenerating marker files for {experience.slug}")
                try:
                    success, nft_file_paths = build_pattern_marker(
                        image_path=experience.image.path,
                        slug=experience.slug,
                        media_root=settings.MEDIA_ROOT
                    )
                    
                    if success:
                        save_nft_files_to_database(experience, nft_file_paths)
                        marker_files_exist = all(
                            (marker_dir / file).exists() for file in required_files
                        )
                        logger.info(f"✅ Marker files regenerated for {experience.slug}")
                    else:
                        logger.error(f"❌ Failed to regenerate marker files for {experience.slug}")
                        
                except Exception as e:
                    logger.error(f"❌ Error regenerating markers for {experience.slug}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error checking marker files for {experience.slug}: {e}")
    
    webcam_config = {
        'enabled': True,
        'auto_activate': True,
        'source_type': 'webcam',
        'device_constraints': {
            'video': {
                'width': {'ideal': 640, 'max': 1280},
                'height': {'ideal': 480, 'max': 720},
                'frameRate': {'ideal': 30, 'max': 60},
                'facingMode': 'environment'
            },
            'audio': False
        },
        'permissions': {
            'required': ['camera'],
            'optional': [],
            'fallback_message': 'Please allow camera access to use AR features'
        },
        'debug_mode': settings.DEBUG
    }
    
    ar_processing_config = {
        'tracking_method': 'best',
        'source_type': 'webcam',
        'debug_ui_enabled': settings.DEBUG,
        'detection_mode': 'mono',
        'matrix_code_type': '3x3',
        'max_detection_rate': 60,
        'canvas_width': 640,
        'canvas_height': 480,
        'label_color': 'white',
        'label_size': '0.5em'
    }
    
    feature_config = {
        'detection_threshold': 0.7,
        'tracking_stability': 5,
        'max_keypoints': 500,
        'smoothing_factor': 0.8,
        'loss_threshold': 3,
        'confidence_threshold': 0.6
    }
    
    video_config = {
        'opacity': 0.95,
        'scale': {'width': 1.6, 'height': 0.9},
        'position': [0, 0, 0.01],
        'rotation': [-90, 0, 0],
        'auto_play': True,
        'loop': True,
        'muted': True,
        'preload': 'metadata',
        'playsinline': True,
        'webkit_playsinline': True,
        'crossorigin': 'anonymous'
    }
    
    context = {
        "experience": experience,
        "title": experience.title if experience else "AR Experience",
        "description": experience.description if experience else "Webcam AR Experience",
        
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "marker_image_url": experience.image.url if experience and experience.image else None,
        "video_url": experience.video.url if experience and experience.video else None,
        
        "webcam_config": json.dumps(webcam_config),
        "ar_processing_config": json.dumps(ar_processing_config),
        "feature_config": json.dumps(feature_config),
        "video_config": json.dumps(video_config),
        
        "base_url": base_url,
        "timestamp": int(time.time()),
        "debug_mode": settings.DEBUG,
        
        "browser_requirements": json.dumps({
            "webrtc_support": True,
            "webgl_support": True,
            "https_required": True,
            "getUserMedia_required": True,
            "modern_browser_required": True
        }),
        
        "user_instructions": {
            "camera_setup": "Allow camera access when prompted",
            "marker_usage": "Point camera at the uploaded image",
            "optimal_distance": "20-60cm from marker",
            "lighting_tips": "Ensure good lighting for tracking",
            "stability_advice": "Keep marker visible and steady"
        }
    }
    
    if experience:
        logger.info(f"🎥📱 Webcam AR activated: {experience.title} (Markers: {'✅' if marker_files_exist else '❌'})")
    
    return render(request, "webcam_ar.html", context)

def ar_status_api(request, slug):
    """API endpoint to check AR experience status and webcam readiness"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
        required_files = [f"{experience.slug}.iset", f"{experience.slug}.fset", f"{experience.slug}.fset3"]
        marker_files_exist = all((marker_dir / file).exists() for file in required_files)
        
        status = {
            "success": True,
            "experience": {
                "title": experience.title,
                "slug": experience.slug,
                "created": experience.created_at.isoformat(),
            },
            "markers": {
                "exist": marker_files_exist,
                "files": required_files,
                "base_url": f"/media/markers/{experience.slug}/{experience.slug}"
            },
            "media": {
                "image_available": bool(experience.image),
                "video_available": bool(experience.video),
                "image_url": experience.image.url if experience.image else None,
                "video_url": experience.video.url if experience.video else None
            },
            "nft_database": {
                "iset_stored": bool(experience.nft_iset_file),
                "fset_stored": bool(experience.nft_fset_file),
                "fset3_stored": bool(experience.nft_fset3_file),
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

def ar_experience_by_slug(request, slug):
    """AR experience viewer accessible by slug"""
    try:
        experience = ARExperience.objects.get(slug=slug)

        context = {
            'experience': experience,
            'video_url': experience.video.url if experience.video else None,
            'marker_url': experience.image.url if experience.image else None,
            'title': experience.title,
            'description': experience.description,
            'slug': experience.slug,
            'base_url': settings.BASE_URL,
            'webcam_enabled': True,
        }

        return render(request, 'ar_viewer.html', context)

    except ARExperience.DoesNotExist:
        messages.error(request, f'AR Experience "{slug}" not found.')
        return redirect('upload')
    except Exception as e:
        logger.error(f"Error loading AR experience {slug}: {str(e)}")
        messages.error(request, 'Error loading AR experience. Please try again.')
        return redirect('upload')

def regenerate_markers(request, slug):
    """Enhanced marker regeneration WITHOUT timeout and with database storage"""
    if request.method == 'POST':
        experience = get_object_or_404(ARExperience, slug=slug)

        try:
            logger.info(f"Starting marker regeneration for {slug} WITHOUT timeout")
            logger.info(f"Processing image: {experience.image.path}")
            logger.info(f"Image size: {experience.image.size} bytes")
            
            success, nft_file_paths = build_pattern_marker(
                image_path=experience.image.path,
                slug=experience.slug,
                media_root=settings.MEDIA_ROOT
            )

            if success and nft_file_paths:
                db_save_success = save_nft_files_to_database(experience, nft_file_paths)
                
                experience.marker_generated = True
                experience.save(update_fields=['marker_generated'])
                
                logger.info(f"Marker regeneration completed for {slug}: Success")
                
                return JsonResponse({
                    'status': 'success', 
                    'message': f'Markers regenerated and saved successfully for {slug}',
                    'database_saved': db_save_success
                })
            else:
                experience.marker_generated = False
                experience.save(update_fields=['marker_generated'])
                
                return JsonResponse({
                    'status': 'error', 
                    'message': f'Failed to regenerate markers for {slug}'
                })

        except Exception as e:
            logger.error(f"Error regenerating markers for {slug}: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'POST method required'})

def qr_view(request, slug):
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
        'webcam_enabled': True,
    })

def marker_status_api(request, slug):
    try:
        experience = get_object_or_404(ARExperience, slug=slug)

        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]

        files_status, all_exist = {}, True
        for filename in required_files:
            fp = marker_dir / filename
            exists = fp.exists()
            files_status[filename] = {
                'exists': exists,
                'size': fp.stat().st_size if exists else 0,
                'path': str(fp)
            }
            all_exist &= exists

        return JsonResponse({
            'slug': slug,
            'marker_generated': experience.marker_generated and all_exist,
            'files_exist': all_exist,
            'files': files_status,
            'can_regenerate': bool(experience.image),
            'webcam_ready': all_exist,
            'nft_database_status': {
                'iset_stored': bool(experience.nft_iset_file),
                'fset_stored': bool(experience.nft_fset_file),
                'fset3_stored': bool(experience.nft_fset3_file),
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def webcam_ar_debug_view(request, slug):
    """Debug view for webcam AR experience development and troubleshooting"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
    marker_analysis = {}
    
    for ext in ['.iset', '.fset', '.fset3']:
        file_path = marker_dir / f"{experience.slug}{ext}"
        marker_analysis[ext] = {
            'exists': file_path.exists(),
            'size': file_path.stat().st_size if file_path.exists() else 0,
            'modified': file_path.stat().st_mtime if file_path.exists() else None
        }
    
    debug_context = {
        "experience": experience,
        "marker_analysis": marker_analysis,
        "media_root": str(settings.MEDIA_ROOT),
        "debug_mode": True,
        "webcam_debug": True,
        "ar_js_version": "3.4.5",
        "aframe_version": "1.5.0",
        "webcam_config": {
            'enabled': True,
            'debug_ui': True,
            'show_stats': True
        },
        "nft_database_info": {
            'iset_file': str(experience.nft_iset_file) if experience.nft_iset_file else None,
            'fset_file': str(experience.nft_fset_file) if experience.nft_fset_file else None,
            'fset3_file': str(experience.nft_fset3_file) if experience.nft_fset3_file else None,
        }
    }
    
    return render(request, "webcam_ar_debug.html", debug_context)

def ar_experience_view(request, experience_id: int):
    """Back-compat: resolve by ID then reuse the slug-based view with webcam."""
    exp = get_object_or_404(ARExperience, id=experience_id)
    return experience_view(request, exp.slug)

def realtime_experience_view(request, slug):
    """Real-time AR experience that doesn't require NFT marker generation"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    context = {
        "experience": experience,
        "title": experience.title,
        "video_url": experience.video.url if experience.video else None,
        "realtime_mode": True,
        "timestamp": int(time.time()),
    }
    
    logger.info(f"🎯 Real-time AR experience loaded: {experience.title}")
    return render(request, "realtime_experience.html", context)
