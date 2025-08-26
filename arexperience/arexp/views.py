from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.contrib.staticfiles import finders
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

logger = logging.getLogger(__name__)

def optimize_image_for_markers(image_path: str, max_size: tuple = (512, 512), quality: int = 85) -> str:
    """
    Optimize image for better marker generation performance.
    Returns path to optimized image.
    """
    try:
        img_path = Path(image_path)
        optimized_path = img_path.parent / f"optimized_{img_path.name}"
        
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Resize if too large
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image from original to {img.size}")
            
            # Enhance contrast for better marker detection
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Save optimized image
            img.save(optimized_path, 'JPEG', quality=quality, optimize=True)
            logger.info(f"Optimized image saved: {optimized_path}")
            
        return str(optimized_path)
        
    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        return image_path  # Return original if optimization fails

def check_node_environment() -> tuple[bool, str]:
    """
    Check if Node.js environment is properly set up.
    Returns (success, message)
    """
    try:
        # Check Node.js
        node_cmd = "node.exe" if os.name == "nt" else "node"
        result = subprocess.run([node_cmd, "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "Node.js not found or not working"
        
        node_version = result.stdout.strip()
        logger.info(f"Node.js version: {node_version}")
        
        # Check if NFT marker creator is installed
        pkg_root = Path(settings.BASE_DIR) / "node_modules" / "@webarkit" / "nft-marker-creator-app"
        script = pkg_root / "src" / "NFTMarkerCreator.js"
        
        if not script.exists():
            return False, f"NFTMarkerCreator.js not found at {script}"
        
        # Check package.json
        package_json = pkg_root / "package.json"
        if not package_json.exists():
            return False, "Package configuration not found"
        
        return True, f"Environment OK - Node.js {node_version}"
        
    except subprocess.TimeoutExpired:
        return False, "Node.js check timed out"
    except Exception as e:
        return False, f"Environment check failed: {e}"

def ensure_named(file_path: str, expected_name: str) -> str:
    """
    Ensures a file has the expected name. If not, renames it.
    Returns the path to the correctly named file.
    """
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

def train_arjs_marker(image_path: str, out_dir: str, slug: str, timeout: int = 120) -> bool:
    """
    Enhanced AR.js marker training with better error handling and logging.
    """
    try:
        # Environment check
        env_ok, env_msg = check_node_environment()
        if not env_ok:
            logger.error(f"Environment check failed: {env_msg}")
            return False
        
        logger.info(f"Starting marker generation for {slug} with timeout {timeout}s")
        
        # Optimize image first
        optimized_image = optimize_image_for_markers(image_path)
        
        img = Path(optimized_image).resolve()
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        # Package locations
        pkg_root = Path(settings.BASE_DIR) / "node_modules" / "@webarkit" / "nft-marker-creator-app"
        script = pkg_root / "src" / "NFTMarkerCreator.js"
        script_dir = script.parent

        # Create unique temporary image name to avoid conflicts
        timestamp = int(time.time())
        temp_img = script_dir / f"temp_{slug}_{timestamp}_{img.name}"
        
        try:
            # Copy image to script directory
            shutil.copy2(img, temp_img)
            logger.info(f"Copied image to: {temp_img}")
            
            # Clean up optimized image if it's different from original
            if optimized_image != image_path:
                try:
                    os.remove(optimized_image)
                except:
                    pass
            
            # Prepare command
            node = "node.exe" if os.name == "nt" else "node"
            cmd = [node, str(script), "-i", temp_img.name]
            
            # Environment setup
            env = os.environ.copy()
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("NODE_OPTIONS", "--max-old-space-size=2048")  # Increase memory
            
            logger.info(f"Running command: {' '.join(cmd)}")
            start_time = time.time()
            
            # Execute with extended timeout
            proc = subprocess.run(
                cmd,
                cwd=str(script_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=timeout,
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Command completed in {execution_time:.2f} seconds")
            
            # Log outputs (truncated for readability)
            if proc.stdout:
                logger.info(f"stdout: {proc.stdout[:500]}...")
            if proc.stderr:
                logger.warning(f"stderr: {proc.stderr[:500]}...")
            
            if proc.returncode != 0:
                logger.error(f"Node process failed with code {proc.returncode}")
                return False
            
            # Check for output files
            gen_dir = script_dir / "output"
            if not gen_dir.exists():
                logger.error(f"Output directory not found: {gen_dir}")
                return False
            
            # Find generated files
            produced = {".iset": None, ".fset": None, ".fset3": None}
            for p in gen_dir.glob("*"):
                if p.suffix in produced and produced[p.suffix] is None:
                    produced[p.suffix] = p
                    logger.info(f"Found generated file: {p}")
            
            # Copy files to destination
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
                    logger.info(f"Copied {src} -> {dest} ({dest.stat().st_size} bytes)")
                except Exception as e:
                    logger.error(f"Copy error: {src} -> {dest}, {e}")
                    return False
            
            # Clean up generated files
            try:
                for file in gen_dir.glob("*"):
                    file.unlink()
                logger.info("Cleaned up generated files")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
            
            success = len(copied_files) >= 3  # Need all three files
            return success
            
        finally:
            # Always clean up temporary image
            if temp_img.exists():
                try:
                    temp_img.unlink()
                    logger.info(f"Cleaned up temp image: {temp_img}")
                except Exception as e:
                    logger.warning(f"Temp cleanup warning: {e}")
    
    except subprocess.TimeoutExpired:
        logger.error(f"Process timed out after {timeout} seconds")
        # Try once more with longer timeout if first attempt timed out
        if timeout < 180:
            logger.info("Retrying with extended timeout...")
            return train_arjs_marker(image_path, out_dir, slug, timeout=180)
        return False
    except Exception as e:
        logger.error(f"Exception in train_arjs_marker: {e}")
        return False

def build_pattern_marker(image_path: str, slug: str, media_root: str, marker_size_m=1.0):
    """
    Enhanced pattern marker generation with better error handling and optimizations.
    """
    try:
        img = Path(image_path)
        out_dir = Path(media_root) / "markers"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if image exists
        if not img.exists():
            logger.error(f"[marker] Image file not found: {img}")
            return None
        
        # Check file size (warn if too large)
        file_size = img.stat().st_size
        if file_size > 5 * 1024 * 1024:  # 5MB
            logger.warning(f"Large image file ({file_size / 1024 / 1024:.1f}MB) may cause timeout")
            
        # Check if Node.js is available
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("[marker] Node.js is not installed or not in PATH")
            return None
            
        # Try to generate AR.js markers using enhanced train_arjs_marker function
        logger.info(f"[marker] Generating AR.js markers for slug: {slug}")
        success = train_arjs_marker(str(img), str(out_dir), slug)
        
        if success:
            logger.info(f"[marker] Successfully generated markers for {slug}")
            return str(out_dir / f"{slug}.patt")  # Return pattern file path
        else:
            logger.warning(f"[marker] Failed to generate markers for {slug}")
            return None
            
    except Exception as e:
        logger.error(f"[marker] Error in build_pattern_marker: {str(e)}")
        return None

# Safe import for marker compiler
try:
    from .marker_compiler import MindARCompiler
except ImportError:
    MindARCompiler = None
    logging.warning("MindARCompiler not available")

def home(request):
    return render(request, "home.html")

def scanner(request):
    return render(request, "scanner.html")

def debug_markers(request, slug):
    """Debug view to check marker file status"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    # Path to the markers folder
    marker_dir = finders.find(f'markers/{slug}')
    
    debug_info = {
        'slug': slug,
        'marker_dir': marker_dir,
        'marker_dir_exists': os.path.exists(marker_dir) if marker_dir else False,
        'files': {}
    }
    
    required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]
    
    if marker_dir:
        for filename in required_files:
            filepath = os.path.join(marker_dir, filename)
            debug_info['files'][filename] = {
                'exists': os.path.exists(filepath),
                'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
            }
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        debug_info['files'][filename]['content_preview'] = f.read(200)
                except:
                    debug_info['files'][filename]['content_preview'] = 'Binary file or read error'
    
    return JsonResponse(debug_info, indent=2)

def upload_view(request):
    """Enhanced upload view with improved marker generation handling"""
    if request.method == 'POST':
        form = ARExperienceForm(request.POST, request.FILES)

        if form.is_valid():
            try:
                with transaction.atomic():
                    experience = form.save(commit=False)
                    
                    # Ensure slug is generated
                    if not experience.slug:
                        base_slug = slugify(experience.title) or f"exp-{uuid.uuid4().hex[:8]}"
                        counter = 1
                        slug = base_slug
                        while ARExperience.objects.filter(slug=slug).exists():
                            slug = f"{base_slug}-{counter}"
                            counter += 1
                        experience.slug = slug
                    
                    experience.save()

                # Enhanced Pattern Marker Generation
                marker_message = "No image provided"
                try:
                    if experience.image:
                        logger.info(f"Starting enhanced marker generation for {experience.slug}")
                        patt_path = build_pattern_marker(
                            image_path=experience.image.path,
                            slug=experience.slug,
                            media_root=settings.MEDIA_ROOT,
                            marker_size_m=float(form.cleaned_data.get("marker_size", 1.0))
                        )
                        
                        if patt_path:
                            logger.info(f"Enhanced marker generation successful for {experience.slug}")
                            experience.marker_generated = True
                            marker_message = "Markers generated successfully"
                        else:
                            logger.warning(f"Enhanced marker generation failed for {experience.slug}")
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

                # QR Code Generation with better error handling
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

                # Success message with marker status
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

    # GET request or form errors
    form = ARExperienceForm() if request.method == 'GET' else form
    
    context = {
        'form': form,
        'experiences': ARExperience.objects.all().order_by('-created_at'),
        'new_experience_slug': request.GET.get('new'),
        'marker_generated': True,
    }
    
    return render(request, 'upload.html', context)

def ar_experience_view(request, experience_id):
    """Enhanced AR experience viewer with better error handling"""
    try:
        experience = ARExperience.objects.get(id=experience_id)

        # Prepare context with full URLs
        context = {
            'experience': experience,
            'video_url': experience.video.url if experience.video else None,
            'marker_url': experience.image.url if experience.image else None,
            'title': experience.title,
            'description': experience.description,
            'slug': experience.slug,
            'base_url': settings.BASE_URL,
        }

        return render(request, 'ar_viewer.html', context)
        
    except ARExperience.DoesNotExist:
        messages.error(request, f'AR Experience with ID {experience_id} not found.')
        return redirect('upload')
    except Exception as e:
        logger.error(f"Error loading AR experience {experience_id}: {str(e)}")
        messages.error(request, 'Error loading AR experience. Please try again.')
        return redirect('upload')

def ar_experience_by_slug(request, slug):
    """AR experience viewer accessible by slug"""
    try:
        experience = ARExperience.objects.get(slug=slug)

        # Prepare context with full URLs
        context = {
            'experience': experience,
            'video_url': experience.video.url if experience.video else None,
            'marker_url': experience.image.url if experience.image else None,
            'title': experience.title,
            'description': experience.description,
            'slug': experience.slug,
            'base_url': settings.BASE_URL,
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
    """Enhanced marker regeneration with better feedback"""
    if request.method == 'POST':
        experience = get_object_or_404(ARExperience, slug=slug)

        try:
            logger.info(f"Regenerating markers for {slug}")
            patt_path = build_pattern_marker(
                image_path=experience.image.path,
                slug=experience.slug,
                media_root=settings.MEDIA_ROOT
            )

            # Update the marker_generated field
            success = patt_path is not None
            experience.marker_generated = success
            experience.save(update_fields=['marker_generated'])

            if success:
                return JsonResponse({
                    'status': 'success', 
                    'message': f'Markers regenerated successfully for {slug}'
                })
            else:
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
    
    # Generate QR code
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
        'experience_url': experience_url
    })

def experience_view(request, slug):
    experience = get_object_or_404(ARExperience, slug=slug)
    
    # Build the base URL for static files
    base_url = getattr(settings, 'BASE_URL', 'http://127.0.0.1:8000')
    static_url = f"{base_url}/static/"

    # Path to marker files (without file extensions)
    marker_base_url = f"{static_url}markers/{slug}/{slug}"

    # Check if marker files exist
    marker_files_exist = False
    try:
        # Use django's static file finder instead of os.path.join for static files
        marker_dir = finders.find(f'markers/{slug}')
        
        if marker_dir:
            required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]
            marker_files_exist = all(
                os.path.exists(os.path.join(marker_dir, file)) for file in required_files
            )
        
        if not marker_files_exist:
            logger.warning(f"⚠️ Some marker files missing for {slug}, regenerating...")
            # Try to regenerate marker files
            patt_path = build_pattern_marker(
                image_path=experience.image.path,
                slug=experience.slug,
                media_root=settings.MEDIA_ROOT
            )
            
            if patt_path:
                logger.info(f"✅ Successfully regenerated marker files for {slug}")
                marker_files_exist = True
            else:
                logger.error(f"❌ Failed to regenerate marker files for {slug}")
                marker_files_exist = False
            
    except Exception as e:
        logger.error(f"❌ Error checking marker files for {slug}: {e}")
        marker_files_exist = False
    
    context = {
        "experience": experience,
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "base_url": base_url,
    }
    
    return render(request, "experience.html", context)

def marker_status_api(request, slug):
    """API endpoint to check marker generation status"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        
        # Check if marker files actually exist
        marker_dir = Path(settings.MEDIA_ROOT) / "markers"
        required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]
        
        files_status = {}
        all_exist = True
        
        for filename in required_files:
            filepath = marker_dir / filename
            exists = filepath.exists()
            size = filepath.stat().st_size if exists else 0
            
            files_status[filename] = {
                'exists': exists,
                'size': size,
                'path': str(filepath)
            }
            
            if not exists:
                all_exist = False
        
        return JsonResponse({
            'slug': slug,
            'marker_generated': experience.marker_generated,
            'files_exist': all_exist,
            'files': files_status,
            'can_regenerate': bool(experience.image)
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Remove the old train_arjs_marker_robust function - we're using the enhanced version above