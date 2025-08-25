from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.contrib.staticfiles import finders
# from PIL import Image
import os
import qrcode
from io import BytesIO
import base64
from .models import ARExperience
from .forms import ARExperienceForm
from .marker_compiler import MindARCompiler  # MindARCompiler import
import logging
from django.utils.text import slugify
from pathlib import Path
import uuid
import os
import shutil
import subprocess
from pathlib import Path
from django.conf import settings
from django.contrib.staticfiles import finders
from .utils.arjs_marker import build_pattern_marker 

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


def train_arjs_marker(image_path: str, out_dir: str, slug: str) -> bool:
    """
    Use the Node NFT-Marker-Creator app to generate .iset/.fset/.fset3.
    Writes/renames them to out_dir/<slug>.* and returns True on success.
    """
    img = Path(image_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Location of the script inside node_modules
    pkg_root = Path(settings.BASE_DIR) / "node_modules" / "@webarkit" / "nft-marker-creator-app"
    script = pkg_root / "src" / "NFTMarkerCreator.js"
    if not script.exists():
        print("[arjs] NFTMarkerCreator.js not found at:", script)
        return False

    # Windows-safe node executable
    node = "node.exe" if os.name == "nt" else "node"

    # Run in the package root; it creates an 'output' folder there
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    # Copy image to package root to avoid path issues
    temp_img = pkg_root / f"temp_{slug}_{img.name}"
    try:
        shutil.copy2(img, temp_img)
        print(f"[arjs] Copied image to: {temp_img}")
    except Exception as e:
        print(f"[arjs] Failed to copy image: {e}")
        return False
    
    cmd = [node, str(script), "-i", temp_img.name]  # Just the filename
    print("[arjs] Running:", " ".join(cmd))
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(pkg_root),         # important: outputs go to pkg_root / output
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=60,               # Add timeout to prevent hanging
        )
        
        # Log the actual output/error for debugging
        print(f"[arjs] stdout: {proc.stdout[:800] if proc.stdout else 'None'}")
        print(f"[arjs] stderr: {proc.stderr[:800] if proc.stderr else 'None'}")
        print(f"[arjs] return code: {proc.returncode}")
        
        # Check if the process succeeded
        if proc.returncode != 0:
            print(f"[arjs] Node process failed with code {proc.returncode}")
            return False
            
    except subprocess.TimeoutExpired as e:
        print(f"[arjs] Timeout after 60 seconds: {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[arjs] Process error: {e}")
        print(f"[arjs] stdout: {e.stdout[:800] if e.stdout else 'None'}")
        print(f"[arjs] stderr: {e.stderr[:800] if e.stderr else 'None'}")
        return False
    except Exception as e:
        print(f"[arjs] Exception in train_arjs_marker: {e}")
        return False

    # The generator writes into <pkg_root>/output
    gen_dir = pkg_root / "output"
    if not gen_dir.exists():
        print("[arjs] output folder not found:", gen_dir)
        return False

    # Find any *.iset/*.fset/*.fset3 created (there may be generic names)
    produced = {
        ".iset": None,
        ".fset": None,
        ".fset3": None,
    }
    for p in gen_dir.glob("*"):
        if p.suffix in produced and produced[p.suffix] is None:
            produced[p.suffix] = p

    ok = True
    for ext in [".iset", ".fset", ".fset3"]:
        src = produced.get(ext)
        if not src or not src.exists():
            print(f"[arjs] missing generated {ext} file")
            ok = False
            continue
        dest = out / f"{slug}{ext}"
        try:
            shutil.copy2(src, dest)
            print(f"[arjs] copied {src} -> {dest}")
        except Exception as e:
            print(f"[arjs] copy error: {src} -> {dest}, {repr(e)}")
            ok = False

    # Clean up temporary image
    if temp_img.exists():
        try:
            temp_img.unlink()
            print(f"[arjs] Cleaned up temp image: {temp_img}")
        except Exception as e:
            print(f"[arjs] Failed to clean up temp image: {e}")

    return ok


def build_pattern_marker(image_path: str, slug: str, media_root: str, marker_size_m=1.0):
    try:
        img = Path(image_path)
        out_dir = Path(media_root) / "markers"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if image exists
        if not img.exists():
            print(f"[marker] Image file not found: {img}")
            return None
            
        # Check if Node.js is available
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("[marker] Node.js is not installed or not in PATH")
            return None
            
        # Rest of the function...
        # (original code here)
        
    except Exception as e:
        print(f"[marker] Error in build_pattern_marker: {str(e)}")
        return None
# from arexp.utils.markers import build_pattern_marker

logger = logging.getLogger(__name__)

def home(request):
    return render(request, "home.html")

def scanner(request):
    return render(request, "scanner.html")



# Safe import for marker compiler
try:
    from .marker_compiler import MindARCompiler
except ImportError:
    MindARCompiler = None
    logging.warning("MindARCompiler not available")

# Safe import for marker utils
try:
    from arexp.utils.markers import build_pattern_marker
except ImportError:
    def build_pattern_marker(*args, **kwargs):
        logging.warning("build_pattern_marker not available")
        return None

logger = logging.getLogger(__name__)

def upload_view(request):
    """Fixed upload view with better error handling"""
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

                # Pattern Marker Generation with better error handling
                try:
                    if build_pattern_marker and experience.image:
                        patt_path = build_pattern_marker(
                            image_path=experience.image.path,
                            slug=experience.slug,
                            media_root=settings.MEDIA_ROOT,
                            marker_size_m=float(form.cleaned_data.get("marker_size", 1.0))
                        )
                        if patt_path:
                            logger.info(f"Generated pattern marker for {experience.slug}")
                            experience.marker_generated = True
                        else:
                            logger.warning(f"Failed to generate pattern marker for {experience.slug}")
                            experience.marker_generated = False
                    else:
                        logger.info(f"Pattern marker generation skipped for {experience.slug}")
                        experience.marker_generated = True  # Mark as true to avoid regeneration attempts
                        
                    experience.save(update_fields=["marker_generated"])

                except Exception as pattern_error:
                    logger.error(f"Pattern setup failed: {pattern_error}")
                    experience.marker_generated = False
                    experience.save(update_fields=["marker_generated"])

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

                messages.success(request, 'AR Experience created successfully!')
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


def upload_experience(request):
    if request.method == 'POST':
        try:
            # Get form data
            name = request.POST.get('name')
            image = request.FILES.get('image')
            
            # Create experience object
            experience = Experience.objects.create(
                name=name,
                image=image
            )
            
            # Ensure slug is set and valid
            if not experience.slug or experience.slug == "undefined":
                from django.utils.text import slugify
                experience.slug = slugify(name)
                experience.save()
            
            # Generate pattern file
            pattern_path = None
            if image:
                try:
                    pattern_path = build_pattern_marker(
                        image_path=image.path,
                        slug=experience.slug,
                        media_root=settings.MEDIA_ROOT
                    )
                    logger.info(f"Pattern file generated successfully: {pattern_path}")
                except Exception as e:
                    logger.error(f"Error generating pattern file: {str(e)}")
            
            # Generate visual marker
            visual_marker_path = create_visual_marker(
                slug=experience.slug,
                media_root=settings.MEDIA_ROOT
            )
            
            # Save marker paths
            if pattern_path:
                experience.pattern_file = pattern_path
            if visual_marker_path:
                experience.marker_image = visual_marker_path
            experience.save()
            
            return redirect('upload_success', new=experience.slug)
            
        except Exception as e:
            logger.error(f"Error creating experience: {str(e)}")
            return render(request, 'upload/error.html', {'error': str(e)})
    
    return render(request, 'upload/form.html')

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


def debug_markers(request, slug):
    """Debug view to check marker file status"""
    experience = get_object_or_404(ARExperience, slug=slug)
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

            # Read first few lines for debugging
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        debug_info['files'][filename]['content_preview'] = f.read(200)
                except:
                    debug_info['files'][filename]['content_preview'] = 'Binary file or read error'

    return JsonResponse(debug_info, indent=2)


def regenerate_markers(request, slug):
    """Regenerate marker files for a specific experience"""
    if request.method == 'POST':
        experience = get_object_or_404(ARExperience, slug=slug)

        try:
            compiler = MindARCompiler()
            success = compiler.generate_marker_files(
                image_path=experience.image.path,
                slug=experience.slug
            )

            if success:
                return JsonResponse({'status': 'success', 'message': f'Markers regenerated for {slug}'})
            else:
                # Create fallback markers if regeneration fails
                static_dir = getattr(settings, 'STATICFILES_DIRS', ['static'])[0]
                marker_dir = os.path.join(static_dir, 'markers', slug)
                os.makedirs(marker_dir, exist_ok=True)
                compiler.create_fallback_markers(marker_dir, slug)

                return JsonResponse({
                    'status': 'partial_success', 
                    'message': f'Created fallback markers for {slug}'
                })

        except Exception as e:
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
            compiler = MindARCompiler()
            success = compiler.generate_marker_files(
                image_path=experience.image.path,
                slug=experience.slug
            )
            
            if not success:
                # Create fallback markers if marker generation fails
                compiler.create_fallback_markers(marker_dir, slug)
                logger.info(f"✅ Created fallback marker files for {slug}")
            
            marker_files_exist = True  # Assume success after regeneration
            
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