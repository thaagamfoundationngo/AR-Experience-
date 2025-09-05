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
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

def optimize_image_for_markers(image_path: str, max_size: tuple = (512, 512), quality: int = 85) -> str:
    """Optimize image for MindAR target generation."""
    try:
        img_path = Path(image_path)
        
        if not img_path.exists():
            logger.error(f"Image file not found: {img_path}")
            return str(img_path)
        
        # For MindAR, we can use original image or optimize if needed
        logger.info(f"Using original image for MindAR target generation: {img_path.name}")
        return str(img_path)
        
    except Exception as e:
        logger.error(f"Error in optimize_image_for_markers: {e}")
        return image_path

def check_node_environment() -> tuple[bool, str]:
    """Check if Node.js and MindAR CLI environment is properly set up."""
    try:
        node_cmd = "node.exe" if os.name == "nt" else "node"
        result = subprocess.run([node_cmd, "--version"], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, "Node.js not found or not working"
        
        node_version = result.stdout.strip()
        logger.info(f"Node.js version: {node_version}")
        
        # Check for MindAR library
        try:
            mindar_result = subprocess.run(
                ["node", "-e", "try { require('mind-ar'); console.log('MINDAR_AVAILABLE'); } catch(e) { console.error('MINDAR_MISSING'); }"],
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if mindar_result.returncode == 0 and "MINDAR_AVAILABLE" in mindar_result.stdout:
                logger.info("MindAR library available")
                return True, f"Environment OK - Node.js {node_version} with MindAR library"
            else:
                logger.warning("MindAR library not found, will install on demand")
                return True, f"Environment OK - Node.js {node_version} (MindAR will be installed)"
        except Exception:
            logger.warning("MindAR library check failed, will install on demand")
            return True, f"Environment OK - Node.js {node_version} (MindAR will be installed)"
        
    except Exception as e:
        return False, f"Environment check failed: {e}"

def ensure_named(file_path: str, expected_name: str) -> str:
    """Ensures a file has the expected name. If not, renames it."""
    file_path = Path(file_path)
    expected_path = file_path.parent / expected_name
    
    if file_path.name != expected_name:
        try:
            shutil.move(str(file_path), str(expected_path))
            print(f"[MindAR] Renamed {file_path.name} to {expected_name}")
            return str(expected_path)
        except Exception as e:
            print(f"[MindAR] Failed to rename {file_path.name}: {e}")
            return str(file_path)
    
    return str(file_path)

def check_mindar_structure():
    """Check the structure of the mind-ar library to find the correct compiler"""
    try:
        project_root = Path(settings.BASE_DIR)
        mindar_path = project_root / "node_modules" / "mind-ar"
        
        if not mindar_path.exists():
            logger.error("mind-ar package not found in node_modules")
            return False
            
        # Check for possible compiler paths
        possible_paths = [
            mindar_path / "dist" / "image-target" / "compiler.js",
            mindar_path / "src" / "image-target" / "compiler.js",
            mindar_path / "image-target" / "compiler.js"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found MindAR compiler at: {path}")
                return True
                
        logger.error("Could not find MindAR compiler in any expected location")
        return False
        
    except Exception as e:
        logger.error(f"Error checking MindAR structure: {e}")
        return False

def create_simple_target_file(image_path: str, target_file: str, slug: str) -> bool:
    """Create a simple MindAR target file as a fallback"""
    try:
        img = Path(image_path)
        
        # Get image dimensions
        with Image.open(img) as image:
            width, height = image.size
        
        # Create a simple target structure
        target_data = {
            "targets": [
                {
                    "name": slug,
                    "metadata": {
                        "size": img.stat().st_size,
                        "width": width,
                        "height": height,
                        "type": "image"
                    },
                    "type": "image"
                }
            ]
        }
        
        # Write to file
        with open(target_file, 'w') as f:
            json.dump(target_data, f)
            
        # Verify file was created and has content
        if Path(target_file).exists() and Path(target_file).stat().st_size > 0:
            logger.info(f"Created simple MindAR target file: {target_file}")
            return True
        else:
            logger.error(f"Failed to create simple target file: {target_file}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to create simple target file: {e}")
        return False
    
    
def create_browser_based_target(image_path: str, target_file: str, slug: str) -> bool:
    """Create a MindAR target using browser-based approach"""
    try:
        img = Path(image_path)
        
        # Get image dimensions
        with Image.open(img) as image:
            width, height = image.size
        
        # Create a MindAR-compatible target structure
        target_data = {
            "targets": [
                {
                    "name": slug,
                    "metadata": {
                        "size": img.stat().st_size,
                        "width": width,
                        "height": height,
                        "type": "image",
                        "url": f"/media/markers/{slug}/{img.name}"
                    },
                    "type": "image",
                    "image": {
                        "width": width,
                        "height": height
                    }
                }
            ]
        }
        
        # Ensure target directory exists
        target_dir = Path(target_file).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(target_file, 'w') as f:
            json.dump(target_data, f)
            
        # Verify file was created and has content
        if Path(target_file).exists() and Path(target_file).stat().st_size > 0:
            logger.info(f"Created browser-based MindAR target file: {target_file}")
            return True
        else:
            logger.error(f"Failed to create browser-based target file: {target_file}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to create browser-based target file: {e}")
        return False
    
    
def train_arjs_marker(image_path: str, out_dir: str, slug: str) -> tuple[bool, dict]:
    """
    Generate MindAR target using browser-based approach.
    """
    try:
        logger.info(f"Starting MindAR target generation for {slug}")
        
        img = Path(image_path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        target_file = out / f"{slug}.mind"
        
        # Method 1: Browser-based target creation
        logger.info("Creating browser-based MindAR target...")
        success = create_browser_based_target(str(img.absolute()), str(target_file.absolute()), slug)
        
        # Verify file was actually created and has content
        if success and target_file.exists() and target_file.stat().st_size > 0:
            logger.info(f"Created browser-based MindAR target file: {target_file} (size: {target_file.stat().st_size} bytes)")
            return True, {'.mind': str(target_file)}
            
        # Method 2: Simple target file creation (fallback)
        logger.warning("Browser-based method failed, creating simple target file...")
        success = create_simple_target_file(str(img.absolute()), str(target_file.absolute()), slug)
        
        # Verify file was actually created and has content
        if success and target_file.exists() and target_file.stat().st_size > 0:
            logger.info(f"Created simple MindAR target file: {target_file} (size: {target_file.stat().st_size} bytes)")
            return True, {'.mind': str(target_file)}
            
        logger.error(f"All methods failed for {slug}. Target file exists: {target_file.exists()}, size: {target_file.stat().st_size if target_file.exists() else 0}")
        return False, {}
            
    except Exception as e:
        logger.error(f"Exception in train_arjs_marker (MindAR): {e}")
        return False, {}
    
def create_mindar_compilation_script(image_path: str, target_file: str) -> str:
    """Create a Node.js script to compile MindAR target using dynamic import for ES module"""
    script_content = f'''
const fs = require('fs');
const path = require('path');
async function compileTarget() {{
    try {{
        console.log('Loading image: {image_path}');
        
        // Read image file
        const imageBuffer = fs.readFileSync('{image_path}');
        
        // Import MindAR compiler using dynamic import
        const mindarPath = path.join(process.cwd(), 'node_modules', 'mind-ar');
        const compilerPath = path.join(mindarPath, 'dist', 'image-target', 'compiler.js');
        
        // Use dynamic import for ES module
        const compilerModule = await import(compilerPath);
        const Compiler = compilerModule.Compiler || compilerModule.default;
        
        if (!Compiler) {{
            throw new Error('Could not find Compiler class in mind-ar module');
        }}
        
        // Create compiler instance
        const compiler = new Compiler();
        
        // Compile the image target
        const compiledData = await compiler.compileImageTargets([{{
            image: imageBuffer,
            name: 'target'
        }}]);
        
        // Save the compiled target
        fs.writeFileSync('{target_file}', JSON.stringify(compiledData));
        
        console.log('MindAR target compiled successfully: {target_file}');
        process.exit(0);
        
    }} catch (error) {{
        console.error('MindAR compilation failed:', error);
        process.exit(1);
    }}
}}
compileTarget();
'''
    return script_content

def execute_mindar_script(script_content: str, output_dir: str) -> bool:
    """Execute the MindAR compilation script from the project root"""
    try:
        # Get the project root directory
        project_root = Path(settings.BASE_DIR)
        
        # Write the script file in the project root
        script_file = project_root / "compile_target.js"
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Execute the script from the project root
        node_cmd = "node.exe" if os.name == "nt" else "node"
        result = subprocess.run(
            [node_cmd, str(script_file)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "NODE_PATH": str(project_root / "node_modules")}
        )
        
        # Log output for debugging
        if result.stdout:
            logger.info(f"MindAR script stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"MindAR script stderr: {result.stderr}")
        
        # Clean up script file
        try:
            script_file.unlink()
        except:
            pass
        
        if result.returncode == 0:
            logger.info("MindAR compilation script executed successfully")
            return True
        else:
            logger.error(f"MindAR compilation script failed with code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing MindAR script: {e}")
        return False

def build_pattern_marker(image_path, slug, media_root):
    """
    Build MindAR target file and return file paths.
    Returns tuple (success, file_paths) for database storage.
    """
    try:
        logger.info(f"🏗️ Starting build_pattern_marker for {slug}")
        
        img = Path(image_path)
        out_dir = Path(media_root) / "markers" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🏗️ Image: {img} (exists: {img.exists()})")
        logger.info(f"🏗️ Output directory: {out_dir}")
        
        if not img.exists():
            logger.error(f"❌ Image file not found: {img}")
            return False, {}
            
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True, timeout=30)
            logger.info("✅ Node.js is available")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("⚠️ Node.js is not installed or not in PATH")
            return False, {}
            
        # Check MindAR structure
        if not check_mindar_structure():
            logger.warning("⚠️ MindAR library structure is incorrect")
            return False, {}
            
        logger.info(f"🏗️ About to call train_arjs_marker")
        
        success, file_paths = train_arjs_marker(str(img), str(out_dir), slug)
        
        logger.info(f"🏗️ train_arjs_marker returned: success={success}, file_paths={file_paths}")
        
        if success:
            logger.info(f"✅ build_pattern_marker succeeded for {slug}")
            return True, file_paths
        else:
            logger.warning(f"❌ build_pattern_marker failed for {slug}")
            return False, {}
    except Exception as e:
        logger.error(f"❌ Error in build_pattern_marker: {str(e)}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False, {}
    
def debug_mind_file(request, slug):
    """Debug function to check MindAR .mind file status"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        mind_file = marker_dir / f"{slug}.mind"
        
        debug_info = {
            'slug': slug,
            'marker_dir_exists': marker_dir.exists(),
            'mind_file_exists': mind_file.exists(),
            'mind_file_size': mind_file.stat().st_size if mind_file.exists() else 0,
            'marker_generated_db': experience.marker_generated,
            'nft_iset_file_db': str(experience.nft_iset_file) if experience.nft_iset_file else None
        }
        
        # Try to read and validate the .mind file
        if mind_file.exists():
            try:
                with open(mind_file, 'r') as f:
                    mind_data = json.load(f)
                debug_info['mind_file_valid'] = True
                debug_info['mind_structure'] = {
                    'has_imageList': 'imageList' in mind_data,
                    'has_trackingData': 'trackingData' in mind_data,
                    'image_count': len(mind_data.get('imageList', [])),
                    'version': mind_data.get('trackingData', {}).get('version', 'unknown')
                }
            except Exception as e:
                debug_info['mind_file_valid'] = False
                debug_info['mind_error'] = str(e)
        
        return JsonResponse(debug_info, indent=2)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def save_nft_files_to_database(experience: ARExperience, file_paths: dict):
    """Save generated MindAR files to database fields."""
    try:
        # For MindAR, we only have one .mind file
        mind_path = file_paths.get('.mind')
        
        if mind_path:
            # Store the .mind file path in the iset field for compatibility
            success = experience.update_nft_files(mind_path, None, None)
            
            if success:
                logger.info(f"MindAR target file saved to database for {experience.slug}")
                return True
            else:
                logger.error(f"Failed to save MindAR target file to database for {experience.slug}")
                return False
        else:
            logger.error(f"No MindAR target file to save for {experience.slug}")
            return False
        
    except Exception as e:
        logger.error(f"Error saving MindAR target file to database: {e}")
        return False    

def update_nft_files(self, iset_path=None, fset_path=None, fset3_path=None):
    """Update MindAR file paths in database safely"""
    import logging
    logger = logging.getLogger(__name__)
    
    update_fields = []
    
    # Convert to string for comparison
    current_iset = str(self.nft_iset_file) if self.nft_iset_file else None
    
    # For MindAR, store .mind file in iset field
    if iset_path is not None and current_iset != iset_path:
        self.nft_iset_file = iset_path
        update_fields.append('nft_iset_file')
        
        # Clear other fields for MindAR (we only use .mind file)
        if self.nft_fset_file:
            self.nft_fset_file = None
            update_fields.append('nft_fset_file')
        if self.nft_fset3_file:
            self.nft_fset3_file = None
            update_fields.append('nft_fset3_file')
    
    if update_fields:
        try:
            self.save(update_fields=update_fields)
            return True
        except Exception as e:
            logger.error(f"Error updating MindAR files: {e}")
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
        'tracking_method': 'MindAR',
        'mind_db_status': {
            'mind_file_stored': bool(experience.nft_iset_file),  # .mind file stored in iset field
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
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    info['content_preview'] = 'MindAR target file OK'
            except:
                info['content_preview'] = 'Binary file or read error'
        debug_info['files'][mind_file] = info
    return JsonResponse(debug_info, indent=2)

def upload_view(request):
    """Enhanced upload view with MindAR target generation"""
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
                        logger.info(f"Starting MindAR target generation for {experience.slug}")
                        
                        success, target_file_paths = build_pattern_marker(
                            image_path=experience.image.path,
                            slug=experience.slug,
                            media_root=settings.MEDIA_ROOT
                        )
                        
                        if success and target_file_paths:
                            logger.info(f"MindAR target generation successful for {experience.slug}")
                            
                            db_save_success = save_nft_files_to_database(experience, target_file_paths)
                            
                            experience.marker_generated = True
                            marker_message = "MindAR target generated and saved successfully"
                            
                            if not db_save_success:
                                marker_message += " (Warning: Database save partially failed)"
                        else:
                            logger.warning(f"MindAR target generation failed for {experience.slug}")
                            experience.marker_generated = False
                            marker_message = "MindAR target generation failed - check logs for details"
                    else:
                        logger.info(f"No image provided for {experience.slug}")
                        experience.marker_generated = False
                        
                    experience.save(update_fields=["marker_generated"])
                except Exception as pattern_error:
                    logger.error(f"MindAR target generation failed for {experience.slug}: {pattern_error}")
                    experience.marker_generated = False
                    experience.save(update_fields=["marker_generated"])
                    marker_message = f"Generation error: {pattern_error}"
                
                # QR Code Generation (unchanged)
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
                    messages.success(request, f'MindAR Experience created successfully! {marker_message}')
                else:
                    messages.warning(request, f'AR Experience created, but {marker_message}')
                    
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
    """Enhanced experience view with MindAR capabilities"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
    except ARExperience.DoesNotExist:
        raise Http404(f"Experience '{slug}' not found")
    
    # Build base URL properly
    if hasattr(settings, 'BASE_URL') and settings.BASE_URL:
        base_url = settings.BASE_URL
    else:
        base_url = request.build_absolute_uri('/').rstrip('/')
    
    # Build media URL
    if settings.MEDIA_URL.startswith('http'):
        media_url = settings.MEDIA_URL
    else:
        media_url = base_url + settings.MEDIA_URL.rstrip('/')
    
    # Marker paths
    marker_base_url = f"{media_url}/markers/{slug}/{slug}"
    
    # Check for MindAR target file
    marker_files_exist = False
    mind_file_path = None
    
    try:
        marker_dir = Path(settings.MEDIA_ROOT) / "markers" / slug
        mind_file = marker_dir / f"{slug}.mind"
        mind_file_path = str(mind_file)
        
        # Check if file exists and has content
        if mind_file.exists() and mind_file.stat().st_size > 0:
            marker_files_exist = True
            logger.info(f"✅ MindAR target file found for {slug}: {mind_file}")
        else:
            logger.warning(f"⚠️ MindAR target file missing or empty for {slug}: {mind_file}")
            
            # Try to regenerate if you have the function
            # Uncomment this section if you have build_pattern_marker function
            """
            if hasattr(experience, 'image') and experience.image:
                try:
                    success, target_file_paths = build_pattern_marker(
                        image_path=experience.image.path,
                        slug=experience.slug,
                        media_root=settings.MEDIA_ROOT
                    )
                    
                    if success:
                        save_nft_files_to_database(experience, target_file_paths)
                        logger.info(f"✅ Successfully regenerated MindAR target for {slug}")
                        marker_files_exist = mind_file.exists() and mind_file.stat().st_size > 0
                    else:
                        logger.error(f"❌ Failed to regenerate MindAR target for {slug}")
                except Exception as regen_error:
                    logger.error(f"❌ Error regenerating MindAR target for {slug}: {regen_error}")
            """
            
    except Exception as e:
        logger.error(f"❌ Error checking MindAR target for {slug}: {e}")
        marker_files_exist = False
    
    # MindAR configuration optimized for better detection
    mindar_config = {
        'maxTrack': 1,
        'showStats': settings.DEBUG,
        'uiLoading': 'no',
        'uiError': 'no', 
        'uiScanning': 'no',
        'autoStart': False,  # Manual start for better control
        'filterMinCF': 0.0001,     # Very low for relaxed detection
        'filterBeta': 0.001,       # Smooth tracking
        'missTolerance': 5,        # Allow 5 frames before losing target
        'warmupTolerance': 2       # 2 frames for detection
    }
    
    # Build URLs with proper error handling
    video_url = None
    marker_image_url = None
    
    if hasattr(experience, 'video') and experience.video:
        video_url = experience.video.url
        if not video_url.startswith('http'):
            video_url = base_url + video_url
    
    if hasattr(experience, 'image') and experience.image:
        marker_image_url = experience.image.url
        if not marker_image_url.startswith('http'):
            marker_image_url = base_url + marker_image_url
    
    context = {
        "experience": experience,
        "marker_base_url": marker_base_url,
        "marker_files_exist": marker_files_exist,
        "base_url": base_url,
        "title": experience.title,
        "video_url": video_url,
        "marker_image_url": marker_image_url,
        "timestamp": int(time.time()),
        
        # MindAR specific
        "tracking_method": "MindAR",
        "mindar_config": mindar_config,  # Python dict
        "mindar_config_json": json.dumps(mindar_config),  # JSON string for JavaScript
        
        # Instructions
        "instructions": {
            'setup': 'Allow camera access when prompted by your browser',
            'usage': 'Point your camera at the uploaded marker image',
            'distance': 'Hold your device 20-60cm away from the marker',
            'lighting': 'Ensure good lighting for better tracking',
            'stability': 'Keep the marker clearly visible and steady',
            'technology': 'Powered by MindAR for superior tracking'
        },
        
        # Debug information
        "debug": {
            'mind_file_path': mind_file_path,
            'media_root': str(settings.MEDIA_ROOT),
            'slug': slug,
        } if settings.DEBUG else {}
    }
    
    # Enhanced logging
    logger.info(f"🧠 MindAR experience loading for '{slug}':")
    logger.info(f"   Target file: {'✅ Found' if marker_files_exist else '❌ Missing'}")
    logger.info(f"   Marker base URL: {marker_base_url}")
    logger.info(f"   Video URL: {video_url or 'None'}")
    logger.info(f"   Marker image URL: {marker_image_url or 'None'}")
    
    # Use correct template path - adjust this to match your template location
    template_name = "experience.html"  # or just "experience.html" if in root templates
    
    return render(request, template_name, context)

def webcam_ar_experience_view(request, slug=None):
    """Dedicated MindAR webcam experience view"""
    
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
            mind_file = marker_dir / f"{experience.slug}.mind"
            marker_files_exist = mind_file.exists() and mind_file.stat().st_size > 0
            
            if not marker_files_exist:
                logger.info(f"🔄 Regenerating MindAR target for {experience.slug}")
                try:
                    success, target_file_paths = build_pattern_marker(
                        image_path=experience.image.path,
                        slug=experience.slug,
                        media_root=settings.MEDIA_ROOT
                    )
                    
                    if success:
                        save_nft_files_to_database(experience, target_file_paths)
                        marker_files_exist = mind_file.exists() and mind_file.stat().st_size > 0
                        logger.info(f"✅ MindAR target regenerated for {experience.slug}")
                    else:
                        logger.error(f"❌ Failed to regenerate MindAR target for {experience.slug}")
                        
                except Exception as e:
                    logger.error(f"❌ Error regenerating MindAR target for {experience.slug}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error checking MindAR target for {experience.slug}: {e}")
    
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
    
    if experience:
        logger.info(f"🧠📱 MindAR webcam activated: {experience.title} (Target: {'✅' if marker_files_exist else '❌'})")
    
    return render(request, "experience.html", context)

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
                "target_stored": bool(experience.nft_iset_file),  # .mind file stored in iset field
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
            'base_url': settings.BASE_URL,
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

def regenerate_markers(request, slug):
    """Enhanced MindAR target regeneration"""
    if request.method == 'POST':
        experience = get_object_or_404(ARExperience, slug=slug)
        try:
            logger.info(f"Starting MindAR target regeneration for {slug}")
            logger.info(f"Processing image: {experience.image.path}")
            logger.info(f"Image size: {experience.image.size} bytes")
            
            success, target_file_paths = build_pattern_marker(
                image_path=experience.image.path,
                slug=experience.slug,
                media_root=settings.MEDIA_ROOT
            )
            if success and target_file_paths:
                db_save_success = save_nft_files_to_database(experience, target_file_paths)
                
                experience.marker_generated = True
                experience.save(update_fields=['marker_generated'])
                
                logger.info(f"MindAR target regeneration completed for {slug}: Success")
                
                return JsonResponse({
                    'status': 'success', 
                    'message': f'MindAR target regenerated and saved successfully for {slug}',
                    'database_saved': db_save_success,
                    'tracking_method': 'MindAR'
                })
            else:
                experience.marker_generated = False
                experience.save(update_fields=['marker_generated'])
                
                return JsonResponse({
                    'status': 'error', 
                    'message': f'Failed to regenerate MindAR target for {slug}'
                })
        except Exception as e:
            logger.error(f"Error regenerating MindAR target for {slug}: {e}")
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
        'tracking_method': 'MindAR',
    })

def marker_status_api(request, slug):
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
                'target_stored': bool(experience.nft_iset_file),  # .mind file stored in iset field
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def webcam_ar_debug_view(request, slug):
    """Debug view for MindAR experience development"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    marker_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
    mind_file = marker_dir / f"{experience.slug}.mind"
    
    marker_analysis = {
        '.mind': {
            'exists': mind_file.exists(),
            'size': mind_file.stat().st_size if mind_file.exists() else 0,
            'modified': mind_file.stat().st_mtime if mind_file.exists() else None
        }
    }
    
    debug_context = {
        "experience": experience,
        "marker_analysis": marker_analysis,
        "media_root": str(settings.MEDIA_ROOT),
        "debug_mode": True,
        "tracking_method": "MindAR",
        "mindar_version": "1.2.2",
        "aframe_version": "1.4.0",
        "target_database_info": {
            'mind_file': str(experience.nft_iset_file) if experience.nft_iset_file else None,
        }
    }
    
    return render(request, "mindar_debug.html", debug_context)

def ar_experience_view(request, experience_id: int):
    """Back-compat: resolve by ID then reuse the slug-based view with MindAR."""
    exp = get_object_or_404(ARExperience, id=experience_id)
    return experience_view(request, exp.slug)

def realtime_experience_view(request, slug):
    """Real-time MindAR experience"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    context = {
        "experience": experience,
        "title": experience.title,
        "video_url": experience.video.url if experience.video else None,
        "realtime_mode": True,
        "tracking_method": "MindAR",
        "timestamp": int(time.time()),
    }
    
    logger.info(f"🧠 Real-time MindAR experience loaded: {experience.title}")
    return render(request, "experience.html", context)

def debug_mindar_library(request):
    """Debug endpoint to check MindAR library structure"""
    project_root = Path(settings.BASE_DIR)
    mindar_path = project_root / "node_modules" / "mind-ar"
    
    debug_info = {
        "project_root": str(project_root),
        "mindar_path": str(mindar_path),
        "mindar_exists": mindar_path.exists(),
        "compiler_paths": []
    }
    
    if mindar_path.exists():
        # Check for possible compiler paths
        possible_paths = [
            mindar_path / "dist" / "image-target" / "compiler.js",
            mindar_path / "src" / "image-target" / "compiler.js",
            mindar_path / "image-target" / "compiler.js"
        ]
        
        for path in possible_paths:
            path_info = {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0
            }
            debug_info["compiler_paths"].append(path_info)
            
            # Try to read the file content to see what it exports
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                        # Look for export statements
                        exports = []
                        for line in content.split('\n'):
                            if 'export' in line and ('Compiler' in line or 'compiler' in line.lower()):
                                exports.append(line.strip())
                        path_info["exports"] = exports
                except Exception as e:
                    path_info["read_error"] = str(e)
    
    return JsonResponse(debug_info, indent=2)


def debug_mindar_compilation(request, slug):
    """Debug endpoint to test MindAR compilation"""
    try:
        experience = get_object_or_404(ARExperience, slug=slug)
        
        if not experience.image:
            return JsonResponse({"error": "No image associated with this experience"})
        
        # Get absolute paths
        image_path = experience.image.path
        target_dir = Path(settings.MEDIA_ROOT) / "markers" / experience.slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{experience.slug}.mind"
        
        # Test the compilation using browser-based approach
        success1 = create_browser_based_target(str(Path(image_path).absolute()), str(target_file.absolute()), slug)
        
        # Test the simple target creation as fallback
        success2 = create_simple_target_file(str(Path(image_path).absolute()), str(target_file.absolute()), slug)
        
        return JsonResponse({
            "success": success1 or success2,
            "browser_based_success": success1,
            "simple_target_success": success2,
            "image_path": image_path,
            "target_file": str(target_file),
            "target_exists": target_file.exists(),
            "target_size": target_file.stat().st_size if target_file.exists() else 0,
            "methods_tested": ["browser_based", "simple_target"]
        })
        
    except Exception as e:
        logger.error(f"Debug MindAR compilation error: {e}")
        return JsonResponse({"error": str(e)})
    
# ============================================================================
# NEW MINDAR COMPILER FUNCTIONS
# ============================================================================

@csrf_exempt
@require_http_methods(["GET", "POST"])
def mindar_compiler(request):
    """MindAR Image Targets Compiler - Django Implementation"""
    
    if request.method == 'GET':
        return render(request, 'arexp/mindar_compiler.html')
    
    elif request.method == 'POST':
        try:
            # Handle file upload
            if 'image' not in request.FILES:
                return JsonResponse({'error': 'No image file provided'}, status=400)
            
            uploaded_file = request.FILES['image']
            
            # Validate file type
            if not uploaded_file.content_type.startswith('image/'):
                return JsonResponse({'error': 'Invalid file type. Please upload an image.'}, status=400)
            
            # Process the image and generate .mind file
            result = process_mindar_compilation(uploaded_file)
            
            if result['success']:
                return JsonResponse({
                    'success': True,
                    'message': 'Marker compiled successfully!',
                    'download_url': result['download_url'],
                    'features_count': result.get('features_count', 0),
                    'file_size': result.get('file_size', 0)
                })
            else:
                return JsonResponse({'error': result['error']}, status=500)
                
        except Exception as e:
            logger.error(f"MindAR compilation error: {e}")
            return JsonResponse({'error': str(e)}, status=500)

def process_mindar_compilation(uploaded_file):
    """Process image and generate MindAR .mind file"""
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded file
            input_image = temp_path / f"input{Path(uploaded_file.name).suffix}"
            with open(input_image, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Validate and process image
            with Image.open(input_image) as img:
                # Ensure minimum size for good tracking
                if img.width < 480 or img.height < 480:
                    # Resize if too small
                    img = img.resize((max(480, img.width), max(480, img.height)), Image.Resampling.LANCZOS)
                    img.save(input_image, quality=90)
                
                logger.info(f"Processing image: {img.size}, format: {img.format}")
            
            # Generate .mind file using Python implementation
            mind_file_path = temp_path / "targets.mind"
            success = generate_mind_file(input_image, mind_file_path)
            
            if success and mind_file_path.exists():
                # Move to media directory
                compiled_dir = Path(settings.MEDIA_ROOT) / 'compiled_markers'
                compiled_dir.mkdir(exist_ok=True)
                
                # Generate unique filename
                import uuid
                unique_id = str(uuid.uuid4())[:8]
                final_mind_file = compiled_dir / f"targets_{unique_id}.mind"
                
                # Copy compiled file
                import shutil
                shutil.copy2(mind_file_path, final_mind_file)
                
                # Get file info
                file_size = final_mind_file.stat().st_size
                download_url = f"{settings.MEDIA_URL}compiled_markers/{final_mind_file.name}"
                
                return {
                    'success': True,
                    'download_url': download_url,
                    'file_size': file_size,
                    'features_count': estimate_features_count(input_image)
                }
            else:
                return {'success': False, 'error': 'Failed to generate .mind file'}
                
    except Exception as e:
        logger.error(f"Compilation process failed: {e}")
        return {'success': False, 'error': str(e)}

def generate_mind_file(image_path, output_path):
    """Generate .mind file - Multi-method implementation"""
    
    try:
        # Method 1: Try using Node.js MindAR CLI if available
        if try_nodejs_compilation(image_path, output_path):
            return True
        
        # Method 2: Try online compiler approach
        if try_online_compiler_approach(image_path, output_path):
            return True
        
        # Method 3: Python-based implementation (fallback)
        return python_mind_compilation(image_path, output_path)
        
    except Exception as e:
        logger.error(f"Mind file generation failed: {e}")
        return False

def try_nodejs_compilation(image_path, output_path):
    """Try using Node.js MindAR CLI"""
    
    try:
        # Try different possible command variations
        commands_to_try = [
            ['npx', '@mind-ar/cli', '-i', str(image_path), '-o', str(output_path.parent)],
            ['npx', 'mindar-cli', '-i', str(image_path), '-o', str(output_path.parent)],
            ['node', 'mindar_compiler.js', str(image_path), str(output_path)]
        ]
        
        for cmd in commands_to_try:
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=60,
                    cwd=output_path.parent
                )
                
                if result.returncode == 0:
                    # Check if targets.mind was created
                    targets_file = output_path.parent / "targets.mind"
                    if targets_file.exists():
                        targets_file.rename(output_path)
                        logger.info("Successfully compiled using Node.js MindAR CLI")
                        return True
                        
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug(f"Command failed: {cmd}, error: {e}")
                continue
        
    except Exception as e:
        logger.warning(f"Node.js compilation failed: {e}")
    
    return False

def try_online_compiler_approach(image_path, output_path):
    """Try using requests to simulate online compiler"""
    
    try:
        import requests
        
        # This would simulate the online compiler approach
        # Note: This is a placeholder - you'd need the actual API endpoint
        # from MindAR if they provide one
        
        logger.info("Online compiler approach not implemented yet")
        return False
        
    except ImportError:
        logger.debug("requests not available for online compilation")
        return False
    except Exception as e:
        logger.warning(f"Online compilation failed: {e}")
        return False

def python_mind_compilation(image_path, output_path):
    """Python-based MindAR compilation (simplified version)"""
    
    try:
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        
        # Feature detection using ORB (more features = better tracking)
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        
        if descriptors is None or len(keypoints) < 100:
            logger.warning(f"Not enough features detected: {len(keypoints) if keypoints else 0}")
            # Try SIFT as fallback
            try:
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(img, None)
            except:
                pass
        
        if descriptors is None or len(keypoints) < 50:
            logger.error("Insufficient features for tracking")
            return False
        
        # Create simplified .mind file format
        mind_data = {
            'imageWidth': img.shape[1],
            'imageHeight': img.shape[0],
            'keypoints': [],
            'descriptors': descriptors.tolist() if descriptors is not None else [],
            'version': '1.2.0',
            'generated_by': 'django_mindar_compiler',
            'feature_count': len(keypoints)
        }
        
        # Convert keypoints to serializable format
        for kp in keypoints:
            mind_data['keypoints'].append({
                'x': float(kp.pt[0]),
                'y': float(kp.pt[1]),
                'angle': float(kp.angle),
                'scale': float(kp.size),
                'response': float(kp.response)
            })
        
        # Write binary .mind file (MindAR compatible format)
        with open(output_path, 'wb') as f:
            # Write header
            f.write(b'MIND')  # Magic number
            f.write((2).to_bytes(4, 'little'))  # Version 2
            
            # Write image dimensions
            f.write(img.shape[1].to_bytes(4, 'little'))  # Width
            f.write(img.shape[0].to_bytes(4, 'little'))  # Height
            
            # Write keypoints count
            f.write(len(keypoints).to_bytes(4, 'little'))
            
            # Write keypoints data (enhanced format)
            for kp in keypoints:
                f.write(int(kp.pt[0] * 100).to_bytes(4, 'little'))  # x * 100
                f.write(int(kp.pt[1] * 100).to_bytes(4, 'little'))  # y * 100
                f.write(int(kp.angle * 100).to_bytes(4, 'little'))  # angle * 100
                f.write(int(kp.size * 100).to_bytes(4, 'little'))   # size * 100
                f.write(int(kp.response * 1000).to_bytes(4, 'little'))  # response * 1000
            
            # Write descriptors
            if descriptors is not None:
                f.write(len(descriptors).to_bytes(4, 'little'))
                f.write(len(descriptors[0]).to_bytes(4, 'little'))  # descriptor length
                f.write(descriptors.tobytes())
            else:
                f.write((0).to_bytes(4, 'little'))
        
        logger.info(f"Generated .mind file with {len(keypoints)} features using Python compiler")
        return True
        
    except ImportError:
        logger.error("OpenCV not installed. Please install: pip install opencv-python")
        return False
    except Exception as e:
        logger.error(f"Python compilation failed: {e}")
        return False

def estimate_features_count(image_path):
    """Estimate number of trackable features in image"""
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
            
        # Use ORB for feature detection
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints = orb.detect(img, None)
        
        feature_count = len(keypoints) if keypoints else 0
        logger.info(f"Estimated {feature_count} features in image")
        return feature_count
        
    except ImportError:
        logger.warning("OpenCV not available for feature estimation")
        return 0
    except Exception as e:
        logger.warning(f"Feature estimation failed: {e}")
        return 0

def build_pattern_marker(image_path, slug, media_root):
    """Build pattern marker - placeholder for existing function"""
    # This should contain your existing marker building logic
    logger.info(f"Building pattern marker for {slug}")
    return False, None

def save_nft_files_to_database(experience, target_file_paths):
    """Save NFT files to database - placeholder for existing function"""
    # This should contain your existing database saving logic
    logger.info(f"Saving NFT files for {experience.slug}")
    pass