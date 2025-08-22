# views.py - Updated version with MindAR integration

from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.templatetags.static import static
from django.urls import reverse
import os
from django.contrib import messages
import qrcode
from io import BytesIO
import base64
from .models import ARExperience
from .forms import ARExperienceForm
from .marker_compiler import MindARCompiler  # Add this import
import logging
logger = logging.getLogger(__name__)

def home(request):
    return render(request, "home.html")

def scanner(request):
    return render(request, "scanner.html")



def upload_view(request):
    if request.method == 'POST':
        print("🔄 POST request received")
        form = ARExperienceForm(request.POST, request.FILES)
        
        if form.is_valid():
            print("✅ Form is valid")
            
            try:
                with transaction.atomic():
                    # Save with commit=False to get instance
                    experience = form.save(commit=False)
                    print(f"Form saved (commit=False), title: {experience.title}")
                    
                    # Generate slug if not present
                    if not experience.slug:
                        experience.slug = experience.generate_unique_slug()
                        print(f"Generated slug: {experience.slug}")
                    
                    # Save to database
                    experience.save()
                    print(f"ARExperience saved to database with ID: {experience.id}")
                
                # Verify the save
                try:
                    saved_experience = ARExperience.objects.get(id=experience.id)
                    print(f"Verification: Record exists in DB - Title: {saved_experience.title}, Slug: {saved_experience.slug}")
                except ARExperience.DoesNotExist:
                    print(f"CRITICAL: Record with ID {experience.id} not found in database!")
                    raise Exception("Database record was not saved properly")
                
                # Generate QR code
                try:
                    qr_url = request.build_absolute_uri(experience.get_absolute_url())
                    qr_code_path = os.path.join(settings.MEDIA_ROOT, 'qrcodes', f'{experience.slug}.png')
                    os.makedirs(os.path.dirname(qr_code_path), exist_ok=True)
                    qr = qrcode.make(qr_url)
                    qr.save(qr_code_path)
                    print(f"QR code generated at {qr_code_path}")
                    experience.qr_code = f'qrcodes/{experience.slug}.png'
                    experience.save(update_fields=['qr_code'])
                    print(f"QR code path saved to database")
                except Exception as qr_error:
                    print(f"QR code generation error: {qr_error}")
                
                # Generate marker files
                try:
                    compiler = MindARCompiler()
                    marker_success = compiler.generate_marker_files(
                        image_path=experience.image.path,
                        slug=experience.slug
                    )
                    if not marker_success:
                        static_dir = settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else 'static'
                        marker_dir = os.path.join(static_dir, 'markers', experience.slug)
                        os.makedirs(marker_dir, exist_ok=True)
                        compiler.create_fallback_markers(marker_dir, experience.slug)
                        print(f"Using fallback markers for {experience.slug}")
                except Exception as marker_error:
                    print(f"Marker generation error: {marker_error}")
                
                messages.success(request, 'AR Experience created successfully!')
                return redirect(f'/upload/?new={experience.slug}')
                
            except Exception as save_error:
                print(f"Critical error in upload process: {save_error}")
                import traceback
                traceback.print_exc()
                form.add_error(None, f"Upload error: {str(save_error)}")
        
        else:
            print("Form validation failed:")
            for field, errors in form.errors.items():
                print(f"  {field}: {errors}")
            messages.error(request, 'Please correct the errors below.')
        
        return render(request, 'upload.html', {
            'form': form,
            'experiences': ARExperience.objects.all().order_by('-created_at'),
            'new_experience_slug': request.GET.get('new'),
        })
    
    else:
        form = ARExperienceForm()
        
    return render(request, 'upload.html', {
        'form': form,
        'experiences': ARExperience.objects.all().order_by('-created_at'),
        'new_experience_slug': request.GET.get('new')
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

# Add a debug view to check marker files
def debug_markers(request, slug):
    """Debug view to check marker file status"""
    experience = get_object_or_404(ARExperience, slug=slug)
    
    static_dir = getattr(settings, 'STATICFILES_DIRS', ['static'])[0] if hasattr(settings, 'STATICFILES_DIRS') else 'static'
    marker_dir = os.path.join(static_dir, 'markers', slug)
    
    debug_info = {
        'slug': slug,
        'marker_dir': marker_dir,
        'marker_dir_exists': os.path.exists(marker_dir),
        'files': {}
    }
    
    required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]
    
    if os.path.exists(marker_dir):
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

# Add a regenerate markers view
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
                # Try fallback
                static_dir = getattr(settings, 'STATICFILES_DIRS', ['static'])[0] if hasattr(settings, 'STATICFILES_DIRS') else 'static'
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