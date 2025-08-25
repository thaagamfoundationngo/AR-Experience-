# models.py - Fixed version without lambda functions

import os
import uuid
from django.db import models
from django.utils.text import slugify
from django.core.validators import FileExtensionValidator
from django.conf import settings

def validate_and_truncate_filename(instance, filename):
    """Truncate long filenames and ensure they're safe"""
    name, ext = os.path.splitext(filename)
    
    # Remove or replace problematic characters
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
    
    # Truncate if too long (leave room for extension and unique suffix)
    max_length = 80  # Conservative limit
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    # Add unique suffix if name is generic or empty
    if not safe_name or safe_name.lower() in ['undefined', 'untitled', 'image', 'video']:
        safe_name = f"file_{uuid.uuid4().hex[:8]}"
    
    return f"{safe_name}{ext}"

def _delete_file(path):
    """Safely delete a file"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not delete file {path}: {e}")

# Upload path functions - these replace the lambda functions
def upload_image_path(instance, filename):
    """Generate upload path for Upload model images"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"images/{clean_filename}"

def upload_video_path(instance, filename):
    """Generate upload path for Upload model videos"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"videos/{clean_filename}"

def upload_mind_file_path(instance, filename):
    """Generate upload path for Upload model mind files"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"targets/{clean_filename}"

def ar_marker_image_path(instance, filename):
    """Generate upload path for AR Experience marker images"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"markers/{clean_filename}"

def ar_video_path(instance, filename):
    """Generate upload path for AR Experience videos"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"videos/{clean_filename}"

def ar_model_path(instance, filename):
    """Generate upload path for AR Experience 3D models"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"models/{clean_filename}"

def ar_qr_path(instance, filename):
    """Generate upload path for AR Experience QR codes"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"qrcodes/{clean_filename}"

class Upload(models.Model):
    """Upload model with proper filename handling"""
    target_name = models.CharField(
        max_length=100, 
        help_text="8th Wall Image Target name", 
        db_index=True
    )

    image = models.ImageField(
        upload_to=upload_image_path,
        blank=False, 
        null=False
    )
    
    video = models.FileField(
        upload_to=upload_video_path,
        validators=[FileExtensionValidator(allowed_extensions=["mp4", "mov", "webm"])],
        blank=False, 
        null=False
    )

    mind_file = models.FileField(
        upload_to=upload_mind_file_path, 
        blank=True, 
        null=True
    )

    slug = models.SlugField(max_length=100, unique=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def image_url(self):
        return self.image.url if self.image else ""

    def video_url(self):
        return self.video.url if self.video else ""

    def __str__(self):
        return f"Upload {self.id} | {self.target_name} | {self.slug}"

    def get_absolute_url(self):
        from django.urls import reverse
        return reverse("experience_slug", args=[self.slug])

    def get_qr_url(self):
        from django.urls import reverse
        return reverse("qr_slug", args=[self.slug])

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = self.generate_unique_slug()
        super().save(*args, **kwargs)

    @staticmethod
    def generate_unique_slug():
        while True:
            s = uuid.uuid4().hex[:8]
            if not Upload.objects.filter(slug=s).exists():
                return s

    def delete(self, *args, **kwargs):
        img_path = self.image.path if self.image else None
        vid_path = self.video.path if self.video else None
        mind_path = self.mind_file.path if self.mind_file else None
        super().delete(*args, **kwargs)
        _delete_file(img_path)
        _delete_file(vid_path)
        _delete_file(mind_path)


class ARExperience(models.Model):
    """AR Experience model with proper file handling"""
    
    title = models.CharField(max_length=200, help_text="Name of the AR experience")
    slug = models.SlugField(max_length=50, unique=True, blank=True)
    description = models.TextField(blank=True, help_text="Description of the AR experience")
    
    # Files with proper filename handling using named functions
    image = models.ImageField(
        upload_to=ar_marker_image_path,
        help_text="Image to be used as AR marker"
    )
    
    video = models.FileField(
        upload_to=ar_video_path,
        blank=True, 
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=["mp4", "mov", "webm"])],
        help_text="Optional video to display in AR experience"
    )
    
    model_file = models.FileField(
        upload_to=ar_model_path,
        blank=True, 
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=["glb", "gltf", "usdz"])],
        help_text="Optional 3D model file"
    )
    
    content_text = models.TextField(blank=True, help_text="Optional text content to display")
    content_url = models.URLField(blank=True, help_text="URL to additional content")
    marker_size = models.FloatField(default=1.0, help_text="Size of the marker in AR space")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True, help_text="Whether this experience is active")
    
    qr_code = models.FileField(
        upload_to=ar_qr_path, 
        blank=True, 
        null=True
    )
    
    marker_generated = models.BooleanField(default=False, help_text="Whether marker files have been generated")

    class Meta:
        ordering = ['-created_at']
        verbose_name = "AR Experience"
        verbose_name_plural = "AR Experiences"
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        # Generate slug if not set
        if not self.slug:
            self.slug = slugify(self.name)
            # Ensure uniqueness
            count = 1
            original_slug = self.slug
            while Experience.objects.filter(slug=self.slug).exists():
                self.slug = f"{original_slug}-{count}"
                count += 1
        super().save(*args, **kwargs)
    
    @property
    def marker_files_exist(self):
        """Check if marker files exist"""
        try:
            from django.conf import settings
            static_dir = (settings.STATICFILES_DIRS[0] 
                         if getattr(settings, "STATICFILES_DIRS", None) 
                         else os.path.join(settings.BASE_DIR, "static"))

            marker_dir = os.path.join(static_dir, "markers", self.slug)
            required = [f"{self.slug}.iset", f"{self.slug}.fset", f"{self.slug}.fset3"]

            return all(os.path.exists(os.path.join(marker_dir, f)) for f in required)
        except Exception:
            return False
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('experience_view', kwargs={'slug': self.slug})
    
    def get_qr_url(self):
        from django.urls import reverse
        return reverse('qr_view', kwargs={'slug': self.slug})
    
    def delete(self, *args, **kwargs):
        """Clean up files when deleting"""
        img_path = self.image.path if self.image else None
        vid_path = self.video.path if self.video else None
        model_path = self.model_file.path if self.model_file else None
        qr_path = self.qr_code.path if self.qr_code else None
        
        super().delete(*args, **kwargs)
        
        # Clean up files
        for path in [img_path, vid_path, model_path, qr_path]:
            _delete_file(path)