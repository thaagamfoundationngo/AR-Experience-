# models.py - Cleaned and updated version
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

# Upload path functions
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

def ar_nft_path(instance, filename):
    """Generate upload path for NFT marker files"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"nft_markers/{clean_filename}"

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
    """AR Experience model with NFT file storage"""
    user = models.ForeignKey(
    settings.AUTH_USER_MODEL,
    on_delete=models.CASCADE,
    null=True,       # ✅ Allow NULL - no user required!
    blank=True,      # ✅ Allow empty in forms
    editable=False,  # ✅ Hide from admin/forms
    help_text="Optional user who created this AR experience"
)
    
    title = models.CharField(max_length=200, help_text="Name of the AR experience")
    slug = models.SlugField(max_length=50, unique=True, blank=True)
    description = models.TextField(blank=True, help_text="Description of the AR experience")
    
    # Main files
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
    
    # NFT Marker Files
    nft_iset_file = models.FileField(
        upload_to=ar_nft_path, 
        blank=True, 
        null=True, 
        editable=False
    )
    
    nft_fset_file = models.FileField(
        upload_to=ar_nft_path, 
        blank=True, 
        null=True, 
        editable=False
    )
    
    nft_fset3_file = models.FileField(
        upload_to=ar_nft_path, 
        blank=True, 
        null=True, 
        editable=False
    )
    
    # Other fields
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
    view_count = models.IntegerField(default=0, help_text="Number of times this experience has been viewed")
    visibility = models.CharField(max_length=20, default='public', help_text="Visibility setting for the experience")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "AR Experience"
        verbose_name_plural = "AR Experiences"
        #exclude = ['user']
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        # Generate slug if not set
        if not self.slug:
            self.slug = slugify(self.title) if self.title else f"exp-{uuid.uuid4().hex[:8]}"
            # Ensure uniqueness
            count = 1
            original_slug = self.slug
            while ARExperience.objects.filter(slug=self.slug).exists():
                self.slug = f"{original_slug}-{count}"
                count += 1
        
        # Check if this is a new object
        is_new = self._state.adding
        
        # Call parent save
        super().save(*args, **kwargs)
        
        # If new object, refresh from DB to get the ID
        if is_new:
            self.refresh_from_db()
    
    @property
    def marker_files_exist(self):
        """Check if marker files exist in filesystem"""
        try:
            from pathlib import Path
            marker_dir = Path(settings.MEDIA_ROOT) / "markers" / self.slug
            required = [f"{self.slug}.iset", f"{self.slug}.fset", f"{self.slug}.fset3"]
            return all((marker_dir / f).exists() for f in required)
        except Exception:
            return False
    
    @property
    def nft_files_in_database(self):
        """Check if NFT files are stored in database"""
        return bool(self.nft_iset_file and self.nft_fset_file and self.nft_fset3_file)
    
    @property
    def marker_status(self):
        """Get comprehensive marker status"""
        return {
            'generated': self.marker_generated,
            'files_exist': self.marker_files_exist,
            'db_stored': self.nft_files_in_database,
            'ready': self.marker_generated and (self.marker_files_exist or self.nft_files_in_database)
        }
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('experience_view', kwargs={'slug': self.slug})
    
    def get_qr_url(self):
        from django.urls import reverse
        return reverse('qr_view', kwargs={'slug': self.slug})
    
    def clean_nft_files(self):
        """Clean up NFT files from database and filesystem"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Delete physical files
        for field_name in ['nft_iset_file', 'nft_fset_file', 'nft_fset3_file']:
            field = getattr(self, field_name)
            if field:
                try:
                    _delete_file(field.path)
                except Exception as e:
                    logger.warning(f"Could not delete {field_name}: {e}")
        
        # Clear database references
        update_fields = []
        for field_name in ['nft_iset_file', 'nft_fset_file', 'nft_fset3_file']:
            if getattr(self, field_name) is not None:
                setattr(self, field_name, None)
                update_fields.append(field_name)
        
        # Only save if there are changes
        if update_fields:
            try:
                self.save(update_fields=update_fields)
            except Exception as e:
                logger.error(f"Error in clean_nft_files: {e}")
    
    def update_nft_files(self, iset_path=None, fset_path=None, fset3_path=None):
        """Update NFT file paths in database safely"""
        import logging
        logger = logging.getLogger(__name__)
        
        update_fields = []
        
        # Update iset file
        if iset_path is not None and self.nft_iset_file != iset_path:
            self.nft_iset_file = iset_path
            update_fields.append('nft_iset_file')
        
        # Update fset file
        if fset_path is not None and self.nft_fset_file != fset_path:
            self.nft_fset_file = fset_path
            update_fields.append('nft_fset_file')
        
        # Update fset3 file
        if fset3_path is not None and self.nft_fset3_file != fset3_path:
            self.nft_fset3_file = fset3_path
            update_fields.append('nft_fset3_file')
        
        # Only save if there are changes
        if update_fields:
            try:
                self.save(update_fields=update_fields)
                return True
            except Exception as e:
                logger.error(f"Error updating NFT files: {e}")
                return False
        
        return True  # No changes needed
    
    def delete(self, *args, **kwargs):
        """Clean up all files when deleting"""
        # Collect all file paths
        file_paths = []
        
        # Main files
        for field in [self.image, self.video, self.model_file, self.qr_code]:
            if field:
                file_paths.append(field.path)
        
        # NFT files
        for field in [self.nft_iset_file, self.nft_fset_file, self.nft_fset3_file]:
            if field:
                file_paths.append(field.path)
        
        # Delete from database
        super().delete(*args, **kwargs)
        
        # Delete physical files
        for path in file_paths:
            _delete_file(path)