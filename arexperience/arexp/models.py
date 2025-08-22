# models.py
import os
import uuid
from django.db import models

from django.utils.text import slugify
import uuid
# app/models.py
import uuid
import os
from django.db import models
from django.urls import reverse
from django.core.validators import FileExtensionValidator

def _delete_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

class Upload(models.Model):
    # 8th Wall Studio-la create pannina image target name (e.g., "poster_001")
    target_name = models.CharField(max_length=100, help_text="8th Wall Image Target name", db_index=True)

    image = models.ImageField(
        upload_to="images/",
        blank=False, null=False
    )
    video = models.FileField(
        upload_to="videos/",
        validators=[FileExtensionValidator(allowed_extensions=["mp4", "mov", "webm"])],
        blank=False, null=False
    )

    # OPTIONAL: any exported target data / JSON you want to keep (not required for 8th Wall)
    mind_file = models.FileField(upload_to="targets/", blank=True, null=True)

    # Shareable short code for links (use in URL instead of pk)
    slug = models.SlugField(max_length=100, unique=True, blank=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    # ---- Convenience ----
    def image_url(self):
        return self.image.url if self.image else ""

    def video_url(self):
        return self.video.url if self.video else ""

    def __str__(self):
        # single __str__ only
        return f"Upload {self.id} | {self.target_name} | {self.slug}"

    def get_absolute_url(self):
        return reverse("experience_slug", args=[self.slug])

    def get_qr_url(self):
        return reverse("qr_slug", args=[self.slug])

    # ---- Slug generation ----
    def save(self, *args, **kwargs):
        creating = self._state.adding
        if not self.slug:
            self.slug = self.generate_unique_slug()
        super().save(*args, **kwargs)

    @staticmethod
    def generate_unique_slug():
        # 8-hex short id; regenerate if collision
        while True:
            s = uuid.uuid4().hex[:8]
            if not Upload.objects.filter(slug=s).exists():
                return s

    # ---- Optional: cleanup old files on replace ----
    def delete(self, *args, **kwargs):
        img_path = self.image.path if self.image else None
        vid_path = self.video.path if self.video else None
        mind_path = self.mind_file.path if self.mind_file else None
        super().delete(*args, **kwargs)
        _delete_file(img_path)
        _delete_file(vid_path)
        _delete_file(mind_path)


class ARExperience(models.Model):
    """Model for AR experiences with image markers"""
    
    # Basic fields
    title = models.CharField(max_length=200, help_text="Name of the AR experience")
    slug = models.SlugField(max_length=50, unique=True, blank=True, help_text="URL-friendly identifier")
    description = models.TextField(blank=True, help_text="Description of the AR experience")
    image = models.ImageField(upload_to='markers/')
    video = models.FileField(upload_to='videos/', blank=True, null=True)  # Add video field
    model_file = models.FileField(upload_to='models/', blank=True, null=True)
    content_text = models.TextField(blank=True)
    content_url = models.URLField(blank=True)
    marker_size = models.FloatField(default=1.0)
    created_at = models.DateTimeField(auto_now_add=True)
    qr_code = models.FileField(upload_to='qrcodes/', blank=True, null=True)
    # Image for marker generation
    image = models.ImageField(
        upload_to='markers/',
        help_text="Image to be used as AR marker (JPG, PNG recommended)"
    )
    
    # 3D Model or content
    model_file = models.FileField(
        upload_to='models/',
        blank=True,
        null=True,
        help_text="3D model file (GLB, GLTF, OBJ)"
    )
    
    # Additional content
    content_text = models.TextField(blank=True, help_text="Text content to display in AR")
    content_url = models.URLField(blank=True, help_text="URL to additional content")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True, help_text="Whether this experience is active")
    
    # MindAR settings
    marker_generated = models.BooleanField(default=False, help_text="Whether marker files have been generated")
    marker_size = models.FloatField(default=1.0, help_text="Size of the marker in AR space")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "AR Experience"
        verbose_name_plural = "AR Experiences"
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        if not self.title:
            raise ValueError("Title is required to generate a slug")
        if not self.slug:
            base_slug = slugify(self.title)
            if not base_slug:
                base_slug = f"experience-{uuid.uuid4().hex[:8]}"
        
            counter = 1
            slug = base_slug
            while ARExperience.objects.filter(slug=slug).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = slug
    
        super().save(*args, **kwargs)

    @property
    def marker_files_exist(self):
        """Check if marker files exist for this experience"""
        import os
        from django.conf import settings
        
        try:
            static_dir = getattr(settings, 'STATICFILES_DIRS', ['static'])[0] if hasattr(settings, 'STATICFILES_DIRS') else 'static'
            marker_dir = os.path.join(static_dir, 'markers', self.slug)
            
            required_files = [f"{self.slug}.iset", f"{self.slug}.fset", f"{self.slug}.fset3"]
            return all(
                os.path.exists(os.path.join(marker_dir, file)) 
                for file in required_files
            )
        except:
            return False
    
    def get_absolute_url(self):
        """Get the URL for this AR experience"""
        from django.urls import reverse
        return reverse('experience_view', kwargs={'slug': self.slug})
    
    def get_qr_url(self):
        """Get the QR code URL for this experience"""
        from django.urls import reverse
        return reverse('qr_view', kwargs={'slug': self.slug})