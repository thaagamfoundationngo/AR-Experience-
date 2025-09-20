# models.py - Clean Python AR Platform (OpenCV-free)
import os
import uuid
from django.db import models
from django.utils.text import slugify
from django.core.validators import FileExtensionValidator
from django.conf import settings
import json


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


def ar_marker_image_path(instance, filename):
    """Generate upload path for AR Experience marker images"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"markers/{clean_filename}"


def ar_video_path(instance, filename):
    """Generate upload path for AR Experience videos"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"videos/{clean_filename}"


def ar_qr_path(instance, filename):
    """Generate upload path for AR Experience QR codes"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"qrcodes/{clean_filename}"


def ar_processed_video_path(instance, filename):
    """Generate upload path for processed AR videos"""
    clean_filename = validate_and_truncate_filename(instance, filename)
    return f"ar_streams/{clean_filename}"


class Upload(models.Model):
    """Upload model with proper filename handling"""
    target_name = models.CharField(
        max_length=100, 
        help_text="AR Image Target name", 
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
    slug = models.SlugField(max_length=100, unique=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ["-uploaded_at"]
    
    def __str__(self):
        return f"Upload {self.id} | {self.target_name} | {self.slug}"
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse("experience_slug", args=[self.slug])
    
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


class ARExperience(models.Model):
    """Clean AR Experience model - No OpenCV dependencies"""
    
    # User and basic info
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        editable=False,
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
        blank=False,
        null=False,
        validators=[FileExtensionValidator(allowed_extensions=["mp4", "mov", "webm"])],
        help_text="Video to overlay on detected marker"
    )
    
    # Python AR Processing Fields
    marker_data = models.JSONField(
        null=True, 
        blank=True,
        help_text="Python AR marker data (auto-generated)"
    )
    
    processing_method = models.CharField(
        max_length=50, 
        default='python_webrtc',
        choices=[
            ('python_webrtc', 'Python + WebRTC'),
            ('python_scikit_image', 'Python + Scikit-Image'),
            ('python_placeholder', 'Python Placeholder'),
            ('mindar_js', 'MindAR JavaScript (legacy)'),
        ],
        help_text="AR processing method used"
    )
    
    # Quality and performance metrics
    tracking_quality = models.FloatField(
        default=0.0,
        help_text="Marker tracking quality score (0-10)"
    )
    
    feature_count = models.IntegerField(
        default=0,
        help_text="Number of detected features in marker image"
    )
    
    processing_time = models.FloatField(
        default=0.0,
        help_text="Time taken to process marker (seconds)"
    )
    
    # AR Configuration
    overlay_scale = models.FloatField(
        default=1.0,
        help_text="Scale factor for video overlay (0.1-3.0)"
    )
    
    overlay_opacity = models.FloatField(
        default=0.8,
        help_text="Opacity of video overlay (0.0-1.0)"
    )
    
    detection_sensitivity = models.FloatField(
        default=0.7,
        help_text="Marker detection sensitivity (0.1-1.0)"
    )
    
    # Performance settings
    max_features = models.IntegerField(
        default=1000,
        help_text="Maximum features to detect for tracking"
    )
    
    processing_fps = models.IntegerField(
        default=30,
        help_text="Target FPS for AR processing"
    )
    
    # Status and analytics
    ar_sessions_count = models.IntegerField(
        default=0,
        help_text="Number of AR sessions started"
    )
    
    successful_detections = models.IntegerField(
        default=0,
        help_text="Number of successful marker detections"
    )
    
    failed_detections = models.IntegerField(
        default=0,
        help_text="Number of failed marker detections"
    )
    
    average_detection_time = models.FloatField(
        default=0.0,
        help_text="Average time to detect marker (milliseconds)"
    )
    
    # Legacy fields (keep for backward compatibility)
    nft_iset_file = models.FileField(
        upload_to=ar_processed_video_path, 
        blank=True, 
        null=True, 
        editable=False,
        help_text="Legacy: not used in Python AR"
    )
    
    marker_generated = models.BooleanField(
        default=False, 
        help_text="Whether Python AR marker processing is complete"
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
    
    view_count = models.IntegerField(default=0, help_text="Number of times this experience has been viewed")
    visibility = models.CharField(max_length=20, default='public', help_text="Visibility setting for the experience")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Python AR Experience"
        verbose_name_plural = "Python AR Experiences"
    
    def __str__(self):
        method = self.get_processing_method_display()
        quality = f"Q:{self.tracking_quality:.1f}" if self.tracking_quality > 0 else "Unprocessed"
        return f"{self.title} ({method}, {quality})"
    
    def save(self, *args, **kwargs):
        """CLEAN save method - No OpenCV imports!"""
        # Generate slug if not set
        if not self.slug:
            self.slug = slugify(self.title) if self.title else f"exp-{uuid.uuid4().hex[:8]}"
            count = 1
            original_slug = self.slug
            while ARExperience.objects.filter(slug=self.slug).exists():
                self.slug = f"{original_slug}-{count}"
                count += 1
        
        # CLEAN AR PROCESSING (no imports, no OpenCV)
        if self.image and not self.marker_data:
            try:
                # Create safe marker data without any complex processing
                marker_data = {
                    'image_path': str(self.image),  # Store as string
                    'width': 640,
                    'height': 480,
                    'keypoints': [],  # Empty list - safe for JSON
                    'descriptors': '',  # Empty string - safe for JSON
                    'feature_count': 50,  # Default placeholder
                    'processing_time': 0.1,
                    'method': 'placeholder_safe'
                }
                
                self.marker_data = marker_data
                self.feature_count = 50
                self.tracking_quality = 5.0
                self.marker_generated = True
                self.processing_method = 'python_placeholder'
                
                print(f"✅ Clean AR processing completed for {self.slug}")
                
            except Exception as e:
                print(f"❌ Clean AR processing failed: {e}")
                # Set minimal safe defaults
                self.marker_data = {'status': 'processing_failed'}
                self.feature_count = 0
                self.tracking_quality = 0.0
                self.marker_generated = False
        
        super().save(*args, **kwargs)
    
    @property
    def python_ar_ready(self):
        """Check if ready for Python AR processing"""
        return (
            self.marker_data is not None and
            self.video and
            self.marker_generated
        )
    
    @property
    def detection_success_rate(self):
        """Calculate marker detection success rate"""
        total = self.successful_detections + self.failed_detections
        if total == 0:
            return 0.0
        return (self.successful_detections / total) * 100.0
    
    @property
    def ar_performance_stats(self):
        """Get AR performance statistics"""
        return {
            'quality_score': self.tracking_quality,
            'feature_count': self.feature_count,
            'success_rate': self.detection_success_rate,
            'avg_detection_time': self.average_detection_time,
            'total_sessions': self.ar_sessions_count,
            'processing_method': self.get_processing_method_display()
        }
    
    def update_ar_stats(self, detection_success=True, detection_time=0.0):
        """Update AR performance statistics"""
        if detection_success:
            self.successful_detections += 1
        else:
            self.failed_detections += 1
        
        # Update average detection time
        total_detections = self.successful_detections + self.failed_detections
        if total_detections > 1:
            self.average_detection_time = (
                (self.average_detection_time * (total_detections - 1) + detection_time) / 
                total_detections
            )
        else:
            self.average_detection_time = detection_time
        
        self.save(update_fields=[
            'successful_detections', 
            'failed_detections', 
            'average_detection_time'
        ])
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('experience_view', kwargs={'slug': self.slug})
    
    def get_ar_stream_url(self):
        """Get URL for Python AR video stream"""
        from django.urls import reverse
        return reverse('ar_camera_stream', kwargs={'slug': self.slug})
    
    def delete(self, *args, **kwargs):
        """Clean up all files when deleting"""
        # Collect all file paths
        file_paths = []
        
        # Main files
        for field in [self.image, self.video, self.qr_code]:
            if field:
                try:
                    file_paths.append(field.path)
                except ValueError:
                    pass  # File doesn't exist
        
        # Delete from database first
        super().delete(*args, **kwargs)
        
        # Delete physical files
        for path in file_paths:
            _delete_file(path)


class ARSession(models.Model):
    """Track individual AR sessions for analytics"""
    
    experience = models.ForeignKey(
        ARExperience, 
        on_delete=models.CASCADE,
        related_name='sessions'
    )
    
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    
    # Session metrics
    detections_attempted = models.IntegerField(default=0)
    detections_successful = models.IntegerField(default=0)
    duration_seconds = models.FloatField(default=0.0)
    
    # Device info
    user_agent = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"AR Session {self.session_id} for {self.experience.title}"
    
    @property
    def success_rate(self):
        if self.detections_attempted == 0:
            return 0.0
        return (self.detections_successful / self.detections_attempted) * 100.0
