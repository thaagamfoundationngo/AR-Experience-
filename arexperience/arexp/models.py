import uuid
from django.db import models

class Upload(models.Model):
    image = models.ImageField(upload_to="images/")
    video = models.FileField(upload_to="videos/")
    mind_file = models.FileField(upload_to="targets/", blank=True, null=True)
    slug = models.SlugField(max_length=100, unique=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
   
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = self.generate_unique_slug()
        super().save(*args, **kwargs)
    
    def generate_unique_slug(self):
        """Generate a unique slug using UUID"""
        while True:
            slug = uuid.uuid4().hex[:8]
            if not Upload.objects.filter(slug=slug).exists():
                return slug
   
    def __str__(self):
        return self.slug