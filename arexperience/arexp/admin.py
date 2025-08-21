from django.contrib import admin
from .models import Upload

@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    list_display = ("id", "image_name", "video_name", "uploaded_at")
    readonly_fields = ("uploaded_at",)

    def image_name(self, obj): return getattr(obj.image, "name", "")
    def video_name(self, obj): return getattr(obj.video, "name", "")
