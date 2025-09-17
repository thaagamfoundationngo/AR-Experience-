from django.contrib import admin
from django.contrib.admin.models import LogEntry, DELETION
from django.contrib.contenttypes.models import ContentType
from django.utils.encoding import force_str
from .models import Upload, ARExperience

# Register Upload model (simple registration)
@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    list_display = ['target_name', 'slug', 'uploaded_at']
    list_filter = ['uploaded_at']
    search_fields = ['target_name', 'slug']
    readonly_fields = ['slug', 'uploaded_at']

# Register ARExperience with custom delete handling
@admin.register(ARExperience)
class ARExperienceAdmin(admin.ModelAdmin):
    list_display = ['title', 'slug', 'marker_generated', 'is_active', 'created_at']
    list_filter = ['marker_generated', 'is_active', 'created_at']
    search_fields = ['title', 'slug', 'description']
    readonly_fields = ['slug', 'created_at', 'updated_at', 'user']
    actions = ['delete_selected_objects']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('title', 'slug', 'description')
        }),
        ('Media Files', {
            'fields': ('image', 'video', 'model_file')
        }),
        ('AR Settings', {
            'fields': ('marker_generated', 'is_active', 'view_count', 'visibility'),
        }),
        ('System Info', {
            'fields': ('user', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def delete_selected_objects(self, request, queryset):
        """Custom delete action to avoid log_action conflicts"""
        count = queryset.count()
        for obj in queryset:
            obj.delete()
        self.message_user(request, f"Successfully deleted {count} AR experience(s).")
    delete_selected_objects.short_description = "Delete selected AR experiences"
    
    def delete_queryset(self, request, queryset):
        """Custom delete to avoid log_action conflicts"""
        for obj in queryset:
            obj.delete()
    
    def delete_model(self, request, obj):
        """Custom single object delete"""
        obj.delete()
    
    def get_actions(self, request):
        """Remove default delete action"""
        actions = super().get_actions(request)
        if 'delete_selected' in actions:
            del actions['delete_selected']
        return actions
    
    def save_model(self, request, obj, form, change):
        """Set user automatically on save"""
        if not change and hasattr(obj, 'user') and not obj.user:
            obj.user = request.user
        super().save_model(request, obj, form, change)
