# arexp/forms.py - FIXED TO MATCH YOUR TEMPLATE
from django import forms
from .models import ARExperience, Upload

class ARExperienceForm(forms.ModelForm):
    """Simplified form matching your current template"""
    
    class Meta:
        model = ARExperience
        fields = ['title', 'description', 'image', 'video', 'marker_size']
        
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter AR experience title',
                'required': True
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Describe your AR experience (optional)'
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'required': True
            }),
            'video': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*',
                'required': True
            }),
            'marker_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.1',
                'max': '5.0',
                'step': '0.1',
                'value': '1.0'
            })
        }

    def clean_image(self):
        """Validate uploaded image"""
        image = self.cleaned_data.get('image')
        if image:
            # Check file size (max 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Image file too large. Maximum size is 10MB.")
            
            # Check image format
            if not image.content_type.startswith('image/'):
                raise forms.ValidationError("Please upload a valid image file.")
        return image
    
    def clean_video(self):
        """Validate uploaded video"""
        video = self.cleaned_data.get('video')
        if video:
            # Check file size (max 50MB)
            if video.size > 50 * 1024 * 1024:
                raise forms.ValidationError("Video file too large. Maximum size is 50MB.")
            
            # Check video format
            allowed_types = ['video/mp4', 'video/quicktime', 'video/webm']
            if video.content_type not in allowed_types:
                raise forms.ValidationError("Please upload MP4, MOV, or WebM video files only.")
        return video
    
    def clean_title(self):
        """Validate and clean title"""
        title = self.cleaned_data.get('title')
        if title:
            # Remove extra whitespace
            title = ' '.join(title.split())
            
            # Check length
            if len(title) < 3:
                raise forms.ValidationError("Title must be at least 3 characters long.")
        return title


class UploadForm(forms.ModelForm):
    """Legacy upload form"""
    
    class Meta:
        model = Upload
        fields = ['target_name', 'image', 'video']
        
        widgets = {
            'target_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter target name'
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*'
            }),
            'video': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*'
            })
        }
