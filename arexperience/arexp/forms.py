# forms.py
from django import forms
from .models import ARExperience

from django import forms
from .models import ARExperience

class ARExperienceForm(forms.ModelForm):
    """Form for creating and editing AR experiences"""
    
    class Meta:
        model = ARExperience
        fields = ['title', 'description', 'image', 'video', 'model_file', 'content_text', 'content_url', 'marker_size']  # Add 'video'
        
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter experience title...'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Describe your AR experience...'
            }),
            'image': forms.FileInput(attrs={
                'class': 'form-control-file',
                'accept': 'image/*'
            }),
            'video': forms.FileInput(attrs={  # Add video widget
                'class': 'form-control-file',
                'accept': 'video/mp4,video/webm,video/quicktime'
            }),
            'model_file': forms.FileInput(attrs={
                'class': 'form-control-file',
                'accept': '.glb,.gltf,.obj'
            }),
            'content_text': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 2,
                'placeholder': 'Additional text content...'
            }),
            'content_url': forms.URLInput(attrs={
                'class': 'form-control',
                'placeholder': 'https://...'
            }),
            'marker_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.1',
                'max': '10',
                'step': '0.1'
            })
        }
            
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Make certain fields required
        self.fields['title'].required = True
        self.fields['image'].required = True
        
        # Add help text
        self.fields['image'].help_text = "Upload a high-contrast image that works well as an AR marker"
        self.fields['model_file'].help_text = "Optional: Upload a 3D model to display in AR"
        self.fields['marker_size'].help_text = "Size of the AR content (1.0 = normal size)"
    
    def clean_image(self):
        """Validate uploaded image"""
        image = self.cleaned_data.get('image')
        
        if image:
            # Check file size (limit to 10MB)
            if image.size > 10 * 1024 * 1024:
                raise forms.ValidationError("Image file too large. Maximum size is 10MB.")
            
            # Check file type
            allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
            if hasattr(image, 'content_type') and image.content_type not in allowed_types:
                raise forms.ValidationError("Only JPEG and PNG images are allowed.")
        
        return image
    
    def clean_model_file(self):
        """Validate uploaded 3D model"""
        model_file = self.cleaned_data.get('model_file')
        
        if model_file:
            # Check file size (limit to 50MB)
            if model_file.size > 50 * 1024 * 1024:
                raise forms.ValidationError("Model file too large. Maximum size is 50MB.")
            
            # Check file extension
            allowed_extensions = ['.glb', '.gltf', '.obj']
            file_extension = model_file.name.lower().split('.')[-1] if '.' in model_file.name else ''
            
            if f'.{file_extension}' not in allowed_extensions:
                raise forms.ValidationError("Only GLB, GLTF, and OBJ files are allowed.")
        
        return model_file