from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from pathlib import Path
from django.contrib.staticfiles import finders
import os
from django.core.management import execute_from_command_line
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arexperience.settings')
import os
from django.core.management import execute_from_command_line
import sys

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arexperience.settings')



def debug_markers(request, slug):
    """Debug view to check marker file status"""
    # Fetch the experience by its slug
    experience = get_object_or_404(ARExperience, slug=slug)
    
    # Path to the markers folder
    marker_dir = finders.find(f'markers/{slug}')
    
    # Prepare the response with file status info
    debug_info = {
        'slug': slug,
        'marker_dir': marker_dir,
        'marker_dir_exists': os.path.exists(marker_dir) if marker_dir else False,
        'files': {}
    }
    
    # List of required files: .iset, .fset, .fset3
    required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]
    
    if marker_dir:
        for filename in required_files:
            filepath = os.path.join(marker_dir, filename)
            debug_info['files'][filename] = {
                'exists': os.path.exists(filepath),
                'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
            }
            
            # Optionally, read the first few bytes of the file for debugging
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        debug_info['files'][filename]['content_preview'] = f.read(200)
                except:
                    debug_info['files'][filename]['content_preview'] = 'Binary file or read error'
    
    # Return a JSON response with the debug information
    return JsonResponse(debug_info, indent=2)



# Now run the server or the Django management command
if __name__ == "__main__":
    # Specify the port (default to 8000 if not provided)
    sys.argv.append('runserver')
    sys.argv.append('127.0.0.1:8000')  # Default address and port
    execute_from_command_line(sys.argv)
