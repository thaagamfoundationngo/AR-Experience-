# verify_media_config.py
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arexperience.settings')
django.setup()

from django.conf import settings
from pathlib import Path

print(f"✅ MEDIA_URL: {settings.MEDIA_URL}")
print(f"✅ MEDIA_ROOT: {settings.MEDIA_ROOT}")
print(f"✅ MEDIA_ROOT exists: {Path(settings.MEDIA_ROOT).exists()}")

# Create directory if it doesn't exist
Path(settings.MEDIA_ROOT).mkdir(exist_ok=True)
markers_dir = Path(settings.MEDIA_ROOT) / "markers"
markers_dir.mkdir(exist_ok=True)

print(f"✅ Markers directory: {markers_dir}")
print(f"✅ Ready for NFT generation!")
