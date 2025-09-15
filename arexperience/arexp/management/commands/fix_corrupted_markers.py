# In: your_app/management/commands/fix_corrupted_markers.py
from django.core.management.base import BaseCommand
from your_app.models import ARExperience
from your_app.views import generate_mindar_marker
import os
import struct
from pathlib import Path
from django.conf import settings

class Command(BaseCommand):
    help = 'Automatically detect and fix corrupted MindAR markers'
    
    def handle(self, *args, **options):
        self.stdout.write("🔍 Scanning for corrupted markers...")
        
        fixed_count = 0
        total_count = 0
        
        for experience in ARExperience.objects.filter(image__isnull=False):
            total_count += 1
            marker_path = Path(settings.MEDIA_ROOT) / "markers" / experience.slug / f"{experience.slug}.mind"
            
            if not marker_path.exists():
                self.stdout.write(f"⚠️ Missing marker for {experience.slug}")
                if self.regenerate_marker(experience, marker_path):
                    fixed_count += 1
                continue
            
            # Check for corruption
            if self.is_corrupted(marker_path):
                self.stdout.write(f"🔧 Fixing corrupted marker: {experience.slug}")
                if self.regenerate_marker(experience, marker_path):
                    fixed_count += 1
            else:
                self.stdout.write(f"✅ {experience.slug} - OK")
        
        self.stdout.write(
            self.style.SUCCESS(f'Fixed {fixed_count}/{total_count} markers')
        )
    
    def is_corrupted(self, file_path):
        """Check if marker file is corrupted"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'MIND':
                    return True
                
                size_bytes = f.read(4)
                json_size = struct.unpack('<I', size_bytes)[0]
                
                json_data = f.read(json_size)
                if len(json_data) != json_size:
                    return True
                
                # Check for extra bytes (corruption signature)
                extra = f.read()
                if extra:
                    return True
                
                return False
        except:
            return True
    
    def regenerate_marker(self, experience, marker_path):
        """Regenerate corrupted marker"""
        try:
            # Delete corrupted file
            if marker_path.exists():
                marker_path.unlink()
            
            # Regenerate
            success = generate_mindar_marker(experience.image.path, str(marker_path))
            if success:
                experience.marker_generated = True
                experience.save()
                return True
            return False
        except Exception as e:
            self.stdout.write(f"❌ Failed to regenerate {experience.slug}: {str(e)}")
            return False
