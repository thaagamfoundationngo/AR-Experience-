# Create a file like utils/marker_generator.py
import os
import cv2
import numpy as np
from pathlib import Path

def generate_marker(output_path, marker_id=0, size=400):
    """Generate an AR marker and save it as an image and .patt file"""
    # Create a simple marker pattern
    marker = np.ones((size, size), dtype=np.uint8) * 255
    
    # Draw a simple pattern (you can customize this)
    border_size = size // 10
    inner_size = size - 2 * border_size
    
    # Create a border
    marker[:border_size, :] = 0
    marker[-border_size:, :] = 0
    marker[:, :border_size] = 0
    marker[:, -border_size:] = 0
    
    # Draw a simple pattern in the middle
    pattern_size = inner_size // 3
    start = border_size + pattern_size
    end = start + pattern_size
    marker[start:end, start:end] = 0
    
    # Save the marker image
    cv2.imwrite(output_path, marker)
    
    # Generate .patt file (simplified version)
    pat_path = output_path.replace('.png', '.patt')
    with open(pat_path, 'w') as f:
        # Write a simple pattern file (this is a basic example)
        f.write(f"# Marker {marker_id}\n")
        f.write(f"{size}\n")
        # In a real implementation, you'd extract the pattern data here
        # This is just a placeholder
        for i in range(size):
            for j in range(size):
                f.write(f"{'1' if marker[i,j] < 128 else '0'} ")
            f.write("\n")
    
    return output_path, pat_path






from utils.marker_generator import generate_marker

def build_pattern_marker_fallback(image_path: str, slug: str, media_root: str):
    try:
        out_dir = Path(media_root) / "markers"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{slug}.png"
        
        # Generate a simple marker
        return generate_marker(str(output_path), marker_id=hash(slug) % 1000)
    except Exception as e:
        print(f"[marker] Fallback also failed: {str(e)}")
        return None