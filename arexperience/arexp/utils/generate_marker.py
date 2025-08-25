#nenerate_maerker.py
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_text_marker(text, output_path, size=400):
    """Create a marker with text"""
    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font
    try:
        font = ImageFont.truetype("arial.ttf", int(size/3))
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width, text_height = draw.textsize(text, font=font)
    position = ((size - text_width) // 2, (size - text_height) // 2)
    
    # Draw the text
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    # Save the image
    img.save(output_path)
    
    # Convert to OpenCV format for pattern generation
    img_cv = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    
    # Generate pattern file
    pattern_path = output_path.replace('.png', '.patt')
    with open(pattern_path, 'w') as f:
        f.write("16\n")  # Pattern size
        # Resize to 16x16 for pattern
        small_img = cv2.resize(img_cv, (16, 16))
        for i in range(16):
            for j in range(16):
                # Normalize to 0-1
                val = small_img[i, j] / 255.0
                f.write(f"{val:.2f} ")
            f.write("\n")
    
    return output_path, pattern_path
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_text_marker(text, output_path, size=400):
    """Create a marker with text"""
    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font
    try:
        font = ImageFont.truetype("arial.ttf", int(size/3))
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width, text_height = draw.textsize(text, font=font)
    position = ((size - text_width) // 2, (size - text_height) // 2)
    
    # Draw the text
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    # Save the image
    img.save(output_path)
    
    # Convert to OpenCV format for pattern generation
    img_cv = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    
    # Generate pattern fileimport os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_text_marker(text, output_path, size=400):
    """Create a marker with text"""
    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font
    try:
        font = ImageFont.truetype("arial.ttf", int(size/3))
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width, text_height = draw.textsize(text, font=font)
    position = ((size - text_width) // 2, (size - text_height) // 2)
    
    # Draw the text
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    # Save the image
    img.save(output_path)
    
    # Convert to OpenCV format for pattern generation
    img_cv = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    
    # Generate pattern file
    pattern_path = output_path.replace('.png', '.patt')
    with open(pattern_path, 'w') as f:
        f.write("16\n")  # Pattern size
        # Resize to 16x16 for pattern
        small_img = cv2.resize(img_cv, (16, 16))
        for i in range(16):
            for j in range(16):
                # Normalize to 0-1
                val = small_img[i, j] / 255.0
                f.write(f"{val:.2f} ")
            f.write("\n")
    
    return output_path, pattern_path

# Example usage
if __name__ == "__main__":
    create_text_marker("AR", "ar_marker.png")
    pattern_path = output_path.replace('.png', '.patt')
    with open(pattern_path, 'w') as f:
        f.write("16\n")  # Pattern size
        # Resize to 16x16 for pattern
        small_img = cv2.resize(img_cv, (16, 16))
        for i in range(16):
            for j in range(16):
                # Normalize to 0-1
                val = small_img[i, j] / 255.0
                f.write(f"{val:.2f} ")
            f.write("\n")
    return output_path, pattern_path

# Example usage
if __name__ == "__main__":
    create_text_marker("AR", "ar_marker.png")
# Example usage
if __name__ == "__main__":
    create_text_marker("AR", "ar_marker.png")