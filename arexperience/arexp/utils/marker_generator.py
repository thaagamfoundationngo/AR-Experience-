import cv2
import numpy as np
from pathlib import Path

def generate_marker(output_path: str, size: int = 400):
    """
    Generate a simple AR marker image with a black border and a basic pattern,
    then save the image and create a corresponding .patt file.

    Args:
        output_path (str): Path where to save the marker PNG image.
        size (int): Size (width and height) of the marker image in pixels.

    Returns:
        tuple(str, str): Paths to the saved PNG image and generated .patt file.
    """
    # Initialize white square image
    marker = np.ones((size, size), dtype=np.uint8) * 255

    # Draw solid black border around the entire marker
    border_size = size // 10
    marker[:border_size, :] = 0
    marker[-border_size:, :] = 0
    marker[:, :border_size] = 0
    marker[:, -border_size:] = 0

    # Draw a simple inner black square pattern
    pattern_size = (size - 2 * border_size) // 3
    start = border_size + pattern_size
    end = start + pattern_size
    marker[start:end, start:end] = 0

    # Save marker image as PNG
    img_path = Path(output_path)
    cv2.imwrite(str(img_path), marker)

    # Create .patt file for AR.js marker detection
    pattern_path = img_path.with_suffix('.patt')
    with open(pattern_path, 'w') as f:
        f.write(f"16\n")  # AR.js expects 16x16 pattern size

        # Resize marker to 16x16 to generate pattern matrix
        small_marker = cv2.resize(marker, (16, 16), interpolation=cv2.INTER_AREA)
        for i in range(16):
            for j in range(16):
                # Normalize pixel to 0.0-1.0 and write pattern value
                val = small_marker[i, j] / 255.0
                f.write(f"{val:.2f} ")
            f.write("\n")

    return str(img_path), str(pattern_path)
