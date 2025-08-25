# arexp/utils/arjs_marker.py
import os
import shutil
import subprocess
from pathlib import Path
from django.conf import settings


def ensure_named(file_path: str, expected_name: str) -> str:
    """
    Ensures a file has the expected name. If not, renames it.
    Returns the path to the correctly named file.
    """
    file_path = Path(file_path)
    expected_path = file_path.parent / expected_name
    
    if file_path.name != expected_name:
        try:
            shutil.move(str(file_path), str(expected_path))
            print(f"[arjs] Renamed {file_path.name} to {expected_name}")
            return str(expected_path)
        except Exception as e:
            print(f"[arjs] Failed to rename {file_path.name}: {e}")
            return str(file_path)
    
    return str(file_path)

def train_arjs_marker(image_path: str, out_dir: str, slug: str) -> bool:
    """
    Use the Node NFT-Marker-Creator app to generate .iset/.fset/.fset3.
    Writes/renames them to out_dir/<slug>.* and returns True on success.
    """
    img = Path(image_path).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Location of the script inside node_modules
    pkg_root = Path(settings.BASE_DIR) / "node_modules" / "@webarkit" / "nft-marker-creator-app"
    script = pkg_root / "src" / "NFTMarkerCreator.js"
    if not script.exists():
        print("[arjs] NFTMarkerCreator.js not found at:", script)
        return False

    # Windows-safe node executable
    node = "node.exe" if os.name == "nt" else "node"

    # Run in the package root; it creates an 'output' folder there
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    # Copy image to package root to avoid path issues
    temp_img = pkg_root / f"temp_{slug}_{img.name}"
    try:
        shutil.copy2(img, temp_img)
        print(f"[arjs] Copied image to: {temp_img}")
    except Exception as e:
        print(f"[arjs] Failed to copy image: {e}")
        return False
    
    cmd = [node, str(script), "-i", temp_img.name]  # Just the filename
    print("[arjs] Running:", " ".join(cmd))
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(pkg_root),         # important: outputs go to pkg_root / output
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=60,               # Add timeout to prevent hanging
        )
        
        # Log the actual output/error for debugging
        print(f"[arjs] stdout: {proc.stdout[:800] if proc.stdout else 'None'}")
        print(f"[arjs] stderr: {proc.stderr[:800] if proc.stderr else 'None'}")
        print(f"[arjs] return code: {proc.returncode}")
        
        # Check if the process succeeded
        if proc.returncode != 0:
            print(f"[arjs] Node process failed with code {proc.returncode}")
            return False
            
    except subprocess.TimeoutExpired as e:
        print(f"[arjs] Timeout after 60 seconds: {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[arjs] Process error: {e}")
        print(f"[arjs] stdout: {e.stdout[:800] if e.stdout else 'None'}")
        print(f"[arjs] stderr: {e.stderr[:800] if e.stderr else 'None'}")
        return False
    except Exception as e:
        print(f"[arjs] Exception in train_arjs_marker: {e}")
        return False

    # The generator writes into <pkg_root>/output
    gen_dir = pkg_root / "output"
    if not gen_dir.exists():
        print("[arjs] output folder not found:", gen_dir)
        return False

    # Find any *.iset/*.fset/*.fset3 created (there may be generic names)
    produced = {
        ".iset": None,
        ".fset": None,
        ".fset3": None,
    }
    for p in gen_dir.glob("*"):
        if p.suffix in produced and produced[p.suffix] is None:
            produced[p.suffix] = p

    ok = True
    for ext in [".iset", ".fset", ".fset3"]:
        src = produced.get(ext)
        if not src or not src.exists():
            print(f"[arjs] missing generated {ext} file")
            ok = False
            continue
        dest = out / f"{slug}{ext}"
        try:
            shutil.copy2(src, dest)
            print(f"[arjs] copied {src} -> {dest}")
        except Exception as e:
            print(f"[arjs] copy error: {src} -> {dest}, {repr(e)}")
            ok = False

    # Clean up temporary image
    if temp_img.exists():
        try:
            temp_img.unlink()
            print(f"[arjs] Cleaned up temp image: {temp_img}")
        except Exception as e:
            print(f"[arjs] Failed to clean up temp image: {e}")

    return ok


def build_pattern_marker(image_path: str, slug: str, media_root: str, marker_size_m=1.0):
    """
    Generate a pattern marker file (.patt) from an image.
    Uses a custom implementation that doesn't rely on external npm packages.
    """
    try:
        img = Path(image_path)
        out_dir = Path(media_root) / "markers"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if image exists
        if not img.exists():
            print(f"[marker] Image file not found: {img}")
            return None
            
        # Define output path for the pattern file
        pattern_file = out_dir / f"{slug}.patt"
        
        # Read the image using OpenCV
        image = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[marker] Failed to read image: {img}")
            return None
        
        # Resize to a fixed size (16x16) for pattern
        pattern_size = 16
        resized_image = cv2.resize(image, (pattern_size, pattern_size))
        
        # Normalize pixel values to 0-1
        normalized_image = resized_image / 255.0
        
        # Write the pattern file in AR.js format
        with open(pattern_file, 'w') as f:
            # Write the pattern size
            f.write(f"{pattern_size}\n")
            # Write the normalized pixel values
            for i in range(pattern_size):
                for j in range(pattern_size):
                    f.write(f"{normalized_image[i, j]:.2f} ")
                f.write("\n")
        
        print(f"[marker] Pattern file created: {pattern_file}")
        return str(pattern_file)
        
    except Exception as e:
        print(f"[marker] Error in build_pattern_marker: {str(e)}")
        return None    
    
def create_visual_marker(slug: str, media_root: str) -> str:
    """
    Create a visual marker image that corresponds to the pattern.
    Returns path to the created image.
    """
    try:
        # Create markers directory if it doesn't exist
        markers_dir = Path(media_root) / "markers"
        markers_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output path
        output_path = markers_dir / f"{slug}.png"
        
        # Create a simple geometric pattern
        size = 400
        marker = np.ones((size, size), dtype=np.uint8) * 255
        
        # Draw border
        border_size = size // 10
        marker[:border_size, :] = 0
        marker[-border_size:, :] = 0
        marker[:, :border_size] = 0
        marker[:, -border_size:] = 0
        
        # Draw inner pattern based on slug
        pattern_size = size - 2 * border_size
        start = border_size
        
        # Create a simple pattern based on the slug
        for i, char in enumerate(slug[:9]):  # Use first 9 characters
            row = i // 3
            col = i % 3
            if char != ' ' and char != '_':  # Draw block for non-space characters
                block_size = pattern_size // 3
                y_start = start + row * block_size
                y_end = y_start + block_size
                x_start = start + col * block_size
                x_end = x_start + block_size
                marker[y_start:y_end, x_start:x_end] = 0
        
        # Save the marker
        cv2.imwrite(str(output_path), marker)
        
        print(f"[marker] Visual marker created: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"[marker] Error creating visual marker: {str(e)}")
        return None
    
        
def generate_simple_marker(slug: str, media_root: str) -> str:
    """Generate a simple fallback marker when AR.js fails"""
    import cv2
    import numpy as np
    from pathlib import Path
    
    # Create markers directory if it doesn't exist
    markers_dir = Path(media_root) / "markers"
    markers_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output path
    output_path = markers_dir / f"{slug}.png"
    
    # Create a simple geometric pattern
    size = 400
    marker = np.ones((size, size), dtype=np.uint8) * 255
    
    # Draw border
    border_size = size // 10
    marker[:border_size, :] = 0
    marker[-border_size:, :] = 0
    marker[:, :border_size] = 0
    marker[:, -border_size:] = 0
    
    # Draw inner pattern
    pattern_size = size - 2 * border_size
    start = border_size
    end = start + pattern_size
    
    # Create a simple pattern based on the slug
    for i, char in enumerate(slug[:9]):  # Use first 9 characters
        row = i // 3
        col = i % 3
        if char != ' ' and char != '_':  # Draw block for non-space characters
            block_size = pattern_size // 3
            y_start = start + row * block_size
            y_end = y_start + block_size
            x_start = start + col * block_size
            x_end = x_start + block_size
            marker[y_start:y_end, x_start:x_end] = 0
    
    # Save the marker
    cv2.imwrite(str(output_path), marker)
    
    return str(output_path)