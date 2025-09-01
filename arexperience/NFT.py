# NFT.py - Standalone NFT marker file checker
import os
from pathlib import Path

def check_nft_files(slug: str, media_root: str = None):
    """
    Check if NFT marker files exist for a given AR experience slug
    """
    # Auto-detect media root if not provided
    if media_root is None:
        # Assume media folder is in parent directory
        current_dir = Path(__file__).parent
        media_root = current_dir.parent / "media"
    
    marker_dir = Path(media_root) / "markers" / slug
    required_files = [f"{slug}.iset", f"{slug}.fset", f"{slug}.fset3"]
    
    print(f"🔍 Checking NFT files for: {slug}")
    print(f"📁 Marker directory: {marker_dir}")
    print(f"📁 Directory exists: {'✅' if marker_dir.exists() else '❌'}")
    print()
    
    all_exist = True
    for file in required_files:
        file_path = marker_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"{file}: ✅ {size:,} bytes")
        else:
            print(f"{file}: ❌ File not found")
            all_exist = False
    
    print()
    print(f"🎯 NFT Tracking Ready: {'✅ YES' if all_exist else '❌ NO'}")
    
    # List all files in marker directory
    if marker_dir.exists():
        print(f"\n📄 All files in {marker_dir}:")
        for file in marker_dir.glob("*"):
            size = file.stat().st_size if file.is_file() else 0
            print(f"  - {file.name} ({size:,} bytes)")
    
    return all_exist

def check_all_experiences(media_root: str = None):
    """
    Check NFT files for all AR experiences
    """
    if media_root is None:
        current_dir = Path(__file__).parent
        media_root = current_dir.parent / "media"
    
    markers_dir = Path(media_root) / "markers"
    
    if not markers_dir.exists():
        print(f"❌ Markers directory not found: {markers_dir}")
        return
    
    print(f"🔍 Checking all AR experiences in: {markers_dir}")
    print("=" * 50)
    
    for experience_dir in markers_dir.iterdir():
        if experience_dir.is_dir():
            check_nft_files(experience_dir.name, media_root)
            print("-" * 30)

if __name__ == "__main__":
    # Check specific experience
    print("🎯 AR.js NFT Marker File Checker")
    print("=" * 40)
    
    # Check for 'mom' experience
    check_nft_files('mom')
    
    print("\n" + "=" * 40)
    
    # Check all experiences
    check_all_experiences()
