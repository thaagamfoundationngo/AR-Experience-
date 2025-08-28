# utils/ar_nft.py
import subprocess, shutil, os
from pathlib import Path

def generate_nft(image_path: str, out_dir: str):
    if not shutil.which("ar-nft"):
        raise RuntimeError("`ar-nft` CLI not found. Install with: npm i -g @ar-js-org/ar-nft")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ar-nft", "generate",
        "-i", image_path,
        "-o", out_dir,
        "-m", "3"
    ]
    subprocess.run(cmd, check=True)
    # Sanity check
    expected = ["fset", "fset3", "iset"]
    base = Path(out_dir).name
    for ext in expected:
        if not Path(out_dir, f"{base}.{ext}").exists():
            raise RuntimeError(f"Missing {ext} in {out_dir}")
    return out_dir
