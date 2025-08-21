# arexp/signals.py
import subprocess, shutil
from pathlib import Path
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Upload

def _resolve_mindar_cli() -> str:
    # 1) Use explicit setting if provided
    cli = getattr(settings, "MINDAR_CLI", None)
    if cli:
        return cli
    # 2) Try PATH
    found = shutil.which("mindar-image")
    if found:
        return found
    # 3) Try common Windows global npm bin
    candidate = Path.home() / "AppData" / "Roaming" / "npm" / "mindar-image.cmd"
    return str(candidate)

@receiver(post_save, sender=Upload)
def generate_mind_after_upload(sender, instance: Upload, created, **kwargs):
    if not created or not instance.image:
        return

    media_root = Path(settings.MEDIA_ROOT)
    input_path = media_root / instance.image.name

    targets_dir = media_root / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    output_path = targets_dir / f"{instance.slug}.mind"

    mindar_cli = _resolve_mindar_cli()

    # Use a list (no shell) when we have an explicit .cmd path; Windows can execute it directly.
    try:
        subprocess.run(
            [mindar_cli, "--input", str(input_path), "--output", str(output_path)],
            check=True
        )
    except Exception as e:
        print(f"[MindAR] Failed to generate target for {instance.slug}: {e}")
        return

    instance.mind_file.name = f"targets/{instance.slug}.mind"
    instance.save(update_fields=["mind_file"])
