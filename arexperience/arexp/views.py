# arexp/views.py
from django.shortcuts import render, redirect, get_object_or_404
from .models import Upload

def home(request):
    if request.method == "POST":
        img = request.FILES.get("image")
        vid = request.FILES.get("video")
        if img and vid:
            u = Upload.objects.create(image=img, video=vid)  # slug auto-generates
            # After upload, go straight to the scanner for this item
            return redirect("scanner", slug=u.slug)
    uploads = Upload.objects.order_by("-uploaded_at")[:5]
    return render(request, "home.html", {"uploads": uploads})

def scanner(request, slug):
    u = get_object_or_404(Upload, slug=slug)
    if not u.mind_file:
        # Mind target not generated yet (or failed); show a simple message page
        return render(request, "scanner_empty.html", {"slug": slug})
    ctx = {
        "title": f"AR Scan: {slug}",
        "mind_target_url": u.mind_file.url,  # served via MEDIA_URL
        "video_url": u.video.url,            # uploaded video
        "plane_w": 1.0,
        "plane_h": 0.56,                     # adjust to your image's aspect ratio
    }
    return render(request, "scanner.html", ctx)
