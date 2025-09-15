# arexperience/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('arexp.urls')),
]

# ✅ Serve media files in development
if settings.DEBUG:
    # serve uploaded media in dev
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # optional: serve collected static (only needed if you also want STATIC_ROOT in dev)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)