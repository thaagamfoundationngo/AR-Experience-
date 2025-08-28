# arexp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Home and basic pages
    path('', views.home, name='home'),
    path('scanner/', views.scanner, name='scanner'),
    path('upload/', views.upload_view, name='upload_view'),
    
    # Main AR Experience viewer (slug-based - RECOMMENDED)
    path('x/<slug:slug>/', views.experience_view, name='experience_view'),
    
    # Alternative AR viewers for compatibility
    path('ar/<slug:slug>/', views.ar_experience_by_slug, name='ar_experience_slug'),
    path('ar/<int:experience_id>/', views.ar_experience_view, name='ar_experience'),
    
    # QR code generator
    path('qr/<slug:slug>/', views.qr_view, name='qr_view'),
    
    # Debug and utility endpoints
    path('debug_markers/<slug:slug>/', views.debug_markers, name='debug_markers'),
    path('regenerate_markers/<slug:slug>/', views.regenerate_markers, name='regenerate_markers'),
    path('marker_status/<slug:slug>/', views.marker_status_api, name='marker_status_api'),
]