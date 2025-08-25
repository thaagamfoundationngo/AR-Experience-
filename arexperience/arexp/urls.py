# arexp/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('scanner/', views.scanner, name='scanner'),
    path('upload/', views.upload_view, name='upload_view'),
    
    # AR Experience viewer (this is what you need for video overlay)
    path('ar/<int:experience_id>/', views.ar_experience_view, name='ar_experience'),
    
    # Experience pages
    path('x/<slug:slug>/', views.experience_view, name='experience_view'),
    
    # QR code views (consolidated - removed duplicate)
    path('qr/<slug:slug>/', views.qr_view, name='qr_view'),
    
    # Optional: Add slug-based AR access too
    path('ar/<slug:slug>/', views.ar_experience_by_slug, name='ar_experience_slug'),
    path('debug_markers/<slug>/', views.debug_markers, name='debug_markers'),
]