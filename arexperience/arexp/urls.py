# arexp/urls.py - MINIMAL WORKING VERSION
from django.urls import path
from . import views

urlpatterns = [
    # Core pages that exist
    path('', views.home, name='home'),
    path('upload/', views.upload_view, name='upload_view'),
    
    # Python AR Experience
    path('x/<slug:slug>/', views.experience_view, name='experience_view'),
    
    # Python AR Live Stream
    path('stream/<slug:slug>/', views.ar_camera_stream, name='ar_camera_stream'),
    
    # API for AR status
    path('api/ar-status/<slug:slug>/', views.python_ar_status_api, name='python_ar_status_api'),
    
    # Legacy compatibility (if exists)
    path('fetch-mind/<str:slug>/', views.fetch_mind_marker, name='fetch_mind_marker'),
    path('api/process-ar-frame/<slug:slug>/', views.process_ar_frame_api, name='process_ar_frame_api'),

]
