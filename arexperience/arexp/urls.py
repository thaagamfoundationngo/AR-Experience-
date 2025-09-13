from django.urls import path
from . import views


urlpatterns = [
    # ============================================================================
    # CORE PAGES
    # ============================================================================
    path('', views.home, name='home'),
    path('scanner/', views.scanner, name='scanner'),
    path('upload/', views.upload_view, name='upload_view'),
    
    # ============================================================================
    # AR EXPERIENCE VIEWERS
    # ============================================================================
    # Main AR Experience viewer (slug-based - RECOMMENDED)
    path('x/<slug:slug>/', views.experience_view, name='experience_view'),
    
    # Webcam AR experience
    path('webcam/', views.webcam_ar_experience_view, name='webcam_ar_experience'),
    path('webcam/<slug:slug>/', views.webcam_ar_experience_view, name='webcam_ar_experience_slug'),
    
    # Alternative AR viewers for compatibility
    path('ar/<slug:slug>/', views.ar_experience_by_slug, name='ar_experience_slug'),
    path('ar/<int:experience_id>/', views.ar_experience_view, name='ar_experience'),
    
    # ============================================================================
    # BROWSER-BASED MINDAR COMPILER
    # ============================================================================
    path('browser-mindar-compiler/', views.browser_mindar_compiler, name='browser_mindar_compiler'),
    path('save-browser-mindar-target/', views.save_browser_mindar_target, name='save_browser_mindar_target'),
    
    # ============================================================================
    # API ENDPOINTS
    # ============================================================================
    path('api/ar-status/<slug:slug>/', views.ar_status_api, name='ar_status_api'),
    path('api/marker-status/<slug:slug>/', views.marker_status_api, name='marker_status_api'),  # FIXED
    
    # Backend tracking validation APIs
    path('api/validate-tracking/<slug:slug>/', views.validate_tracking_api, name='validate_tracking_api'),
    path('api/start-realtime-tracking/<slug:slug>/', views.start_realtime_tracking_api, name='start_realtime_tracking_api'),
    path('api/stop-realtime-tracking/<slug:slug>/', views.stop_realtime_tracking_api, name='stop_realtime_tracking_api'),
    
    # ============================================================================
    # UTILITY ENDPOINTS
    # ============================================================================
    path('qr/<slug:slug>/', views.qr_view, name='qr_view'),
    
    # ============================================================================
    # DEBUG ENDPOINTS (Development only)
    # ============================================================================
    path('debug/markers/<slug:slug>/', views.debug_markers, name='debug_markers'),
    
]
