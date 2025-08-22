# arexp/urls.py
from django.urls import path
from . import views
from .views import home, scanner, upload_view, experience_view, qr_view
from .views import home, upload_view, experience_view, qr_view, scanner

urlpatterns = [
    path('', views.home, name='home'),
    path('scanner/', views.scanner, name='scanner'),
    path('upload/', views.upload_view, name='upload_view'), # Make sure this matches
    path('x/<slug:slug>/', views.experience_view, name='experience_view'),
    path('qr/<slug:slug>/', views.qr_view, name='qr_view'),
    
    path("qr/<slug:slug>/", qr_view, name="qr"),
    
    # Debug routes
    #path("debug/<slug:slug>/", debug_markers, name="debug_markers"),
    #path("regenerate/<slug:slug>/", regenerate_markers, name="regenerate_markers"),
]