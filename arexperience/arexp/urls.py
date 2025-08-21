from django.urls import path
from .views import home, scanner

urlpatterns = [
    path("", home, name="home"),
    path("x/<slug:slug>/", scanner, name="scanner"),
]
