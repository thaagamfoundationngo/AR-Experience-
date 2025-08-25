# arexp/apps.py
from django.apps import AppConfig

class ArexpConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "arexp"

    def ready(self):
        from . import signals  # register signal handlers

