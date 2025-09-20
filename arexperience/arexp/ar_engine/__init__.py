# ar_engine/__init__.py
"""
Python AR Engine for Video Overlay Platform

This package provides a complete AR tracking and video overlay system
using OpenCV and PyTorch for marker-based augmented reality experiences.
"""

from .video_overlay import PythonVideoAR
from .marker_tracker import ARMarkerTracker
from .ar_processor import ARProcessor
from .optimizers import AROptimizer

# Optional ML tracker (requires PyTorch)
try:
    from .ml_tracker import PythonARTracker
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Python AR Platform"

# Default AR engine instance
default_ar_engine = PythonVideoAR()

# Export main classes
__all__ = [
    'PythonVideoAR',
    'ARMarkerTracker', 
    'ARProcessor',
    'AROptimizer',
    'default_ar_engine',
    'ML_AVAILABLE'
]

if ML_AVAILABLE:
    __all__.append('PythonARTracker')
