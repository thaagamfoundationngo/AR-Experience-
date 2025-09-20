# ar_engine/optimizers.py
"""
Performance optimizers for Python AR platform
Provides GPU acceleration, memory management, and real-time optimizations
"""

import numpy as np
import cv2
import threading
import queue
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from multiprocessing import Pool, cpu_count
import psutil

try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AROptimizer:
    """Main AR performance optimizer"""
    
    def __init__(self):
        self.cpu_count = cpu_count()
        self.memory_info = psutil.virtual_memory()
        self.gpu_available = CUPY_AVAILABLE or (NUMBA_AVAILABLE and cuda.is_available())
        
        # Performance tracking
        self.performance_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'frame_times': [],
            'optimization_enabled': True
        }
        
        logger.info(f"AR Optimizer initialized:")
        logger.info(f"  - CPU cores: {self.cpu_count}")
        logger.info(f"  - Available RAM: {self.memory_info.total // (1024**3)} GB")
        logger.info(f"  - GPU acceleration: {self.gpu_available}")
        logger.info(f"  - Numba available: {NUMBA_AVAILABLE}")
        logger.info(f"  - CuPy available: {CUPY_AVAILABLE}")
    
    def optimize_image_processing(self, use_gpu: bool = True) -> Dict[str, Callable]:
        """Get optimized image processing functions"""
        
        if use_gpu and CUPY_AVAILABLE:
            return self._get_gpu_functions()
        elif NUMBA_AVAILABLE:
            return self._get_numba_functions()
        else:
            return self._get_standard_functions()
    
    def _get_gpu_functions(self) -> Dict[str, Callable]:
        """GPU-accelerated functions using CuPy"""
        
        def gpu_resize(image: np.ndarray, size: tuple) -> np.ndarray:
            """GPU-accelerated image resize"""
            gpu_img = cp.asarray(image)
            # CuPy doesn't have direct resize, use OpenCV on CPU for now
            return cv2.resize(cp.asnumpy(gpu_img), size)
        
        def gpu_grayscale(image: np.ndarray) -> np.ndarray:
            """GPU-accelerated grayscale conversion"""
            gpu_img = cp.asarray(image)
            # Weighted average for grayscale
            weights = cp.array([0.114, 0.587, 0.299])  # BGR weights
            gray_gpu = cp.dot(gpu_img, weights)
            return cp.asnumpy(gray_gpu).astype(np.uint8)
        
        def gpu_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
            """GPU-accelerated Gaussian blur"""
            # Use OpenCV for now, can be optimized with custom CUDA kernels
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return {
            'resize': gpu_resize,
            'grayscale': gpu_grayscale,
            'blur': gpu_gaussian_blur
        }
    
    def _get_numba_functions(self) -> Dict[str, Callable]:
        """Numba-accelerated functions"""
        
        @jit(nopython=True, parallel=True)
        def fast_feature_matching(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
            """Numba-optimized feature matching"""
            matches = []
            for i in numba.prange(len(desc1)):
                min_dist = np.inf
                best_match = -1
                for j in range(len(desc2)):
                    dist = np.sum((desc1[i] - desc2[j]) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = j
                if min_dist < 10000:  # Distance threshold
                    matches.append([i, best_match, min_dist])
            return np.array(matches)
        
        @jit(nopython=True)
        def fast_homography_check(H: np.ndarray) -> bool:
            """Fast homography validation"""
            if H is None:
                return False
            
            # Check determinant
            det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
            if abs(det) < 0.001:
                return False
            
            # Check if transformation is reasonable
            corners = np.array([[0, 0, 1], [100, 0, 1], [100, 100, 1], [0, 100, 1]]).T
            transformed = H @ corners
            
            # Check if points are still in reasonable positions
            for i in range(4):
                if transformed[2, i] == 0:
                    return False
                x = transformed[0, i] / transformed[2, i]
                y = transformed[1, i] / transformed[2, i]
                if abs(x) > 10000 or abs(y) > 10000:
                    return False
            
            return True
        
        return {
            'feature_matching': fast_feature_matching,
            'homography_check': fast_homography_check,
            'resize': cv2.resize,  # Standard function
            'grayscale': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            'blur': lambda img, k: cv2.GaussianBlur(img, (k, k), 0)
        }
    
    def _get_standard_functions(self) -> Dict[str, Callable]:
        """Standard OpenCV functions"""
        return {
            'resize': cv2.resize,
            'grayscale': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            'blur': lambda img, k: cv2.GaussianBlur(img, (k, k), 0),
            'feature_matching': self._standard_feature_matching,
            'homography_check': self._standard_homography_check
        }
    
    def _standard_feature_matching(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Standard feature matching"""
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        return [[m.queryIdx, m.trainIdx, m.distance] for m in matches]
    
    def _standard_homography_check(self, H: np.ndarray) -> bool:
        """Standard homography validation"""
        if H is None:
            return False
        return np.linalg.cond(H) < 1e12

class FrameBuffer:
    """Optimized frame buffer for real-time processing"""
    
    def __init__(self, maxsize: int = 10):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.dropped_frames = 0
        self.total_frames = 0
    
    def put_frame(self, frame: np.ndarray, metadata: Dict = None) -> bool:
        """Add frame to buffer, drop if full"""
        self.total_frames += 1
        
        try:
            frame_data = {
                'frame': frame,
                'metadata': metadata or {},
                'timestamp': time.time()
            }
            
            self.buffer.put_nowait(frame_data)
            return True
            
        except queue.Full:
            self.dropped_frames += 1
            # Drop oldest frame and add new one
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(frame_data)
                return True
            except queue.Empty:
                return False
    
    def get_frame(self, timeout: float = 0.1) -> Optional[Dict]:
        """Get frame from buffer"""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'dropped_frames': self.dropped_frames,
            'total_frames': self.total_frames,
            'drop_rate': self.dropped_frames / max(1, self.total_frames),
            'current_size': self.buffer.qsize()
        }

class MemoryOptimizer:
    """Memory management for AR processing"""
    
    def __init__(self):
        self.memory_pools = {}
        self.allocation_stats = {
            'total_allocated': 0,
            'peak_usage': 0,
            'current_usage': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
    
    def get_array(self, shape: tuple, dtype: np.dtype = np.uint8) -> np.ndarray:
        """Get reusable array from memory pool"""
        key = (shape, dtype)
        
        if key not in self.memory_pools:
            self.memory_pools[key] = []
        
        pool = self.memory_pools[key]
        
        if pool:
            self.allocation_stats['pool_hits'] += 1
            return pool.pop()
        else:
            self.allocation_stats['pool_misses'] += 1
            array_size = np.prod(shape) * np.dtype(dtype).itemsize
            self.allocation_stats['total_allocated'] += array_size
            self.allocation_stats['current_usage'] += array_size
            self.allocation_stats['peak_usage'] = max(
                self.allocation_stats['peak_usage'],
                self.allocation_stats['current_usage']
            )
            return np.empty(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray):
        """Return array to memory pool"""
        key = (array.shape, array.dtype)
        
        if key not in self.memory_pools:
            self.memory_pools[key] = []
        
        # Limit pool size to prevent memory bloat
        if len(self.memory_pools[key]) < 10:
            self.memory_pools[key].append(array)
    
    def cleanup(self, force: bool = False):
        """Clean up memory pools"""
        if force:
            self.memory_pools.clear()
            self.allocation_stats['current_usage'] = 0
        else:
            # Remove old pools if memory usage is high
            current_memory = psutil.virtual_memory()
            if current_memory.percent > 80:
                for key in list(self.memory_pools.keys()):
                    if len(self.memory_pools[key]) > 5:
                        self.memory_pools[key] = self.memory_pools[key][:5]
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return self.allocation_stats.copy()

class PerformanceMonitor:
    """Monitor system performance during AR processing"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'frame_times': [],
            'cpu_usage': [],
            'memory_usage': [],
            'detection_times': [],
            'overlay_times': []
        }
        
        self.start_time = time.time()
        self.frame_count = 0
    
    def record_frame_time(self, frame_time: float):
        """Record frame processing time"""
        self.metrics['frame_times'].append(frame_time)
        if len(self.metrics['frame_times']) > self.window_size:
            self.metrics['frame_times'].pop(0)
        
        self.frame_count += 1
    
    def record_detection_time(self, detection_time: float):
        """Record marker detection time"""
        self.metrics['detection_times'].append(detection_time)
        if len(self.metrics['detection_times']) > self.window_size:
            self.metrics['detection_times'].pop(0)
    
    def record_overlay_time(self, overlay_time: float):
        """Record video overlay time"""
        self.metrics['overlay_times'].append(overlay_time)
        if len(self.metrics['overlay_times']) > self.window_size:
            self.metrics['overlay_times'].pop(0)
    
    def record_system_stats(self):
        """Record current system statistics"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        self.metrics['cpu_usage'].append(cpu_percent)
        self.metrics['memory_usage'].append(memory.percent)
        
        if len(self.metrics['cpu_usage']) > self.window_size:
            self.metrics['cpu_usage'].pop(0)
        if len(self.metrics['memory_usage']) > self.window_size:
            self.metrics['memory_usage'].pop(0)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        runtime = time.time() - self.start_time
        
        # Calculate averages
        avg_frame_time = np.mean(self.metrics['frame_times']) if self.metrics['frame_times'] else 0
        avg_detection_time = np.mean(self.metrics['detection_times']) if self.metrics['detection_times'] else 0
        avg_overlay_time = np.mean(self.metrics['overlay_times']) if self.metrics['overlay_times'] else 0
        avg_cpu = np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
        avg_memory = np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        
        # Calculate FPS
        estimated_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        actual_fps = self.frame_count / runtime if runtime > 0 else 0
        
        return {
            'runtime_seconds': runtime,
            'total_frames': self.frame_count,
            'estimated_fps': estimated_fps,
            'actual_fps': actual_fps,
            'performance': {
                'avg_frame_time_ms': avg_frame_time * 1000,
                'avg_detection_time_ms': avg_detection_time * 1000,
                'avg_overlay_time_ms': avg_overlay_time * 1000,
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory
            },
            'bottlenecks': self._identify_bottlenecks(),
            'recommendations': self._get_recommendations()
        }
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        avg_frame_time = np.mean(self.metrics['frame_times']) if self.metrics['frame_times'] else 0
        avg_detection_time = np.mean(self.metrics['detection_times']) if self.metrics['detection_times'] else 0
        avg_overlay_time = np.mean(self.metrics['overlay_times']) if self.metrics['overlay_times'] else 0
        avg_cpu = np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
        avg_memory = np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        
        if avg_frame_time > 0.033:  # > 30 FPS
            bottlenecks.append("Frame processing too slow for real-time")
        
        if avg_detection_time > 0.020:  # > 20ms
            bottlenecks.append("Marker detection is slow")
        
        if avg_overlay_time > 0.010:  # > 10ms
            bottlenecks.append("Video overlay is slow")
        
        if avg_cpu > 80:
            bottlenecks.append("High CPU usage")
        
        if avg_memory > 80:
            bottlenecks.append("High memory usage")
        
        return bottlenecks
    
    def _get_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        bottlenecks = self._identify_bottlenecks()
        
        if "Frame processing too slow for real-time" in bottlenecks:
            recommendations.append("Consider reducing image resolution or feature count")
        
        if "Marker detection is slow" in bottlenecks:
            recommendations.append("Try using ORB instead of SIFT, or reduce max_features")
        
        if "Video overlay is slow" in bottlenecks:
            recommendations.append("Optimize video resolution or use simpler blending")
        
        if "High CPU usage" in bottlenecks:
            recommendations.append("Enable GPU acceleration if available")
        
        if "High memory usage" in bottlenecks:
            recommendations.append("Enable memory pooling and cleanup")
        
        if not recommendations:
            recommendations.append("Performance is optimal")
        
        return recommendations

# Global optimizer instance
global_optimizer = AROptimizer()
global_memory_optimizer = MemoryOptimizer()
global_performance_monitor = PerformanceMonitor()
