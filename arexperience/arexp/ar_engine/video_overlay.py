# arexp/ar_engine/video_overlay.py
import cv2
import numpy as np
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class PythonVideoAR:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def create_marker_from_image(self, image_path: str) -> dict:
        """Create marker data from uploaded image - Pure Python"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect features
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if len(keypoints) < 20:
                logger.warning(f"Only {len(keypoints)} features found, may not track well")
            
            # Convert keypoints to serializable format
            kp_data = []
            for kp in keypoints:
                kp_data.append({
                    'x': float(kp.pt[0]),
                    'y': float(kp.pt[1]),
                    'angle': float(kp.angle),
                    'size': float(kp.size),
                    'response': float(kp.response)
                })
            
            marker_data = {
                'image_path': image_path,
                'width': img.shape[1],
                'height': img.shape[0],
                'keypoints': kp_data,
                'descriptors': descriptors.tobytes() if descriptors is not None else b'',
                'feature_count': len(keypoints)
            }
            
            logger.info(f"✅ Marker created: {len(keypoints)} features from {image_path}")
            return marker_data
            
        except Exception as e:
            logger.error(f"❌ Marker creation failed: {str(e)}")
            return None
    
    def overlay_video_on_marker(self, camera_frame: np.ndarray, 
                               marker_data: dict, video_frame: np.ndarray,
                               overlay_size: Tuple[int, int] = (200, 150)) -> np.ndarray:
        """Overlay video on detected marker in camera frame"""
        try:
            # Convert camera frame to grayscale for detection
            gray_camera = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features in camera frame
            kp_camera, des_camera = self.orb.detectAndCompute(gray_camera, None)
            
            if des_camera is None or len(kp_camera) < 10:
                return camera_frame
            
            # Reconstruct marker descriptors
            marker_descriptors = np.frombuffer(marker_data['descriptors'], dtype=np.uint8)
            if len(marker_descriptors) == 0:
                return camera_frame
                
            marker_descriptors = marker_descriptors.reshape(-1, 32)  # ORB descriptors are 32 bytes
            
            # Match features
            matches = self.matcher.match(marker_descriptors, des_camera)
            
            if len(matches) < 10:
                return camera_frame
                
            # Sort matches by distance (quality)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:min(50, len(matches))]
            
            # Extract matched keypoints
            src_pts = []
            dst_pts = []
            
            for match in good_matches:
                # Source points from marker
                marker_kp = marker_data['keypoints'][match.queryIdx]
                src_pts.append([marker_kp['x'], marker_kp['y']])
                
                # Destination points from camera
                camera_kp = kp_camera[match.trainIdx]
                dst_pts.append(camera_kp.pt)
            
            src_pts = np.float32(src_pts).reshape(-1, 1, 2)
            dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
            
            # Find homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                 cv2.RANSAC, 5.0)
            
            if homography is None:
                return camera_frame
            
            # Define marker corners in original image
            h, w = marker_data['height'], marker_data['width']
            marker_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            
            # Transform marker corners to camera frame
            transformed_corners = cv2.perspectiveTransform(marker_corners, homography)
            
            # Resize video frame to overlay size
            video_resized = cv2.resize(video_frame, overlay_size)
            
            # Define video corners for perspective transform
            video_corners = np.float32([[0, 0], [overlay_size[0], 0], 
                                      [overlay_size[0], overlay_size[1]], [0, overlay_size[1]]])
            
            # Calculate perspective transform matrix
            perspective_matrix = cv2.getPerspectiveTransform(video_corners, 
                                                           transformed_corners.reshape(4, 2))
            
            # Warp video to match marker perspective
            warped_video = cv2.warpPerspective(video_resized, perspective_matrix, 
                                             (camera_frame.shape[1], camera_frame.shape[0]))
            
            # Create mask for blending
            mask = np.zeros(camera_frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(transformed_corners)], 255)
            mask_inv = cv2.bitwise_not(mask)
            
            # Apply mask to camera frame
            camera_bg = cv2.bitwise_and(camera_frame, camera_frame, mask=mask_inv)
            video_fg = cv2.bitwise_and(warped_video, warped_video, mask=mask)
            
            # Combine
            result = cv2.add(camera_bg, video_fg)
            
            # Optional: Draw marker outline for debugging
            cv2.polylines(result, [np.int32(transformed_corners)], True, (0, 255, 0), 2)
            
            return result
            
        except Exception as e:
            logger.error(f"Video overlay failed: {str(e)}")
            return camera_frame
