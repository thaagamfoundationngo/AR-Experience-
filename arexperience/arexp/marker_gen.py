import cv2
import msgpack
import os
import logging

logger = logging.getLogger(__name__)

def generate_simple_mind_file(image_path, output_path):
    """Simple .mind file generation without extra data"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=50)
        keypoints, _ = orb.detectAndCompute(gray, None)
        
        if len(keypoints) < 10:
            return False
        
        # Minimal data structure with required fields
        data = {
            "imageWidth": width,
            "imageHeight": height,
            "targets": [{
                "keypoints": [[float(kp.pt[0]), float(kp.pt[1])] for kp in keypoints[:30]],
                "descriptors": [[0] * 32 for _ in range(min(30, len(keypoints)))]
            }]
        }
        
        packed = msgpack.packb(data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(packed)
        
        return True
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False