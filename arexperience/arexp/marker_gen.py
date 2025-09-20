import cv2
import msgpack
import os
import logging

logger = logging.getLogger(__name__)

def generate_simple_mind_file(image_path, output_path):
    """MindAR-compatible marker file generation - FIXED VERSION"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Cannot read image: {image_path}")
            return False

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.info(f"Processing image: {width}x{height}")

        orb = cv2.ORB_create(nfeatures=50)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if len(keypoints) < 4:
            # Create minimal corner keypoints if not enough features
            keypoints_data = [
                {"x": width*0.25, "y": height*0.25, "scale": 8.0, "orientation": 0.0, "response": 1.0},
                {"x": width*0.75, "y": height*0.25, "scale": 8.0, "orientation": 0.0, "response": 1.0},
                {"x": width*0.25, "y": height*0.75, "scale": 8.0, "orientation": 0.0, "response": 1.0},
                {"x": width*0.75, "y": height*0.75, "scale": 8.0, "orientation": 0.0, "response": 1.0}
            ]
            descriptors_data = [bytes(32)] * 4
            logger.info("Using minimal corner keypoints")
        else:
            # Use detected keypoints (limited)
            keypoints = keypoints[:15]  # Keep it small
            keypoints_data = [{
                "x": float(kp.pt[0]),
                "y": float(kp.pt[1]),
                "scale": float(kp.size),
                "orientation": float(kp.angle if kp.angle is not None else 0.0),
                "response": float(kp.response if hasattr(kp, 'response') else 1.0)
            } for kp in keypoints]
            
            if descriptors is not None and len(descriptors) > 0:
                descriptors_data = [desc.tobytes() for desc in descriptors[:len(keypoints)]]
            else:
                descriptors_data = [bytes(32) for _ in range(len(keypoints_data))]
            
            logger.info(f"Using {len(keypoints)} detected keypoints")

        # CRITICAL: Use ONLY the basic structure MindAR expects
        data = {
            "imageWidth": width,
            "imageHeight": height,
            "targets": [{
                "keypoints": keypoints_data,
                "descriptors": descriptors_data
            }]
        }

        packed = msgpack.packb(data, use_bin_type=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(packed)

        file_size = os.path.getsize(output_path)
        logger.info(f"✅ Mind file created: {file_size} bytes")
        return True

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return False
