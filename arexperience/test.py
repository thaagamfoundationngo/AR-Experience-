import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        # Load target image
        self.target_img = cv2.imread('target.jpeg')
        if self.target_img is None:
            raise ValueError("Target image not found!")
        self.target_gray = cv2.cvtColor(self.target_img, cv2.COLOR_BGR2GRAY)
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.kp_target, self.des_target = self.orb.detectAndCompute(self.target_gray, None)
        
        # Initialize video
        self.video_path = 'target_video.mp4'
        self.video_cap = cv2.VideoCapture(self.video_path)
        if not self.video_cap.isOpened():
            raise ValueError(f"Video '{self.video_path}' not found!")

    def transform(self, frame):
        # Convert frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des_target, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) > 15:  # Minimum matches for detection
            # Extract matched points
            src_pts = np.float32([self.kp_target[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Compute homography
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                h, w = self.target_gray.shape[:2]
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Read video frame
                ret_video, video_frame = self.video_cap.read()
                if not ret_video:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    ret_video, video_frame = self.video_cap.read()
                
                if ret_video:
                    # Resize and warp video to fit detected image
                    resized_video = cv2.resize(video_frame, (w, h))
                    warped_video = cv2.warpPerspective(resized_video, M, (img.shape[1], img.shape[0]))
                    
                    # Create mask and overlay video
                    mask = np.zeros_like(img)
                    cv2.fillPoly(mask, [np.int32(dst)], (255, 255, 255))
                    img = cv2.bitwise_and(img, cv2.bitwise_not(mask))
                    img = cv2.add(img, warped_video)
        
        return img

def main():
    st.title("Mobile-Compatible AR Video Overlay")
    
    ctx = webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)
    
    if ctx.video_processor:
        st.write("Point the camera at the target image to see the video overlay.")

if __name__ == "__main__":
    main()