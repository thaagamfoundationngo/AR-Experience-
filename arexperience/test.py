import cv2
import numpy as np

# --- Load reference image (the image you want to detect) ---
ref_img = cv2.imread("1(5).jpeg", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(1000)  # Feature detector
kp_ref, des_ref = orb.detectAndCompute(ref_img, None)

# --- Setup video capture (webcam + overlay video) ---
cap = cv2.VideoCapture(0)  # webcam
video = cv2.VideoCapture("overlay.mp4")  # video to overlay

# FLANN matcher for feature matching
index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6, key_size=12, multi_probe_level=1)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect keypoints in webcam frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is not None:
        matches = flann.knnMatch(des_ref, des_frame, k=2)

        # Apply Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > 15:  # enough matches = found object
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # Homography to map video on image
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = ref_img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Read frame from overlay video
            ret_v, overlay = video.read()
            if not ret_v:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                ret_v, overlay = video.read()

            overlay = cv2.resize(overlay, (w, h))

            # Warp video onto detected plane
            warp_overlay = cv2.warpPerspective(overlay, M, (frame.shape[1], frame.shape[0]))

            # Create mask for overlay
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst), 255)
            mask_inv = cv2.bitwise_not(mask)

            # Combine frame + overlay
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            frame = cv2.add(frame_bg, warp_overlay)

    cv2.imshow("AR Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()