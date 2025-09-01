# camera_test.py
import streamlit as st
import cv2
import numpy as np

st.title("Camera Test for AR Debugging")

# Create tabs for different tests
tab1, tab2, tab3 = st.tabs(["Basic Camera", "OpenCV Test", "WebRTC Test"])

with tab1:
    st.header("Streamlit Camera Input")
    
    # Basic camera input
    picture = st.camera_input("Take a picture")
    
    if picture is not None:
        st.image(picture, caption="Camera is working!")
        st.success("✅ Camera access successful")

with tab2:
    st.header("OpenCV Camera Stream")
    
    if st.button("Test OpenCV Camera"):
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="OpenCV Camera Working")
                st.success("✅ OpenCV can access camera")
            else:
                st.error("❌ OpenCV cannot read from camera")
        else:
            st.error("❌ OpenCV cannot open camera")
        
        cap.release()

with tab3:
    st.header("Browser Camera Permissions")
    st.info("Check browser console for getUserMedia errors")
    
    # JavaScript to test getUserMedia
    camera_js = """
    <script>
    async function testCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: true, 
                audio: false 
            });
            console.log("✅ getUserMedia successful:", stream);
            document.getElementById('camera-status').innerHTML = 
                '<p style="color: green;">✅ Browser camera access works!</p>';
            
            // Stop the stream
            stream.getTracks().forEach(track => track.stop());
        } catch (error) {
            console.error("❌ getUserMedia failed:", error);
            document.getElementById('camera-status').innerHTML = 
                '<p style="color: red;">❌ Camera access failed: ' + error.message + '</p>';
        }
    }
    testCamera();
    </script>
    <div id="camera-status">Testing camera...</div>
    """
    
    st.components.v1.html(camera_js, height=100)

# System information
st.sidebar.header("System Info")
st.sidebar.text("This helps debug camera issues")

# Add debugging tips
st.sidebar.markdown("""
**Common Camera Issues:**
- Browser permissions not granted
- Camera in use by another app
- HTTPS required for getUserMedia
- Wrong camera index in OpenCV
- Hardware/driver issues
""")
