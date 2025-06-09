# app.py
import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---- CONFIG ----
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_THRESHOLD = 0.15  # fraction of frame

# ---- LOAD HAAR CASCADES ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

class FaceAnalyzer(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
        posture = "No face"
        gaze    = "N/A"

        if len(faces):
            x, y, w, h = faces[0]
            cx, cy = x + w//2, y + h//2

            # draw face box + center dot
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.circle(img, (cx,cy), 5, (0,0,255), -1)

            # central “posture” box
            xl = int(FRAME_WIDTH*(0.5 - CENTER_THRESHOLD))
            xh = int(FRAME_WIDTH*(0.5 + CENTER_THRESHOLD))
            yl = int(FRAME_HEIGHT*(0.5 - CENTER_THRESHOLD))
            yh = int(FRAME_HEIGHT*(0.5 + CENTER_THRESHOLD))
            cv2.rectangle(img, (xl, yl), (xh, yh), (255,255,0), 1)
            posture = "Centered" if (xl < cx < xh and yl < cy < yh) else "Not centered"

            # gaze: detect eyes & check vertical alignment
            eyes = eye_cascade.detectMultiScale(
                gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=5
            )
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                (ex1,ey1,ew1,eh1), (ex2,ey2,ew2,eh2) = eyes
                y1 = y + ey1 + eh1//2
                y2 = y + ey2 + eh2//2
                gaze = "Looking" if abs(y1 - y2) < h*0.1 else "Not looking"

        # overlay text
        cv2.putText(img, f"Posture: {posture}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"Gaze: {gaze}", (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---- STREAMLIT UI ----
st.title("🔴 Live Webcam Analytics (Haar Cascade)")
st.markdown(
    """
    - **Face**: Haar‐cascade detection  
    - **Posture**: centered in frame?  
    - **Gaze**: simple eye‐alignment check  
    """
)

webrtc_streamer(
    key="haar-face",
    mode="SENDRECV",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FaceAnalyzer,
    media_stream_constraints={
        "video": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
        "audio": False
    },
)
