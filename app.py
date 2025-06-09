import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---- CONFIG ----
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_THRESHOLD = 0.15  # fraction of frame size

# ---- MEDIAPIPE SETUP ----
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class FaceAnalyzer(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))

        # Face detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det = mp_face_detection.process(rgb)
        centered = "No face"
        looking = "N/A"

        if det.detections:
            d = det.detections[0]
            # bounding box
            bbox = d.location_data.relative_bounding_box
            x1 = int(bbox.xmin * FRAME_WIDTH)
            y1 = int(bbox.ymin * FRAME_HEIGHT)
            w  = int(bbox.width * FRAME_WIDTH)
            h  = int(bbox.height * FRAME_HEIGHT)
            cx, cy = x1 + w//2, y1 + h//2

            # draw box + center marker
            cv2.rectangle(img, (x1,y1), (x1+w, y1+h), (0,255,0), 2)
            cv2.circle(img, (cx,cy), 5, (0,0,255), -1)

            # posture: is face center within central zone?
            x_low = int(FRAME_WIDTH*(0.5 - CENTER_THRESHOLD))
            x_high = int(FRAME_WIDTH*(0.5 + CENTER_THRESHOLD))
            y_low = int(FRAME_HEIGHT*(0.5 - CENTER_THRESHOLD))
            y_high = int(FRAME_HEIGHT*(0.5 + CENTER_THRESHOLD))
            cv2.rectangle(img, (x_low, y_low), (x_high, y_high), (255,255,0), 1)
            centered = "Centered" if (x_low < cx < x_high and y_low < cy < y_high) else "Not centered"

            # face mesh for iris landmarks → approximate gaze
            mesh = mp_face_mesh.process(rgb)
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark
                # left iris center ≈ mean of landmarks 474–478
                left = np.mean([[lm[i].x, lm[i].y] for i in range(474,479)], axis=0)
                right = np.mean([[lm[i].x, lm[i].y] for i in range(469,474)], axis=0)
                # convert to pixel coords
                l_px = np.array([left[0]*FRAME_WIDTH, left[1]*FRAME_HEIGHT])
                r_px = np.array([right[0]*FRAME_WIDTH, right[1]*FRAME_HEIGHT])
                # if both irises are close to frame center horizontally → looking forward
                avg_x = (l_px[0] + r_px[0]) / 2
                looking = "Looking" if abs(avg_x - FRAME_WIDTH/2) < w*0.15 else "Not looking"

        # overlay text
        cv2.putText(img, f"Posture: {centered}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"Gaze: {looking}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---- STREAMLIT UI ----
st.title("🔴 Live Webcam Analytics")
st.markdown(
    """
    - **Face position**: box + center marker  
    - **Posture**: is your face in the central zone?  
    - **Gaze**: looking straight at camera or not  
    """
)

webrtc_streamer(
    key="face-analytics",
    mode="SENDRECV",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FaceAnalyzer,
    media_stream_constraints={
        "video": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT},
        "audio": False
    }
)
