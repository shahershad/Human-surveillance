import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import os

st.set_page_config(page_title="YOLOv8 Human & Luggage", layout="wide")
st.title("YOLOv8 Detection: Human & Luggage 🚀")

# ---------------- Settings ----------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Model:", ["Custom (best.pt)", "COCO (filtered)"], index=0)
CONF = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Allowed class remap for COCO
allowed_classes = {"person": "human", "suitcase": "luggage"}

# Load models once
@st.cache_resource
def load_custom():
    return YOLO("models/best.pt")   # <-- your trained model here

@st.cache_resource
def load_coco():
    return YOLO("yolov8n.pt")       # COCO pretrained

custom_model = load_custom()
coco_model = load_coco()

# ---------------- Detection logic ----------------
def process_frame(frame, mode, conf, custom_model, coco_model):
    """Run YOLO inference and filter if needed."""
    if mode == "Custom (best.pt)":
        res = custom_model(frame, conf=conf)[0]
    elif mode == "COCO (filtered)":
        res = coco_model(frame, conf=conf)[0]
        mask = [coco_model.names[int(c)] in allowed_classes for c in res.boxes.cls]
        res.boxes = Boxes(res.boxes.data[mask], res.orig_shape)
        for c in res.boxes.cls:
            old = coco_model.names[int(c)]
            res.names[int(c)] = allowed_classes[old]
    return res.plot()

def annotate_video(input_path, output_path, mode, conf, custom_model, coco_model):
    """Process full video and save annotated MP4 with progress bar."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    progress = st.progress(0)
    processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated = process_frame(frame, mode, conf, custom_model, coco_model)
        out.write(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))  # save frame

        processed += 1
        if processed % 5 == 0 or processed == total_frames:  # update less often for speed
            pct = int(processed / total_frames * 100)
            progress.progress(min(pct, 100))

    cap.release()
    out.release()
    progress.progress(100)  # ensure full bar
    return True

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📷 Image", "🎞️ Video File", "🎥 Live Camera"])

# ---------- Tab 1: Image ----------
with tab1:
    st.subheader("Test Image or Upload")
    use_sample = st.checkbox("Use sample image (/samples/test_image.jpg)")
    img_file = None
    if not use_sample:
        img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if use_sample:
        frame = cv2.imread("samples/test_image.jpg")
    elif img_file:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        frame = None

    if frame is not None:
        annotated = process_frame(frame, mode, CONF, custom_model, coco_model)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption=f"Detections ({mode})", use_container_width=True)

# ---------- Tab 2: Video File ----------
with tab2:
    st.subheader("Test Video or Upload")
    use_sample_vid = st.checkbox("Use sample video (/samples/test_video.mp4)")
    vid_file = None
    if not use_sample_vid:
        vid_file = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"])

    if use_sample_vid:
        video_path = "samples/test_video.mp4"
    elif vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_file.read())
        video_path = tfile.name
    else:
        video_path = None

    if video_path:
        t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        st.info("⏳ Processing video... please wait.")
        success = annotate_video(video_path, t_out.name, mode, CONF, custom_model, coco_model)
        if success:
            st.success("✅ Done! Annotated video is ready:")
            st.video(t_out.name)  # just play, no download button

# ---------- Tab 3: Live Camera ----------
with tab3:
    st.subheader("Webcam Stream")
    if os.environ.get("STREAMLIT_RUNTIME"):  # detect Cloud
        st.info("⚠️ Webcam not supported on Streamlit Cloud.")
    else:
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])
        if run:
            cap = cv2.VideoCapture(0)
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame")
                    break
                annotated = process_frame(frame, mode, CONF, custom_model, coco_model)
                FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   caption=f"Live Detection ({mode})",
                                   use_container_width=True)
            cap.release()
