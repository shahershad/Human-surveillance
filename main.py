import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import os
from glob import glob
from collections import defaultdict

st.set_page_config(page_title="YOLOv8 Human & Luggage", layout="wide")
st.title("YOLOv8 Detection: Human & Luggage 🚀")

# ---------------- Settings ----------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Model Source:", ["Custom (choose .pt)", "COCO (filtered)"], index=0)
CONF = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Allowed class remap for COCO
allowed_classes = {"person": "human", "suitcase": "luggage"}

# --------------- Helpers ----------------
def to_cpu_int_list(tensor_like):
    try:
        return [int(x) for x in tensor_like.detach().cpu().tolist()]
    except Exception:
        return [int(x) for x in tensor_like]

def count_detections(res, mode, allowed_map):
    counts = defaultdict(int)
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return dict(counts)
    cls_ids = to_cpu_int_list(res.boxes.cls)
    for c in cls_ids:
        name = res.names[int(c)]
        if mode == "COCO (filtered)":
            name = allowed_map.get(name, None)
        if name is not None:
            counts[name] += 1
    return dict(counts)

def render_counts(title, counts_dict):
    with st.expander(title, expanded=True):
        if not counts_dict:
            st.info("No detections.")
            return
        cols = st.columns(max(1, len(counts_dict)))
        for i, (k, v) in enumerate(sorted(counts_dict.items())):
            cols[i % len(cols)].metric(k.capitalize(), v)

def list_custom_models(models_dir="models"):
    """Return sorted list of .pt files under models_dir (non-recursive)."""
    paths = sorted(glob(os.path.join(models_dir, "*.pt")))
    return paths

@st.cache_resource
def load_yolo_model(path_or_name: str):
    """Cache models by path/name."""
    return YOLO(path_or_name)

# Preload COCO once
coco_model = load_yolo_model("yolov8n.pt")

# Sidebar custom model picker
selected_custom_path = None
if mode == "Custom (choose .pt)":
    candidates = list_custom_models("models")
    if not candidates:
        st.sidebar.error("No .pt files found in ./models. Place your custom checkpoints there.")
    else:
        selected_custom_path = st.sidebar.selectbox(
            "Choose custom checkpoint (.pt):",
            options=candidates,
            index=0,
            help="All .pt files found in ./models"
        )
        # Optional: quick peek at model labels
        try:
            _tmp_model = load_yolo_model(selected_custom_path)
            st.sidebar.caption(f"Classes: {len(_tmp_model.names)} → {list(_tmp_model.names.values())[:5]}{'...' if len(_tmp_model.names)>5 else ''}")
        except Exception as e:
            st.sidebar.warning(f"Could not load names from {selected_custom_path}: {e}")

# ---------------- Detection logic ----------------
def process_frame(frame, mode, conf, selected_custom_path, coco_model):
    """
    Returns (annotated_frame_bgr, result_obj, per_frame_counts)
    """
    if mode == "Custom (choose .pt)":
        if not selected_custom_path:
            return frame, None, {}
        custom_model = load_yolo_model(selected_custom_path)
        res = custom_model(frame, conf=conf, verbose=False)[0]

    elif mode == "COCO (filtered)":
        res = coco_model(frame, conf=conf, verbose=False)[0]
        # Keep only allowed classes
        cls_ids = to_cpu_int_list(res.boxes.cls) if res.boxes is not None else []
        if len(cls_ids) and res.boxes is not None:
            mask = np.array([coco_model.names[int(c)] in allowed_classes for c in cls_ids], dtype=bool)
            res.boxes = Boxes(res.boxes.data[mask], res.orig_shape)
        # Update label mapping for display
        if res.boxes is not None and len(res.boxes):
            for i_c in to_cpu_int_list(res.boxes.cls):
                old_name = coco_model.names[int(i_c)]
                new_name = allowed_classes.get(old_name, old_name)
                res.names[int(i_c)] = new_name

    annotated = res.plot()  # BGR
    counts = count_detections(res, mode, allowed_classes)
    return annotated, res, counts

def annotate_video(input_path, output_path, mode, conf, selected_custom_path, coco_model):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_placeholder = st.empty()
    perframe_counts_placeholder = st.empty()
    cumulative_counts_placeholder = st.empty()

    cumulative = defaultdict(int)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated, res, counts = process_frame(frame, mode, conf, selected_custom_path, coco_model)
        out.write(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        for k, v in counts.items():
            cumulative[k] += v

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"Video Detection ({mode}) — frame {frame_idx}",
            use_container_width=True
        )
        with perframe_counts_placeholder.container():
            render_counts("Per-frame counts", counts)
        with cumulative_counts_placeholder.container():
            render_counts("Cumulative counts (this run)", dict(cumulative))

        frame_idx += 1
        time.sleep(0.01)

    cap.release()
    out.release()
    return True

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📷 Image", "🎞️ Video File", "🎥 Live Camera"])

# ---------- Tab 1: Image ----------
with tab1:
    st.subheader("Test Image or Upload")
    use_sample = st.checkbox("Use sample image from /samples/test_image.jpg")
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
        annotated, res, counts = process_frame(frame, mode, CONF, selected_custom_path, coco_model)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption=f"Detections ({mode})", use_container_width=True)
        render_counts("Image counts", counts)

# ---------- Tab 2: Video File ----------
with tab2:
    st.subheader("Test Video or Upload")
    use_sample_vid = st.checkbox("Use sample video from /samples/test_video.mp4")
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
        success = annotate_video(video_path, t_out.name, mode, CONF, selected_custom_path, coco_model)
        if success:
            st.success("Processing complete ✅")
            st.video(t_out.name)
            with open(t_out.name, "rb") as f:
                st.download_button(
                    "Download Annotated Video",
                    data=f,
                    file_name="annotated_output.mp4",
                    mime="video/mp4"
                )

# ---------- Tab 3: Live Camera ----------
with tab3:
    st.subheader("Webcam Stream")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    perframe_counts_placeholder = st.empty()
    cumulative_counts_placeholder = st.empty()
    cumulative = defaultdict(int)

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break
            annotated, res, counts = process_frame(frame, mode, CONF, selected_custom_path, coco_model)
            for k, v in counts.items():
                cumulative[k] += v
            FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               caption=f"Live Detection ({mode})", use_container_width=True)
            with perframe_counts_placeholder.container():
                render_counts("Per-frame counts", counts)
            with cumulative_counts_placeholder.container():
                render_counts("Cumulative counts (this session)", dict(cumulative))
        cap.release()
