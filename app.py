"""
AI Home Security System - Streamlit App (single file)

Save this file as ai_home_security_app.py and run:
    pip install -r requirements.txt
    streamlit run ai_home_security_app.py

requirements.txt minimal:
    streamlit
    opencv-python
    numpy
    pandas

Optional (for much better person detection / pose-based fall detection):
    ultralytics   # provides YOLOv8 (will auto-download model weights on first run)
    mediapipe     # for pose-based fall detection

What this app does (prototype):
- Supports Webcam (device 0) or uploaded video file
- Detects persons (YOLOv8 if available, otherwise HOG) and draws bounding boxes
- Motion detection using MOG2 background subtractor
- Simple color-based Fire detection (HSV masks)
- Fall detection heuristic: either pose-based (Mediapipe if available) or bbox-height drop
- Event logging with timestamps and CSV download

This is a prototype for experimentation & IPD. Not production-ready. Carefully test before any deployment.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
from pathlib import Path
from email_alert import send_email_alert

st.set_page_config(page_title="AI Home Security System", layout="wide")

# --------------------------- Helper: Optional imports ---------------------------
USE_YOLO = False
USE_POSE = False

try:
    from ultralytics import YOLO
    USE_YOLO = True
except Exception:
    USE_YOLO = False

try:
    import mediapipe as mp
    USE_POSE = True
except Exception:
    USE_POSE = False

# --------------------------- UI: Title & Description ---------------------------
st.title("AI Home Security System — Prototype")
st.markdown("""
Lightweight prototype that detects: **intruder (person)**, **motion**, **fire-like colors**, and **fall-like** events.
- Uses **YOLOv8** if `ultralytics` is installed (recommended). Falls back to OpenCV HOG detector otherwise.
- Uses **MediaPipe** pose if available for better fall detection; falls back to bbox-height heuristic.
""")

# --------------------------- Sidebar controls ---------------------------
st.sidebar.header("Video & Detection Settings")
video_source = st.sidebar.radio("Video source", ("Webcam (0)", "Upload file"))
use_yolo_switch = st.sidebar.checkbox("Use YOLOv8 (if installed)", value=USE_YOLO)
use_pose_switch = st.sidebar.checkbox("Use Mediapipe pose for fall detection (if installed)", value=USE_POSE)

motion_threshold = st.sidebar.slider("Motion sensitivity (percent of frame)", 1, 50, 8)
fire_threshold = st.sidebar.slider("Fire-pixel threshold (%)", 1, 20, 5)
fall_drop_ratio = st.sidebar.slider("Fall drop ratio (%) (bbox method)", 10, 70, 40)
min_person_area = st.sidebar.slider("Min person box area (px)", 1000, 50000, 4000)
show_fps = st.sidebar.checkbox("Show FPS", True)
record_output = st.sidebar.checkbox("Record detection output to file", False)

st.sidebar.markdown("---")
st.sidebar.write("Start / Stop")
start_btn = st.sidebar.button("Start")
stop_btn = st.sidebar.button("Stop")

# --------------------------- State ---------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "events" not in st.session_state:
    st.session_state.events = []
if "last_person_h" not in st.session_state:
    st.session_state.last_person_h = None
if "capture_path" not in st.session_state:
    st.session_state.capture_path = None

# --------------------------- Models initialization ---------------------------
model = None
pose_mp = None
pose_detector = None

if use_yolo_switch and USE_YOLO:
    try:
        # Use YOLOv8n by default (ultralytics will download weights first run)
        model = YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Failed to load YOLO: {e}")
        model = None

if use_pose_switch and USE_POSE:
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Fallback HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Background subtractor for motion
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# --------------------------- UI layout ---------------------------
video_col, log_col = st.columns([2, 1])
video_placeholder = video_col.empty()
log_placeholder = log_col.empty()

st.markdown("---")
st.markdown("**Notes:** This prototype is for learning & prototyping. Replace simple detectors with robust models for production (YOLOv8, pose-based fall detection, CNN for fire).")

# --------------------------- File uploader (only shown when needed) ---------------------------
uploaded_file = None
if video_source == "Upload file":
    uploaded_file = st.sidebar.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"]) 

# --------------------------- Helper functions ---------------------------

def log_event(kind, details=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.events.append({"time": now, "event": kind, "details": details})
    # update log display (newest first)
    df = pd.DataFrame(st.session_state.events[::-1])
    log_placeholder.dataframe(df, height=600)


def init_video_capture():
    # Returns cv2.VideoCapture or None
    if video_source == "Webcam (0)":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None
        return cap
    else:
        if uploaded_file is None:
            return None
        # Save upload to temp file
        tmp_dir = Path("./tmp_uploads")
        tmp_dir.mkdir(exist_ok=True)
        tpath = tmp_dir / f"uploaded_{int(time.time())}.mp4"
        with open(tpath, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.capture_path = str(tpath)
        cap = cv2.VideoCapture(str(tpath))
        if not cap.isOpened():
            return None
        return cap


def detect_persons_yolo(frame):
    # returns list of (x,y,w,h,conf,class_id,label)
    results = []
    # ultralytics YOLO expects RGB numpy
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yres = model(rgb)[0]
    for det in yres.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        # COCO class 0 == person
        if cls == 0:
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            results.append((x,y,w,h,float(conf),cls,'person'))
    return results


def detect_persons_hog(frame):
    # HOG expects grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects, weights = hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
    out = []
    for (x,y,w,h),score in zip(rects, weights):
        if w*h < min_person_area:
            continue
        out.append((int(x),int(y),int(w),int(h), float(score), 0, 'person'))
    return out


def pose_fall_check(frame):
    # Return True if pose suggests fall (based on orientation of torso / hips) — heuristic
    if not USE_POSE or pose_detector is None:
        return False, None
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    if not results.pose_landmarks:
        return False, None
    lm = results.pose_landmarks.landmark
    # get key points: nose (0), left_hip(23), right_hip(24)
    nose = lm[0]
    lh = lm[23]
    rh = lm[24]
    # compute vertical distance nose to hip in normalized coordinates
    hip_y = (lh.y + rh.y) / 2.0
    nose_y = nose.y
    # if nose is close to hip (y difference small) and person oriented horizontally -> possible fall
    diff = abs(hip_y - nose_y)
    # This is a loose heuristic — tune after experiments
    if diff < 0.05:
        return True, {'nose_y': nose_y, 'hip_y': hip_y, 'diff': diff}
    return False, {'nose_y': nose_y, 'hip_y': hip_y, 'diff': diff}


def fire_detection(frame):
    # Returns (fire_percent, mask)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # ranges for orange-yellow and red
    lower1 = np.array([0, 120, 120])
    upper1 = np.array([25, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    lower2 = np.array([160, 120, 120])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    fire_px = np.sum(mask > 0)
    total = frame.shape[0] * frame.shape[1]
    percent = (fire_px / total) * 100.0
    return percent, mask

# --------------------------- Main loop ---------------------------

def run_detection_loop():
    cap = init_video_capture()
    if cap is None:
        st.error("Cannot open video source. If using webcam, ensure permissions and that 'opencv-python' (not headless) is installed.")
        st.session_state.running = False
        return

    # prepare output video writer if recording
    writer = None
    if record_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = f"detection_out_{int(time.time())}.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
        st.sidebar.success(f"Recording output to {out_path}")

    placeholder = video_placeholder
    last_time = time.time()
    fps = 0.0
    frame_idx = 0

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.info("End of video or cannot fetch frame.")
            break
        frame_idx += 1
        annotated = frame.copy()

        # MOTION
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_area = np.sum(fgmask > 0)
        motion_percent = (motion_area / (frame.shape[0]*frame.shape[1])) * 100.0
        if motion_percent > motion_threshold:
            cv2.putText(annotated, f"Motion: {motion_percent:.2f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            log_event("Motion", f"{motion_percent:.2f}% area")

        # FIRE
        fire_percent, fire_mask = fire_detection(frame)
        if fire_percent > fire_threshold:
            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            image_path = f"fire_{timestamp}.jpg"

            cv2.imwrite(image_path, frame)

            send_email_alert("Fire", timestamp, image_path)
        

    
            # overlay mask in red
            annotated[fire_mask > 0] = (0,0,255)
            cv2.putText(annotated, f"Fire: {fire_percent:.3f}%", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            log_event("Fire", f"{fire_percent:.3f}% fire-like pixels")



        # PERSONS detection
        persons = []
        if use_yolo_switch and model is not None:
            try:
                persons = detect_persons_yolo(frame)
            except Exception as e:
                # model may fail on some frames; fallback to HOG
                persons = detect_persons_hog(frame)
        else:
            persons = detect_persons_hog(frame)

        # draw person boxes; compute largest for fall heuristic
        largest_h = None
        if len(persons) > 0:
            for (x,y,w,h,conf,cls,label) in persons:
                cv2.rectangle(annotated, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(annotated, f"Person {conf:.2f}" if conf is not None else "Person", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # pick largest by area
            persons_sorted = sorted(persons, key=lambda r: r[2]*r[3], reverse=True)
            _,_,w,h,_,_,_ = persons_sorted[0]
            largest_h = h
            # log person event
            log_event("Person", f"{len(persons)} detected; largest_h={largest_h}")

        # FALL DETECTION
        fall_detected = False
        fall_info = ""
        if use_pose_switch and USE_POSE and pose_detector is not None:
            # run pose-based check on whole frame (expensive but more accurate)
            fall, info = pose_fall_check(frame)
            if fall:
                fall_detected = True
                fall_info = f"pose heuristic {info}"
        else:
            # bbox heuristic
            if largest_h is not None and st.session_state.last_person_h is not None:
                drop = (st.session_state.last_person_h - largest_h) / max(st.session_state.last_person_h, 1) * 100.0
                if drop > fall_drop_ratio and drop > 15:
                    fall_detected = True
                    fall_info = f"height drop {drop:.1f}%"
            # update last_person_h
            if largest_h is not None:
                st.session_state.last_person_h = largest_h
            else:
                # decay to avoid stale values
                if st.session_state.last_person_h is not None:
                    st.session_state.last_person_h *= 0.95

        if fall_detected:
            cv2.putText(annotated, "POSSIBLE FALL", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 3)
            log_event("Fall-like", fall_info)


            timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            image_path = f"fall_{timestamp}.jpg"

            cv2.imwrite(image_path, frame)

            send_email_alert("Fall", timestamp, image_path)

            

        # FPS
        now = time.time()
        if show_fps:
            inst_fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
            fps = 0.9*fps + 0.1*inst_fps
            last_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, annotated.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Write output file if enabled
        if writer is not None:
            writer.write(annotated)

        # Show in streamlit
        placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        # small sleep to let UI breathe
        time.sleep(0.01)

    cap.release()
    if writer is not None:
        writer.release()
        st.sidebar.success(f"Saved output video to {out_path}")

# --------------------------- Start/Stop handling ---------------------------

if start_btn:
    st.session_state.running = True
    st.success("Starting detection loop — press Stop to end")

if stop_btn:
    st.session_state.running = False
    st.warning("Stopping detection loop — please wait for loop to end")

# If running, run detection loop
if st.session_state.running:
    run_detection_loop()

# Show log & download
st.sidebar.markdown("---")
st.sidebar.markdown("### Event log")
if len(st.session_state.events) > 0:
    df = pd.DataFrame(st.session_state.events[::-1])
    st.sidebar.dataframe(df.head(20))
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download events CSV", csv, file_name="events_log.csv", mime="text/csv")
else:
    st.sidebar.write("No events yet.")

# Small instructions at end
st.markdown("---")
st.caption("Tip: To improve detection accuracy, install 'ultralytics' for YOLOv8 and optionally 'mediapipe' for pose-based fall detection.\nUse 'pip install ultralytics mediapipe' and restart the app.")
