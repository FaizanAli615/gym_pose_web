from __future__ import annotations
import io
import os
import time
from collections import defaultdict

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from pose_engine import EXERCISES, GymPoseEngine, SharedState

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 250 * 1024 * 1024
MAX_VIDEO_FRAMES = 18000
MAX_IMAGE_BYTES = 8 * 1024 * 1024

st.set_page_config(page_title="Gym Form & Rep Tracker", page_icon="🏋️", layout="wide")

@st.cache_resource(show_spinner="Loading the face model…")
def get_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    import threading
    app._gym_lock = threading.Lock()
    return app


def get_gallery():
    return st.session_state.setdefault("gallery", {})


def enroll_photo(name, image_bytes):
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Photo exceeds the 8 MB limit.")
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("This file is not a readable image.")
    h, w = image.shape[:2]
    if w < 96 or h < 96 or w*h > 20_000_000:
        raise ValueError("Photo dimensions must be at least 96×96 and no more than 20 megapixels.")
    face_app = get_face_app()
    with face_app._gym_lock:
        faces = face_app.get(image)
    if not faces:
        raise ValueError("No face detected. Try a clear, front-facing photo with one person.")
    face = max(faces, key=lambda item: float(item.det_score))
    if float(face.det_score) < .6:
        raise ValueError("Face confidence is too low. Try a clearer photo.")
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if not norm:
        raise ValueError("Could not create a face embedding from that photo.")
    emb = emb / norm
    gallery = get_gallery()
    uid = f"user_{len(gallery)+1:03d}"
    gallery[uid] = {"name": name.strip()[:60], "embedding": emb}
    return uid


class LiveProcessor(VideoProcessorBase):
    def __init__(self, exercise, gallery, state, face_app, max_people):
        self.engine = GymPoseEngine(exercise, gallery, max_people=max_people,
                                    face_app=face_app, state=state, max_frame_size=480)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(self.engine.process(image), format="bgr24")

    def on_ended(self):
        # Keep the session processor warm so a camera stop/restart does not
        # reload MediaPipe inside streamlit-webrtc's short offer timeout.
        pass

    def close(self):
        self.engine.close()


def render_enrollment():
    st.subheader("Enroll users for this browser session")
    st.caption("Profiles and face embeddings stay in server memory for this session. They are not written to disk. Each person must consent to face enrollment.")
    consent = st.checkbox("I have permission from this person to enroll and process their face image.")
    with st.form("enroll_form", clear_on_submit=True):
        name = st.text_input("Name or nickname", max_chars=60)
        photo = st.file_uploader("One clear face photo", type=["jpg", "jpeg", "png"], key="enroll_photo")
        submitted = st.form_submit_button("Enroll user", disabled=not consent)
    if submitted:
        if not name.strip() or photo is None:
            st.error("Enter a name and select a photo.")
        else:
            try:
                uid = enroll_photo(name, photo.getvalue())
                st.success(f"Enrolled {name.strip()} as {uid}. Stop and restart live detection to apply roster changes.")
            except Exception as exc:
                st.error(str(exc))
    gallery = get_gallery()
    if gallery:
        st.write("Current session roster")
        for uid, profile in gallery.items():
            st.write(f"**{uid}** — {profile['name']}")
        if st.button("Clear session roster", type="secondary"):
            st.session_state["gallery"] = {}
            st.rerun()


@st.fragment(run_every=1)
def render_live_stats(state, exercise):
    snapshot = state.snapshot()
    col_people, col_fps = st.columns(2)
    col_people.metric("People detected", snapshot["detected"])
    col_fps.metric("Processing FPS", f"{snapshot['fps']:.1f}")
    if not snapshot["people"]:
        st.caption("No tracked people yet. Start the camera and step into view.")
        return
    for uid, name in snapshot["people"].items():
        reps = snapshot["counts"].get((uid, exercise), 0)
        with st.container(border=True):
            st.markdown(f"**{name}** · `{uid}` · **{exercise}: {reps} reps**")
            angles = snapshot["angles"].get(uid, {})
            if angles:
                st.caption("Joint angles: " + " · ".join(
                    f"{key.replace('_', ' ')} {value:.0f}°"
                    for key, value in angles.items()
                ))


def rtc_configuration():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_url = os.environ.get("TURN_SERVER_URL", "").strip()
    turn_user = os.environ.get("TURN_USERNAME", "").strip()
    turn_password = os.environ.get("TURN_PASSWORD", "").strip()
    if turn_url and turn_user and turn_password:
        servers.append({"urls": [turn_url], "username": turn_user, "credential": turn_password})
    return {"iceServers": servers}


def render_live(exercise, max_people):
    st.subheader("Live webcam")
    st.caption("Allow camera access. Video frames are sent to this app server for inference; they are not recorded. Keep the browser tab open while exercising.")
    state = st.session_state.setdefault("live_state", SharedState())
    gallery_snapshot = dict(get_gallery())
    face_model = get_face_app() if gallery_snapshot else None
    roster_key = tuple((uid, profile.get("name", uid))
                       for uid, profile in gallery_snapshot.items())
    processor_key = (exercise, roster_key, max_people)
    processor = st.session_state.get("live_processor")
    if processor is None or st.session_state.get("live_processor_key") != processor_key:
        if processor is not None:
            processor.close()
        with st.spinner("Initializing pose detection for this session…"):
            processor = LiveProcessor(exercise, gallery_snapshot, state, face_model, max_people)
        st.session_state["live_processor"] = processor
        st.session_state["live_processor_key"] = processor_key
    ctx = webrtc_streamer(
        key="gym-live-camera", mode=WebRtcMode.SENDRECV,
        # The expensive model is already initialized before WebRTC negotiation.
        video_processor_factory=lambda: processor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 480},
                "height": {"ideal": 360},
                "frameRate": {"ideal": 15, "max": 15},
            },
            "audio": False,
        },
        # Keep processing synchronous: the pose task is per-session and not
        # shared with a background processing thread. Limit capture FPS and
        # resolution above to avoid building a queue of stale frames.
        rtc_configuration=rtc_configuration(), async_processing=False,
    )
    render_live_stats(state, exercise)
    if not ctx.state.playing:
        st.info("Click START to begin live detection. Stop the stream before changing the enrolled roster.")


def safe_video_bytes(upload):
    raw = upload.getvalue()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a non-empty video smaller than 150 MB.")
    if not (upload.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))):
        raise ValueError("Supported file extensions: MP4, MOV, AVI, MKV, WEBM.")
    return raw


def analyze_video(upload, exercise, progress):
    raw = safe_video_bytes(upload)
    import tempfile
    temp_path = None
    cap = None
    writer = None
    engine = None
    try:
        # OpenCV accepts paths more reliably than an in-memory video stream; use a private random temp file and always delete it.
        suffix = os.path.splitext(upload.name)[1].lower()
        with tempfile.NamedTemporaryFile(prefix="gym_upload_", suffix=suffix, delete=False) as f:
            temp_path = f.name
            f.write(raw)
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("The video could not be decoded. Try an MP4 encoded with H.264.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 1 <= fps <= 120:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width < 2 or height < 2 or total <= 0:
            raise ValueError("The video has invalid dimensions or no decodable frames.")
        if total > MAX_VIDEO_FRAMES or total / fps > 600:
            raise ValueError("This video is longer than the 10-minute processing limit. Trim it and upload again.")
        if width*height > 1920*1080:
            raise ValueError("Video resolution exceeds 1080p. Resize it before uploading.")
        # Inference is capped at 960 px on the long edge; match the writer dimensions.
        scale = min(1.0, 960 / max(width, height))
        output_size = (max(2, int(width*scale)), max(2, int(height*scale)))
        out_path = temp_path + ".mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
        if not writer.isOpened():
            raise RuntimeError("Video output encoder is unavailable on this host.")
        state = SharedState()
        engine = GymPoseEngine(exercise, dict(get_gallery()), max_people=5, face_app=get_face_app(), state=state)
        processed = 0
        while processed < MAX_VIDEO_FRAMES:
            ok, frame = cap.read()
            if not ok: break
            writer.write(engine.process(frame))
            processed += 1
            if processed % 10 == 0:
                progress.progress(min(processed/max(total, 1), 1.0), text=f"Processed {processed}/{total} frames")
        writer.release(); writer = None
        cap.release(); cap = None
        counts = dict(engine.counts)
        people = {uid: profile["name"] for uid, profile in get_gallery().items()}
        for key, _ in counts.items():
            people.setdefault(key[0], key[0])
        if os.path.getsize(out_path) > MAX_OUTPUT_BYTES:
            raise ValueError("Annotated output exceeds the 250 MB download limit. Use a shorter or lower-resolution clip.")
        with open(out_path, "rb") as f:
            output = f.read()
        return output, counts, people, processed
    finally:
        if engine: engine.close()
        if cap: cap.release()
        if writer: writer.release()
        for candidate in (temp_path, temp_path + ".mp4" if temp_path else None):
            if candidate:
                try: os.remove(candidate)
                except FileNotFoundError: pass


def render_video(exercise):
    st.subheader("Analyze a workout video")
    st.caption("Upload is limited to 100 MB, 10 minutes, and 1080p. The original and annotated output are temporary; download the output before leaving this page.")
    upload = st.file_uploader("Workout video", type=["mp4", "mov", "avi", "mkv", "webm"], key="workout_video")
    if upload and st.button("Analyze video", type="primary"):
        progress = st.progress(0, text="Starting video analysis…")
        try:
            result, counts, people, frames = analyze_video(upload, exercise, progress)
            progress.progress(1.0, text=f"Finished {frames} frames")
            st.success("Analysis complete.")
            st.video(result)
            for (uid, ex), reps in sorted(counts.items()):
                st.write(f"**{people.get(uid, uid)} — {ex}:** {reps} reps")
            st.download_button("Download annotated video", result, file_name="gym_analysis.mp4", mime="video/mp4")
            import csv
            csv_buffer = io.StringIO(newline="")
            csv_writer = csv.writer(csv_buffer)
            csv_writer.writerow(["participant", "exercise", "repetitions"])
            for (uid, ex), reps in sorted(counts.items()):
                participant = str(people.get(uid, uid))
                if participant.lstrip().startswith(("=", "+", "-", "@")):
                    participant = "'" + participant
                csv_writer.writerow([participant, ex, reps])
            csv_rows = csv_buffer.getvalue()
            st.download_button("Download CSV summary", csv_rows, file_name="gym_summary.csv", mime="text/csv")
        except Exception as exc:
            progress.empty()
            st.error(f"Could not analyze video: {exc}")


st.title("🏋️ Gym Form & Rep Tracker")
st.caption("Live multi-person pose estimation, joint angles, rep counts, and uploaded-video review.")
with st.sidebar:
    st.header("Workout")
    exercise = st.selectbox("Exercise to count", EXERCISES)
    st.caption("Rep counts use starter angle heuristics. Test and tune the thresholds for your camera and movement style.")
    st.divider()
    st.write("Identity matching")
    st.caption("Visible enrolled faces can reconnect session IDs. Motion tracking bridges brief gaps. Roster and counts are session-only.")
    max_people = st.slider("Maximum people to track", min_value=1, max_value=5, value=2)
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()

section = st.radio("Page", ["Users", "Live camera", "Upload video"],
                   horizontal=True, label_visibility="collapsed")
if section == "Users":
    render_enrollment()
elif section == "Live camera":
    render_live(exercise, max_people)
else:
    render_video(exercise)

st.divider()
st.caption("Personal prototype. InsightFace buffalo_l pretrained weights are restricted to non-commercial research use. A separate suitable license is required before commercial deployment. Use face enrollment only with consent.")
