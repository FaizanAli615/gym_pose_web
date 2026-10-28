"""Gym pose and repetition engine. No media is written to disk."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

EXERCISES = ("Bicep curl", "Squat", "Push-up", "Shoulder press", "Lunge", "Jumping jack")
TRIPLES = {
    "left_elbow": (11, 13, 15), "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23), "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25), "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27), "right_knee": (24, 26, 28),
    "left_ankle": (25, 27, 31), "right_ankle": (26, 28, 32),
}


def calculate_angle(a, b, c):
    """Angle ABC, in degrees, without requiring NumPy."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    d1 = math.hypot(*v1)
    d2 = math.hypot(*v2)
    if d1 == 0 or d2 == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (d1*d2)))
    return math.degrees(math.acos(cosine))


def cosine_similarity(a, b):
    # Embeddings from InsightFace are NumPy arrays; `not a` raises an
    # ambiguous-truth-value error for arrays and silently prevented matching.
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return -1.0
    dot = sum(float(x)*float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x)*float(x) for x in a))
    nb = math.sqrt(sum(float(y)*float(y) for y in b))
    return dot / (na*nb) if na and nb else -1.0


@dataclass
class SharedState:
    """Thread-safe latest counters for the Streamlit live-video callback."""
    lock: Any = field(default_factory=threading.Lock)
    counts: dict = field(default_factory=dict)
    people: dict = field(default_factory=dict)
    angles: dict = field(default_factory=dict)
    fps: float = 0.0
    detected: int = 0

    def publish(self, counts, people, angles, fps, detected):
        with self.lock:
            self.counts = dict(counts)
            self.people = dict(people)
            self.angles = dict(angles)
            self.fps = float(fps)
            self.detected = int(detected)

    def snapshot(self):
        with self.lock:
            return {"counts": dict(self.counts), "people": dict(self.people),
                    "angles": dict(self.angles), "fps": self.fps,
                    "detected": self.detected}


class GymPoseEngine:
    """MediaPipe Tasks inference, face re-identification, simple motion IDs and rep counters."""
    def __init__(self, exercise: str, gallery: dict | None = None, max_people: int = 5,
                 face_app=None, state: SharedState | None = None,
                 max_frame_size: int = 960):
        import cv2
        import mediapipe as mp
        self.cv2, self.mp = cv2, mp
        self.exercise = exercise
        self.gallery = gallery or {}
        self.max_people = max(1, min(int(max_people), 8))
        self.face_app = face_app
        self.state = state
        self.max_frame_size = max(320, int(max_frame_size))
        self.face_executor = None
        self.face_future = None
        self.face_future_frame = -1
        self.face_future_people = []
        if self.face_app is not None and self.gallery:
            from concurrent.futures import ThreadPoolExecutor
            self.face_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="gym-face-scan"
            )
        self.frame_no = 0
        self.next_id = 1
        self.tracks = {}
        self.counts = {}
        self.exercise_states = {}
        # Keep face inference off the video callback thread and scan sparsely.
        # Motion tracking bridges the gaps between face scans.
        self.face_scan_period = 60
        self.last_time = None
        self.fps = 0.0
        self.connections = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
            (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
            (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),
            (23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32)]
        from pathlib import Path
        import urllib.request
        cache_dir = Path.home() / ".cache" / "gym_pose_web"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = cache_dir / "pose_landmarker_lite.task"
        if not self.model_path.exists():
            import tempfile
            url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                   "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
            with tempfile.NamedTemporaryFile(prefix="pose_model_", dir=cache_dir, delete=False) as tmp:
                download_path = Path(tmp.name)
            try:
                urllib.request.urlretrieve(url, download_path)
                download_path.replace(self.model_path)
            finally:
                if download_path.exists():
                    download_path.unlink()
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=self.max_people, min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)

    def close(self):
        if getattr(self, "landmarker", None):
            self.landmarker.close()
            self.landmarker = None
        if self.face_executor is not None:
            self.face_executor.shutdown(wait=False, cancel_futures=True)
            self.face_executor = None

    def _detect_faces(self, frame):
        lock = getattr(self.face_app, "_gym_lock", None)
        if lock:
            with lock:
                return self.face_app.get(frame)
        return self.face_app.get(frame)

    def _angles(self, landmarks):
        out = {}
        for name, (ia, ib, ic) in TRIPLES.items():
            points = (landmarks[ia], landmarks[ib], landmarks[ic])
            if min(float(getattr(p, "visibility", 1.0)) for p in points) < 0.35:
                continue
            out[name] = calculate_angle(*[(p.x, p.y) for p in points])
        return out

    def _face_ids(self, frame, poses, assigned):
        if self.face_executor is None:
            return

        # Collect completed face work without ever blocking camera frame delivery.
        if self.face_future is not None and self.face_future.done():
            try:
                faces = self.face_future.result()
                h, w = self.face_future_shape
                claimed = set()
                for face in sorted(faces, key=lambda f: float(f.det_score), reverse=True):
                    if float(face.det_score) < 0.60:
                        continue
                    emb = getattr(face, "normed_embedding", None)
                    scores = sorted(
                        ((uid, cosine_similarity(emb, profile.get("embedding")))
                         for uid, profile in self.gallery.items()
                         if uid not in claimed and profile.get("embedding") is not None),
                        key=lambda pair: pair[1], reverse=True,
                    )
                    if not scores or scores[0][1] < 0.48:
                        continue
                    if len(scores) > 1 and scores[0][1] - scores[1][1] < 0.08:
                        continue
                    uid = scores[0][0]
                    x1, y1, x2, y2 = [float(v) for v in face.bbox]
                    fw, fh = max(1., x2-x1), max(1., y2-y1)
                    candidates = []
                    for tid, nx, ny in self.face_future_people:
                        if (x1-.12*fw <= nx <= x2+.12*fw and
                                y1-.18*fh <= ny <= y2+.18*fh):
                            distance = ((nx-(x1+x2)/2)/fw)**2 + ((ny-(y1+y2)/2)/fh)**2
                            candidates.append((distance, tid))
                    if candidates:
                        _, tid = min(candidates)
                        if tid in self.tracks:
                            self.tracks[tid]["uid"] = uid
                            claimed.add(uid)
            except Exception:
                pass
            self.face_future = None

        if poses and self.frame_no % self.face_scan_period == 0 and self.face_future is None:
            self.face_future_frame = self.frame_no
            h, w = frame.shape[:2]
            self.face_future_shape = (h, w)
            self.face_future_people = [
                (tid, float(poses[index][0].x) * w, float(poses[index][0].y) * h)
                for index, (tid, _) in enumerate(assigned)
            ]
            self.face_future = self.face_executor.submit(self._detect_faces, frame.copy())

    def _assign_ids(self, poses, face_ids):
        centers=[]
        for pose in poses:
            xs=[p.x for p in pose if getattr(p,"visibility",1.0)>.25]
            ys=[p.y for p in pose if getattr(p,"visibility",1.0)>.25]
            centers.append((sum(xs)/len(xs), sum(ys)/len(ys)) if xs and ys else (pose[0].x,pose[0].y))
        available=set(self.tracks)
        assigned=[]
        for center, face_uid in zip(centers, face_ids):
            best_id=None; best_distance=.22
            for tid in available:
                old=self.tracks[tid]
                d=math.hypot(center[0]-old["center"][0],center[1]-old["center"][1])
                if d<best_distance:
                    best_distance,best_id=d,tid
            if best_id is None:
                best_id=self.next_id; self.next_id+=1
                self.tracks[best_id]={"center":center,"lost":0,"uid":None}
            else:
                available.remove(best_id)
            tr=self.tracks[best_id]
            tr["center"],tr["lost"]=center,0
            if face_uid:
                tr["uid"]=face_uid
            if tr["uid"] is None:
                tr["uid"]=f"guest-{best_id:03d}"
            assigned.append((best_id,tr["uid"]))
        used={tid for tid,_ in assigned}
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["lost"]+=1
                if self.tracks[tid]["lost"]>90:
                    del self.tracks[tid]
        return assigned

    def _count(self, person_key, landmarks, angles):
        count_key=(person_key,self.exercise)
        state=self.exercise_states.get(count_key,"ready")
        def avg(keys):
            vals=[angles[k] for k in keys if k in angles]
            return sum(vals)/len(vals) if vals else None
        if self.exercise=="Bicep curl":
            vals=[angles[k] for k in ("left_elbow","right_elbow") if k in angles]
            down=bool(vals) and min(vals)<50
            top=bool(vals) and max(vals)>155
        elif self.exercise=="Squat":
            knee=avg(("left_knee","right_knee")); down=knee is not None and knee<105; top=knee is not None and knee>155
        elif self.exercise=="Push-up":
            elbow=avg(("left_elbow","right_elbow")); down=elbow is not None and elbow<95; top=elbow is not None and elbow>155
        elif self.exercise=="Shoulder press":
            shoulder=avg(("left_shoulder","right_shoulder")); down=shoulder is not None and shoulder<45; top=shoulder is not None and shoulder>95
        elif self.exercise=="Lunge":
            knee=avg(("left_knee","right_knee")); down=knee is not None and knee<105; top=knee is not None and knee>155
        else:
            nose,lw,rw=landmarks[0],landmarks[15],landmarks[16]
            apart=abs(landmarks[27].x-landmarks[28].x)>max(abs(landmarks[23].x-landmarks[24].x)*1.8,.12)
            down=(lw.y>nose.y and rw.y>nose.y) or not apart
            top=lw.y<nose.y and rw.y<nose.y and apart
        if down:
            self.exercise_states[count_key]="down"
        elif top and state=="down":
            self.exercise_states[count_key]="up"
            self.counts[count_key]=self.counts.get(count_key,0)+1

    def process(self, frame):
        cv2,mp=self.cv2,self.mp
        h,w=frame.shape[:2]
        if max(h,w)>self.max_frame_size:
            scale=self.max_frame_size/max(h,w)
            frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
            h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        timestamp=max(self.frame_no*33,1)
        result=self.landmarker.detect_for_video(image,timestamp)
        poses=(result.pose_landmarks or [])[:self.max_people]
        assigned=self._assign_ids(poses,[None]*len(poses))
        self._face_ids(frame,poses,assigned)
        # A face result can arrive after its source frame. Apply the recognized
        # name to the motion track it matched, which remains stable meanwhile.
        assigned=[(tid,self.tracks[tid]["uid"]) for tid,_ in assigned if tid in self.tracks]
        latest_angles={}
        for pose,(_,uid) in zip(poses,assigned):
            angles=self._angles(pose)
            self._count(uid,pose,angles)
            latest_angles[uid]=angles
            color=(60,220,90)
            for a,b in self.connections:
                pa,pb=pose[a],pose[b]
                if min(float(getattr(pa,"visibility",1)),float(getattr(pb,"visibility",1)))<.3: continue
                cv2.line(frame,(int(pa.x*w),int(pa.y*h)),(int(pb.x*w),int(pb.y*h)),color,2,cv2.LINE_AA)
            nose=pose[0]
            label=f"{uid}  {self.exercise}: {self.counts.get((uid,self.exercise),0)}"
            cv2.putText(frame,label,(max(5,int(nose.x*w)),max(24,int(nose.y*h)-15)),
                        cv2.FONT_HERSHEY_SIMPLEX,.58,color,2,cv2.LINE_AA)
            for name,angle in angles.items():
                if name.endswith("elbow") or name.endswith("knee"):
                    idx=13 if "left_elbow"==name else 14 if "right_elbow"==name else 25 if "left_knee"==name else 26
                    p=pose[idx]
                    cv2.putText(frame,f"{angle:.0f}",(int(p.x*w),int(p.y*h)),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
        now=__import__("time").monotonic()
        if self.last_time is not None:
            instant=1/max(now-self.last_time,1e-6)
            self.fps=.85*self.fps+.15*instant if self.fps else instant
        self.last_time=now
        self.frame_no+=1
        names={uid:profile.get("name",uid) for uid,profile in self.gallery.items()}
        for _,uid in assigned:
            names.setdefault(uid,uid)
        if self.state:
            self.state.publish(self.counts,names,latest_angles,self.fps,len(poses))
        return frame
