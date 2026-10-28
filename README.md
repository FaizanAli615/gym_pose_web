# Gym Form & Rep Tracker — Web App

A Streamlit frontend for the personal notebook prototype. It includes live webcam pose detection, multi-person overlays, joint angles, heuristic repetition counts, user face enrollment, and uploaded-video analysis.

## Run locally

Use Python 3.12 for the broadest wheel availability across MediaPipe, OpenCV, and InsightFace. Install system FFmpeg if you need video output support.

```powershell
cd gym_pose_web
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:APP_PASSWORD = "use-a-long-random-password"
streamlit run app.py
```

Open the local Streamlit URL, sign in, enroll each user with a consented face photo, then start the live camera or upload a clip. `streamlit-webrtc` requires a secure browser context (HTTPS) for remote camera access. Remote deployment needs HTTPS and STUN/TURN connectivity for WebRTC. A public Google STUN endpoint is configured by default; if the host network requires relay, set `TURN_SERVER_URL`, `TURN_USERNAME`, and `TURN_PASSWORD` as deployment secrets. See the [streamlit-webrtc deployment guidance](https://github.com/whitphx/streamlit-webrtc#serving-from-remote-host). Local access is simplest for a first run.

## Deploy with Docker

```sh
docker build -t gym-pose-web .
docker run --rm -p 8501:8501 -e APP_PASSWORD='use-a-long-random-password' gym-pose-web
```

For public deployment, terminate HTTPS at a trusted reverse proxy or hosting provider, set a strong `APP_PASSWORD` secret, and configure TURN if browsers cannot establish WebRTC directly. Do not commit credentials. This prototype uses a single shared password, not individual accounts or roles; it is not ready for a public multi-tenant gym service.

## Privacy and data handling

- Enrollment requires an explicit consent checkbox. Images are decoded in memory; face embeddings live only in the Streamlit browser session and are not written to a database.
- Browser camera frames are transmitted to the server for inference. Live camera frames are not recorded by the app.
- Uploaded videos are copied to random temporary files for OpenCV decoding. The app removes both original and generated temp files in a `finally` block. The task model and InsightFace pretrained model weights are cached separately in the user cache directory; the host needs outbound access the first time they are loaded. Annotated output is returned to the browser for download.
- User roster and repetition totals are session-only. Restarting the app/session clears them.
- Upload validation checks file extension, size, decoder open, dimensions, frame count, and duration. The file extension is not treated as proof of safe content; OpenCV must decode it.

## Model and limitations

The included pretrained InsightFace `buffalo_l` weights are licensed for non-commercial research use. This app is a personal prototype; obtain a suitable commercial license or replace the model before commercial deployment. Face matching is probabilistic and can fail or falsely match. The movement counters use starter thresholds and are not medical or coaching advice. Exercise counts should be validated against labeled clips before relying on them.

MediaPipe Tasks PoseLandmarker supports image/video/live-stream processing modes and multiple poses; this app uses video mode on decoded WebRTC/video frames. The interface is built on Streamlit WebRTC. See [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) and [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc).
