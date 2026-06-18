import cv2
import time
import numpy as np
import mediapipe as mp
import os
import json
from collections import deque
from insightface.app import FaceAnalysis

# Modules Import
from dms_config import *
from dms_utils import *
from dms_database import DMSDatabase
from dms_alerts import play_alert_sound  # Updated Alert Module

mp_face_mesh = mp.solutions.face_mesh


class DriverEngine:
    def __init__(self):
        print("🔄 System Initialization...")
        self.db = DMSDatabase()

        # InsightFace
        self.face_recognizer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_recognizer.prepare(ctx_id=0, det_size=(640, 640))

        # MediaPipe
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

        self.known_embeddings = []
        self.known_names = []
        self.load_reference_images()

        self.current_driver = None
        self.last_seen_time = 0
        self.frames_eye_closed = 0
        self.frames_yawn = 0
        self.is_recording = False
        self.video_writer = None
        self.last_json_update = 0

        self.sleep_start_time = None
        self.yawn_start_time = None

        self.ear_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.mar_buffer = deque(maxlen=SMOOTHING_WINDOW)

        if not os.path.exists(RECORDING_PATH): os.makedirs(RECORDING_PATH)

    def load_reference_images(self):
        print("📂 Loading Drivers...")
        if not os.path.exists(DB_PATH): return
        for driver_name in os.listdir(DB_PATH):
            path = os.path.join(DB_PATH, driver_name)
            if not os.path.isdir(path): continue
            for img_file in os.listdir(path):
                img = cv2.imread(os.path.join(path, img_file))
                if img is None: continue
                faces = self.face_recognizer.get(img)
                if faces:
                    self.known_embeddings.append(faces[0].normed_embedding)
                    self.known_names.append(driver_name)
        print(f"✅ Ready! {len(self.known_names)} faces loaded.")

    def check_identity(self, face_embedding):
        highest_score = 0
        best_match = "Unknown"
        if not self.known_embeddings: return "Unknown", 0
        for idx, saved_emb in enumerate(self.known_embeddings):
            score = np.dot(face_embedding, saved_emb)
            if score > highest_score:
                highest_score = score
                best_match = self.known_names[idx]
        return (best_match, highest_score) if highest_score > 0.50 else ("Unknown", highest_score)

    def update_status_file(self, driver_name, status_msg="Awake", force=False):
        if not force and (time.time() - self.last_json_update < 0.2): return
        data = {
            "driver_name": driver_name if driver_name else "---",
            "is_logged_in": driver_name is not None,
            "drowsiness_status": status_msg,
            "is_recording": self.is_recording,
            "timestamp": time.strftime("%H:%M:%S")
        }
        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump(data, f)
            self.last_json_update = time.time()
        except:
            pass

    def start_recording(self, frame, driver_name):
        if not ENABLE_RECORDING: return
        h, w, _ = frame.shape
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"{RECORDING_PATH}{driver_name}_{ts}.avi"
        self.video_writer = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))
        self.is_recording = True
        print(f"🔴 RECORDING STARTED: {fn}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("⚪ RECORDING SAVED")
        self.is_recording = False

    def initialize_camera(self):
        if USE_USB_CAMERA:
            print("🔌 Connecting USB...")
            for index in [0, 1, 2]:
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret: return cap
                    cap.release()
        print(f"📡 Trying IP: {DROIDCAM_IP_URL}")
        return cv2.VideoCapture(DROIDCAM_IP_URL)

    def run(self):
        cap = self.initialize_camera()
        if not cap or not cap.isOpened():
            print("❌ CAMERA ERROR.")
            return

        print("⚡ ENGINE STABILIZED...")
        self.update_status_file(None, "Waiting", force=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1);
                cap = self.initialize_camera();
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            status_msg = "Awake"
            face_detected_now = False

            if results.multi_face_landmarks:
                face_detected_now = True
                lm = results.multi_face_landmarks[0].landmark
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in lm]

                # --- A. DROWSINESS CHECK ---
                current_ear = (calculate_EAR(landmarks, RIGHT_EYE_LM) + calculate_EAR(landmarks, LEFT_EYE_LM)) / 2.0
                current_mar = calculate_MAR(landmarks)

                self.ear_buffer.append(current_ear)
                self.mar_buffer.append(current_mar)

                if len(self.ear_buffer) == SMOOTHING_WINDOW:
                    avg_ear = sum(self.ear_buffer) / len(self.ear_buffer)
                    avg_mar = sum(self.mar_buffer) / len(self.mar_buffer)
                else:
                    avg_ear = current_ear
                    avg_mar = current_mar

                # Sleep Logic
                if avg_ear < EAR_THRESHOLD:
                    if self.sleep_start_time is None:
                        self.sleep_start_time = time.time()
                    elif time.time() - self.sleep_start_time > 2.0:  # 2 Sec threshold
                        status_msg = "SLEEPING!"
                        play_alert_sound("FATIGUE")  # Beep Sound
                        self.db.log_event(self.current_driver, "ALERT: SLEEPING")
                        cv2.putText(frame, "SLEEPING!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    self.sleep_start_time = None

                # Yawn Logic
                if avg_mar > MAR_THRESHOLD:
                    if self.yawn_start_time is None:
                        self.yawn_start_time = time.time()
                    elif time.time() - self.yawn_start_time > 1.5:
                        status_msg = "YAWNING"
                        cv2.putText(frame, "YAWNING", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                else:
                    self.yawn_start_time = None

                # --- B. IDENTITY CHECK ---
                if self.current_driver is None or (status_msg == "Awake" and time.time() % 1.0 < 0.2):
                    faces_rec = self.face_recognizer.get(frame)
                    if faces_rec:
                        name, score = self.check_identity(faces_rec[0].normed_embedding)
                        bbox = faces_rec[0].bbox.astype(int)

                        if self.current_driver is None:
                            if name != "Unknown":
                                # === LOGIN DETECTED ===
                                self.current_driver = name
                                print(f"👋 LOGIN: {name}")

                                # 🔊 BOLNE WALA CODE (NEW)
                                play_alert_sound("LOGIN", name=name)

                                self.db.log_event(name, "LOGIN")
                                self.start_recording(frame, name)
                        else:
                            if name != self.current_driver:
                                status_msg = "WRONG DRIVER"
                                # Wrong driver par bhi bolega
                                play_alert_sound("UNAUTHORIZED")
                                cv2.putText(frame, "WRONG DRIVER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                            3)

                        color = (0, 255, 0) if name == self.current_driver else (0, 0, 255)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        self.last_seen_time = time.time()

            # --- C. LOGOUT ---
            if self.current_driver and not face_detected_now:
                if time.time() - self.last_seen_time > LOGOUT_TIMER:
                    self.db.log_event(self.current_driver, "LOGOUT")
                    self.stop_recording()
                    self.current_driver = None
                    status_msg = "Logged Out"
                    self.ear_buffer.clear()
                    self.mar_buffer.clear()
                    self.sleep_start_time = None

            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

            self.update_status_file(self.current_driver, status_msg)
            cv2.imshow("Driver Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        if self.video_writer: self.video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DriverEngine().run()
