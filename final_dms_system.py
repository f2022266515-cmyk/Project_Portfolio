"""
Driver Monitoring System — Orchestrator v7 (THE FINAL FIX)
===========================================================
- Fix: end_trip attribute error resolved.
- Fix: Dual Phone Detection (Custom + Standard YOLO ID 67).
- Fix: Seatbelt stats increment logic.
"""
import cv2
import time
import numpy as np
import mediapipe as mp
import os
import sys
import json
import threading  # Yeh add karein
import requests   # Yeh add karein
import traceback
from ultralytics import YOLO

try:
    from dms_config import *
    from dms_database import DMSDatabase
    from dms_alerts import play_alert_sound
    from insightface.app import FaceAnalysis
    from detectors import FatigueDetector, YawnDetector, ObjectDetector, Calibrator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

mp_face_mesh   = mp.solutions.face_mesh
RECORDING_PATH = "recordings/"
DB_PATH        = "datasets/faces/"

# UI Positions
UI_SLEEP_Y, UI_YAWN_Y, UI_PHONE_Y, UI_SMOKE_Y = 150, 210, 310, 370

def _draw_progress_bar(frame, x, y, width, progress_0to1, color, label):
    p = max(0.0, min(1.0, progress_0to1))
    cv2.rectangle(frame, (x, y), (x + width, y + 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + int(width * p), y + 20), color, -1)
    cv2.putText(frame, label, (x + width + 8, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

def _driver_box_from_landmarks(landmarks, frame_w, frame_h):
    if not landmarks: return None
    xs = [pt[0] for pt in landmarks]; ys = [pt[1] for pt in landmarks]
    fw, fh = max(xs)-min(xs), max(ys)-min(ys)
    return [max(0, min(xs)-int(fw*1.5)), max(0, min(ys)-int(fh*0.5)), min(frame_w, max(xs)+int(fw*1.5)), min(frame_h, max(ys)+int(fh*3.0))]

class DriverEngine:
    def __init__(self):
        print("🚗 DMS v7 — NABIA EDITION")
        self.db = DMSDatabase()
        self.face_recognizer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_recognizer.prepare(ctx_id=0, det_size=(640, 640))
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

        # Load Model
        self.model_yolo = YOLO(DETECTION_MODEL_PATH)

        self.fatigue = FatigueDetector(EAR_THRESHOLD, SLEEP_TIME_THRESHOLD, SMOOTHING_WINDOW)
        self.yawn = YawnDetector(MAR_THRESHOLD, YAWN_TIME_THRESHOLD, SMOOTHING_WINDOW)
        self.objects = ObjectDetector(DETECTION_MODEL_PATH, SEATBELT_MEMORY_SECONDS, OBJECT_CHECK_INTERVAL, SEATBELT_CHECK_INTERVAL, imgsz=YOLO_IMGSZ)
        self.calibrator = Calibrator(CALIBRATION_FRAMES, EAR_THRESHOLD, MAR_THRESHOLD)

        self.known_embeddings, self.known_names = [], []
        self.load_reference_images()
        self.current_driver, self.last_seen_time = None, 0
        self.last_status_write = 0
        self._write_status(None)
        os.makedirs(RECORDING_PATH, exist_ok=True)

        self.load_reference_images()
        self.current_driver, self.last_seen_time = None, 0
        self.last_status_write = 0
        self._write_status(None)

        self.my_unit_id = "DMS-UNIT-01"
        threading.Thread(target=self._send_heartbeat, daemon=True).start()

    def load_reference_images(self):
        if not os.path.exists(DB_PATH): return
        for name in os.listdir(DB_PATH):
            p = os.path.join(DB_PATH, name);
            if os.path.isdir(p):
                for img_f in os.listdir(p):
                    img = cv2.imread(os.path.join(p, img_f))
                    if img is not None:
                        f = self.face_recognizer.get(img)
                        if f: self.known_embeddings.append(f[0].normed_embedding); self.known_names.append(name)

    def start_trip(self, driver_name):
        self.current_driver = driver_name
        self.fatigue.reset(); self.yawn.reset(); self.objects.reset(time.time()); self.calibrator.reset()
        self.db.log_event(driver_name, "LOGIN")
        play_alert_sound("LOGIN", name=driver_name)

    def end_trip(self):
        if self.current_driver:
            print(f"⚪ TRIP ENDED: {self.current_driver}")
            self.current_driver = None

    def _write_status(self, driver_name, drowsiness="Awake", phone=False, smoking=False, seatbelt_on=True, ear=0.0, mar=0.0):
        now = time.time()
        if now - self.last_status_write < 0.3: return
        self.last_status_write = now
        data = {"driver_name": driver_name or "---", "is_logged_in": driver_name is not None, "drowsiness_status": drowsiness, "phone_alert": phone, "smoking_alert": smoking, "seatbelt_on": seatbelt_on, "ear": round(ear, 3), "mar": round(mar, 3), "timestamp": time.strftime("%H:%M:%S")}
        with open(STATUS_FILE, 'w') as f: json.dump(data, f, indent=2)

    def _send_heartbeat(self):
        url = "https://vigilieai.onrender.com/api/heartbeat"
        while True:
            try:
                response = requests.post(url, json={"unit_id": self.my_unit_id})
                print(f"📡 Heartbeat status: {response.status_code}") # Yeh line add karein
            except Exception as e:
                print(f"❌ Heartbeat error: {e}")
            time.sleep(30)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            while True:
                ret, frame = cap.read()
                if not ret: continue
                h, w, _ = frame.shape; rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); curr_t = time.time()
                face_detected, display_ear, display_mar = False, 0.0, 0.0

                mesh_res = self.face_mesh.process(rgb)
                if mesh_res.multi_face_landmarks:
                    face_detected, self.last_seen_time = True, curr_t
                    lm = mesh_res.multi_face_landmarks[0].landmark
                    landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in lm]

                    if self.current_driver is None:
                        fr = self.face_recognizer.get(frame)
                        if fr:
                            scores = [float(np.dot(fr[0].normed_embedding, e)) for e in self.known_embeddings]
                            if scores and max(scores) > 0.6: self.start_trip(self.known_names[np.argmax(scores)])

                    elif self.calibrator.is_active():
                        self.calibrator.add_sample(landmarks)
                        cv2.putText(frame, "CALIBRATING...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    elif self.current_driver and self.calibrator.is_done():
                        # Fatigue
                        fatigue_r = self.fatigue.detect(landmarks, w, h, curr_t)
                        display_ear = fatigue_r.avg_ear
                        if fatigue_r.is_sleeping:
                            cv2.rectangle(frame, (0, UI_SLEEP_Y-35), (400, UI_SLEEP_Y+15), (0,0,150), -1)
                            cv2.putText(frame, "SLEEPING!", (15, UI_SLEEP_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                            if fatigue_r.should_alert:
                                play_alert_sound("FATIGUE")
                                self.db.log_event(self.current_driver, "Sleep Alert", frame)

                        # Yawn
                        yawn_r = self.yawn.detect(landmarks, curr_t)
                        display_mar = yawn_r.avg_mar
                        if yawn_r.is_yawning:
                            cv2.rectangle(frame, (0, UI_YAWN_Y-35), (400, UI_YAWN_Y+15), (0,150,150), -1)
                            cv2.putText(frame, "YAWNING!", (15, UI_YAWN_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                            if yawn_r.should_alert:
                                play_alert_sound("YAWN")
                                self.db.log_event(self.current_driver, "Yawning Detected", frame)

                # Objects

                if self.current_driver and self.calibrator.is_done():
                    dbox = _driver_box_from_landmarks(landmarks, w, h) if face_detected else None
                    obj_r = self.objects.detect(frame, dbox, curr_t, landmarks if face_detected else None)

                    # Optimized Phone Detection
                    standard_yolo_phone = False
                    if frame is not None and frame.size > 0:
                        results = self.model_yolo(frame, conf=0.3, verbose=False)
                        for r in results:
                            for box in r.boxes:
                                if int(box.cls[0]) == 67:
                                    standard_yolo_phone = True
                                    break

                    # Alerts logic
                    if obj_r.phone or standard_yolo_phone:
                        cv2.rectangle(frame, (0, UI_PHONE_Y - 30), (450, UI_PHONE_Y + 15),
                                                (0, 165, 255), -1)
                        cv2.putText(frame, "PHONE USAGE!", (15, UI_PHONE_Y), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (255, 255, 255), 3)
                        if obj_r.should_alert_phone or standard_yolo_phone:
                            play_alert_sound("PHONE")
                            self.db.log_event(self.current_driver, "PHONE_USAGE", frame)

                    if obj_r.smoking:
                        cv2.rectangle(frame, (0, UI_SMOKE_Y-30), (450, UI_SMOKE_Y+15), (0, 0, 200), -1)
                        cv2.putText(frame, "SMOKING!", (15, UI_SMOKE_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        if obj_r.should_alert_smoking:
                            play_alert_sound("SMOKING")
                            self.db.log_event(self.current_driver, "SMOKING", frame)

                    if not obj_r.seatbelt_on and obj_r.should_alert_seatbelt:
                        cv2.rectangle(frame, (0, h-80), (350, h-30), (0, 0, 150), -1)
                        cv2.putText(frame, "NO SEATBELT!", (15, h-45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                        play_alert_sound("SEATBELT")
                        self.db.log_event(self.current_driver, "NO_SEATBELT", frame)

                    self._write_status(self.current_driver, phone=(obj_r.phone or standard_yolo_phone), smoking=obj_r.smoking, seatbelt_on=obj_r.seatbelt_on, ear=display_ear, mar=display_mar)

                if self.current_driver and not face_detected and (curr_t - self.last_seen_time > LOGOUT_TIMER):
                    self.end_trip()

                cv2.imshow("DMS v7 — Nabia Edition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.end_trip(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    DriverEngine().run()
