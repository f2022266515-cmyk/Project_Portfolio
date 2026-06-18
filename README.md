# Driver Monitoring System (DMS) — Portfolio Showcase

This repository serves as a professional showcase of my contributions to the Driver Monitoring System (DMS) project, highlighting both the Core AI/ML pipeline and the Web Integration.

---

## 🛠️ Project Tech Stack
* **AI/ML Core:** Python, YOLOv8/v11, OpenCV, MediaPipe
* **Web & Backend:** Flask, WebSockets (Flask-SocketIO), Node.js / Render Deployment
* **Database:** MongoDB Cloud Atlas
* **Algorithms/Metrics:** Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) for yawn detection, Head Pose Estimation.

---

## 🚀 Key Features & Functionality
* **Drowsiness Detection:** Tracks the eye closure duration using facial landmarks. If the eyes remain closed beyond a threshold, an alert is triggered.
* **Distraction Tracking:** Monitors head movements and gaze direction to detect if the driver is looking away from the road.
* **Yawn Detection:** Tracks mouth movements to identify signs of fatigue.

---

## 💻 My Engineering Contributions

### 1. AI/ML Engineering Collaboration 
Collaborating closely on the core AI/ML repository, I contributed heavily to the system's analytical performance:
* **Detection & Logic Optimization:** Worked on configuring algorithms like Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to refine drowsiness and yawn detection thresholds.
* **Alert Mechanisms:** Developed parts of the conditional logic (`dms_alerts.py`) that trigger system alarms based on live landmark tracking coordinates.

### 2. Web Development & System Integration
I engineered and managed the web-based infrastructure for this project to transition the ML model from a local script to a live cloud application:
* **Full-Stack Deployment:** Built the frontend/backend architecture and successfully deployed the platform on cloud hosting (Render) for live webcam feed processing.
* **Real-time Streaming:** Integrated WebSocket protocols to sync the client-side camera feed with the backend analytics engine seamlessly.


---

## 📱 Edge AI & Mobile Expansion (Current Focus)
* **Mobile Integration:** We have successfully initiated a local server architecture on mobile devices, mapping smartphone camera inputs to readiness for live driver monitoring.

---

## 📊 Repository Structure
* `app.py`: Contains the core web server integration and deployment logic.
* `dms_core.py`: Showcases the localized AI/ML tracking and alert logic used in the system pipeline.

---

theek hai?
