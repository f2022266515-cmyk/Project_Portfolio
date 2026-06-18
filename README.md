# Driver Monitoring System (DMS) — Portfolio Showcase

This repository serves as a professional showcase of my contributions to the Driver Monitoring System (DMS) project, highlighting both the backend web integration and the core AI/ML pipeline.

### 🛠️ Project Tech Stack

* **AI/ML Core:** Python, YOLO (v8/v11), MediaPipe, InsightFace, OpenCV
* **Web & Backend:** Flask, WebSockets (Flask-SocketIO), Node.js, Render Deployment
* **Database:** MongoDB Cloud Atlas
* **Algorithms/Metrics:** Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) for yawn detection, Head Pose Estimation.

### 🚀 Key Features & Functionality

* **Drowsiness Detection:** Tracks the eye closure duration using facial landmarks. If the eyes remain closed beyond a threshold, an alert is triggered.
* **Distraction Tracking:** Monitors head movements and gaze direction to detect if the driver is looking away from the road.
* **Yawn Detection:** Tracks mouth movements to identify signs of fatigue.

### 💻 My Engineering Contributions

#### 1. Web Architecture & System Integration

I engineered the backend infrastructure and real-time communication protocols to transition the ML models from a localized script into a live, cloud-based application:

* **Backend & Cloud Deployment:** Engineered the Flask backend and successfully deployed the platform on Render for live webcam feed processing, collaborating closely with the team on the Tailwind CSS and MongoDB database integration.
* **Real-time Streaming:** Integrated WebSocket protocols (Flask-SocketIO) to seamlessly sync the client-side camera feed with the backend analytics engine with minimal latency.

#### 2. AI/ML Engineering Collaboration

Collaborating on the core AI/ML repository, I contributed to the system's analytical performance and decision-making logic:

* **Detection & Logic Optimization:** Worked on configuring algorithms like Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to refine drowsiness and yawn detection thresholds.
* **Alert Mechanisms:** Developed the conditional logic and mathematical evaluations (`dms_alerts.py`) that trigger system alarms based on live landmark tracking coordinates.

### 📱 Edge AI & Mobile Expansion (Current Focus)

* **Mobile Integration:** We have successfully initiated a local server architecture on mobile devices, mapping smartphone camera inputs to readiness for live driver monitoring.

### 📊 Repository Structure

* `app.py`: Contains the core web server, WebSocket real-time streaming, and Render deployment logic.
* `dms_core.py` / `dms_alerts.py`: Showcases the localized AI/ML tracking logic, algorithmic thresholds, and alert mechanisms used in the system pipeline.
* `requirements.txt`: Contains all necessary environment dependencies for seamless cloud deployment.
