# 🏫 Smart Hall System
### AI-Powered Classroom Monitoring & Analytics Platform

<p align="center">
  <img src="assets/neurolytics-showcase.jpg" width="900">
</p>

> Smart Hall System is an AI-powered classroom monitoring platform that combines **Computer Vision**, **Machine Learning**, **IoT Sensors**, and **Face Recognition** to improve educational environments through real-time analytics, automated attendance, and environmental monitoring.

---

## 🚀 Overview

Smart Hall System analyzes both **student behavior** and **classroom conditions** to provide instructors and administrators with actionable insights.

Developed under the **Neurolytics Project**, the system integrates:

- 👁️ Computer Vision for focus and distraction detection
- 🧠 Machine Learning for classroom environment assessment
- 📷 Face Recognition for automatic attendance
- 🌡️ IoT Sensors for environmental monitoring
- 📊 Real-time Dashboard for visualization and alerts

---

## 📸 Project Showcase

| Dashboard & Hardware | Project Poster | Live Deployment |
|----------------------|----------------|-----------------|
| ![](assets/dashboard.jpg) | ![](assets/poster.jpg) | ![](assets/deployment.jpg) |

---

## ✨ Key Features

### 🎯 Student Focus Monitoring
Detects:

- Eyes closed
- Looking away
- Head down
- Face hidden
- Covered eyes
- Distracted gaze
- Holding distracting objects (phone, book, laptop, etc.)

### ✅ Automatic Attendance
Using facial recognition:

- Identifies students from a preloaded dataset
- Marks attendance automatically
- Records timestamps
- Generates attendance logs

### 🌡️ Smart Environment Analysis

Monitors:

- Temperature
- Humidity
- Noise Levels
- Ambient Light
- Motion Detection

A Random Forest model evaluates whether classroom conditions are:

- Focused
- Half Focused
- Not Focused

### 📊 Real-Time Analytics Dashboard

Displays:

- Student focus status
- Attendance records
- Environmental sensor data
- Alert history
- Focus rate trends
- ML predictions

---

## 🏗️ System Architecture

```text
                +--------------------+
                |      ESP32-S3      |
                +---------+----------+
                          |
                          v
               +---------------------+
               |  ML Backend (5051)  |
               | Random Forest Model |
               +----------+----------+
                          |
                          |
                          v
+-------------+   +---------------------+
| Webcam Feed |-->| CV Backend (5050)   |
| MediaPipe   |   | YOLOv8 + Face Rec.  |
+-------------+   +----------+----------+
                            |
                            v
                 +----------------------+
                 |  Web Dashboard       |
                 | Real-Time Monitoring |
                 +----------------------+
```

---

## 🔧 Technologies Used

### Artificial Intelligence

- Python
- Scikit-Learn
- Random Forest Classifier
- MediaPipe
- YOLOv8
- face_recognition
- dlib

### IoT & Hardware

- ESP32-S3
- DHT11 Sensor
- PIR Motion Sensor
- LDR Light Sensor
- Analog Microphone
- USB Camera

### Backend

- Flask
- Flask-CORS
- REST APIs

### Frontend

- HTML5
- CSS3
- JavaScript

---

## 📂 Project Structure

```text
Neurolytics/
│
├── Neurolytics-Cv.py
├── Neurolytics-ML.py
├── ESP32_Firmware.ino
├── index.html
│
├── student_dataset/
│   ├── Ali_20210042/
│   ├── Sara_20210055/
│   └── Ahmed_20210099/
│
├── neurolytics_data/
│   ├── env_model.pkl
│   ├── session_logs.csv
│   └── summary.json
│
├── assets/
│   ├── dashboard.jpg
│   ├── poster.jpg
│   ├── deployment.jpg
│   └── neurolytics-showcase.jpg
│
└── README.md
```

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Neurolytics.git
cd Neurolytics
```

### Install Dependencies

```bash
pip install flask flask-cors numpy scikit-learn joblib pyserial
pip install opencv-python mediapipe ultralytics pandas
```

### Install Face Recognition

```bash
conda install -c conda-forge dlib
pip install face_recognition
```

---

## ▶️ Running the Project

### Start Environmental Monitoring

```bash
python Neurolytics-ML.py --port COM6
```

Or run without hardware:

```bash
python Neurolytics-ML.py --no-arduino
```

### Start Computer Vision Module

```bash
python Neurolytics-Cv.py
```

### Open Dashboard

Simply open:

```text
index.html
```

in Chrome or Edge.

---

## 📈 Machine Learning Model

| Property | Value |
|-----------|---------|
| Algorithm | Random Forest |
| Trees | 300 |
| Validation | 5-Fold Cross Validation |
| Inputs | Temperature, Humidity, Light, Motion, Noise |
| Outputs | Focused, Half Focused, Not Focused |

---

## 👁️ Computer Vision Pipeline

1. Face Detection
2. Face Tracking
3. Head Pose Estimation
4. Eye State Detection
5. Gaze Analysis
6. Object Detection (YOLOv8)
7. Face Recognition
8. Attendance Registration
9. Focus Scoring

---

## 📡 API Endpoints

### CV Backend

```http
GET /api/state
```

Returns:

```json
{
  "students": {},
  "attendance": [],
  "summary": {},
  "alerts": []
}
```

### ML Backend

```http
GET /api/state
```

Returns:

```json
{
  "sensors": {},
  "env_focus": {},
  "alerts": []
}
```

---

## 🎓 Academic Context

**Institution:** Imam Abdulrahman Bin Faisal University (IAU)

**Track:** Artificial Intelligence • IoT • Computer Vision

**Event:** College Hackathon — Saudi Vision 2030

---

## 🏆 Impact

Neurolytics helps educational institutions by:

- Improving classroom engagement
- Automating attendance processes
- Monitoring environmental quality
- Providing data-driven insights
- Supporting smarter learning environments

---

## 👥 Team Neurolytics

Developed as part of an AI & IoT innovation project at:

**Imam Abdulrahman Bin Faisal University (IAU)**

---

## 📜 License

This project is intended for educational and research purposes.

---

### Built with ❤️ using Python, AI, Computer Vision, IoT, and Machine Learning
