<div align="center">
# 🏫 Smart Hall
### AI-Powered Classroom Monitoring System

**Smart Hall** is an AI-powered classroom system that monitors student focus in real-time by merging IoT sensors (temperature, light, noise, motion) with Computer Vision. It classifies the environment and detects student distractions, sending live alerts to educators through a web dashboard.

---

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-ESP32--S3-00979D?style=flat&logo=arduino&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-black?style=flat&logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-CV-blue?style=flat)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-purple?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**Team:** Neurolytics &nbsp;|&nbsp; **Institution:** IAU — Imam Abdulrahman Bin Faisal University &nbsp;|&nbsp; **Event:** College Hackathon · Saudi Vision 2030

</div>

---

## 📖 Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Software Prerequisites](#3-software-prerequisites)
4. [Project File Structure](#4-project-file-structure)
5. [Step-by-Step: Installation & Running](#5-step-by-step-installation--running)
6. [Dashboard Guide](#6-dashboard-guide)
7. [Machine Learning Model](#7-machine-learning-model)
8. [Computer Vision Module](#8-computer-vision-module)
9. [Troubleshooting](#9-troubleshooting)
10. [Quick-Start Checklist](#10-quick-start-checklist)

---

## 1. System Architecture

The project is structured into three independent layers that communicate at runtime:

| Layer | Module | Description |
|-------|--------|-------------|
| **Hardware** | ESP32-S3 + sensors | Reads physical environment, streams JSON over USB serial |
| **ML Backend** | `Neurolytics-ML.py` | Classifies environment, logs data, hosts `/api/state` on `:5050` |
| **CV Backend** | `CV_Focus_Monitor.py` | Detects student focus via webcam, also hosts `/api/state` on `:5050` |
| **Dashboard** | `index.html` | Browser-based live dashboard, polls API, renders charts and alerts |

> ⚠️ **Important:** The ML module and the CV module are **INDEPENDENT**. Run **only one** at a time. Both serve the same API endpoint (`:5050/api/state`). The web dashboard works with whichever backend is currently active.

---

## 2. Hardware Requirements

| Component | Model / Type | Role |
|-----------|-------------|------|
| **Microcontroller** | ESP32-S3 | Reads all sensors, sends JSON over USB serial at 115200 baud |
| **Temperature & Humidity** | DHT11 (pin 11) | Measures classroom temperature (°C) and relative humidity (%) |
| **PIR Motion Sensor** | Digital (pin 21) | Detects physical presence / motion in the room |
| **Light Sensor** | LDR Analog (pin 10) | Measures ambient light level (ADC 0–4095) |
| **Microphone / Sound** | Analog mic (pin 12) | Samples noise level 10× per reading, returns average ADC value |
| **Laptop / PC Camera** | Built-in or USB webcam | Used exclusively by the CV module for face and gaze tracking |

---

## 3. Software Prerequisites

### 3.1 Required Software

| Tool | Version | Download |
|------|---------|----------|
| **Python** | 3.9 or later | https://www.python.org/downloads |
| **Arduino IDE** | 2.x | https://www.arduino.cc/en/software |
| **VS Code** *(optional)* | Any | https://code.visualstudio.com |
| **Web Browser** | Chrome / Edge | Opens the local dashboard — no server needed |

### 3.2 Python Libraries

**Common libraries (both modules):**
```bash
pip install flask flask-cors numpy scikit-learn joblib pyserial
```

**CV module only (additional):**
```bash
pip install opencv-python mediapipe ultralytics pandas
```

**Or install everything at once:**
```bash
pip install flask flask-cors numpy scikit-learn joblib pyserial
pip install opencv-python mediapipe ultralytics pandas
```

### 3.3 Arduino Board Setup

Before uploading the firmware, configure Arduino IDE for the ESP32-S3:

1. Open Arduino IDE → **File > Preferences**
2. In **"Additional Boards Manager URLs"** add:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools > Board > Boards Manager**, search `esp32`, and install **"esp32 by Espressif Systems"**
4. Go to **Tools > Manage Libraries**, search `DHT sensor library` by Adafruit, and install it
5. Connect the ESP32-S3 via USB → select **Tools > Board > ESP32S3 Dev Module** and choose the correct COM port

---

## 4. Project File Structure

```
SmartHall/
├── Neurolytics-ML.py          ←  ML Environmental Monitor
├── CV_Focus_Monitor.py        ←  Computer Vision Student Monitor
├── ESP32_Firmware.ino         ←  Arduino sketch for ESP32-S3
├── index.html                 ←  Web Dashboard (open in browser)
├── face_landmarker.task       ←  MediaPipe model file (CV module)
├── yolov8n.pt                 ←  YOLOv8 model file (CV module)
└── neurolytics_data/          ←  Auto-created: CSV logs + ML model
    ├── env_model.pkl              ←  Trained RandomForest classifier
    ├── env_YYYYMMDD_HHMMSS.csv    ←  Session data log
    └── env_YYYYMMDD_summary.json  ←  Session summary
```

> 📥 **Required Model Files (CV Module)**
> - `face_landmarker.task` — Download from [MediaPipe Model Zoo](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
> - `yolov8n.pt` — Auto-downloaded by Ultralytics on first run, or manually from [here](https://github.com/ultralytics/assets/releases)
>
> Place both files in the same directory as `CV_Focus_Monitor.py`.

---

## 5. Step-by-Step: Installation & Running

### Step 1 — Upload Firmware to ESP32-S3

1. Open `ESP32_Firmware.ino` in Arduino IDE
2. Verify sensor pin definitions match your wiring (`DHT_PIN=11`, `PIR=21`, `LDR=10`, `MIC=12`)
3. Select the correct board and COM port under the Tools menu
4. Click **Upload** — the LED will blink during upload
5. Open Serial Monitor (115200 baud) to confirm JSON output:
   ```json
   {"temp":22.5,"noise":2300,"light":1200,"motion":1,"humidity":60.0}
   ```
6. **Close the Serial Monitor** before running the Python script (they share the COM port)

---

### Step 2A — Run the ML Environmental Monitor

*Use this module to monitor classroom environmental conditions using IoT sensors.*

**With ESP32-S3 hardware connected:**
```bash
# Windows
python Neurolytics-ML.py --port COM6

# Linux / macOS
python Neurolytics-ML.py --port /dev/ttyUSB0

# Custom polling rate (e.g. 2 readings/second)
python Neurolytics-ML.py --port COM6 --hz 2
```

**Demo mode — no hardware needed:**
```bash
python Neurolytics-ML.py --no-arduino
```

**Available flags:**

| Flag | Effect |
|------|--------|
| `--retrain` | Force retrain the ML model and exit |
| `--smoke-test` | Run built-in accuracy self-test and exit |
| `--no-save` | Disable CSV and JSON session logging |
| `--api-port PORT` | Override the default Flask API port (default: 5050) |

---

### Step 2B — Run the CV Focus Monitor

*Use this module for real-time per-student engagement detection via webcam.*

1. Ensure `face_landmarker.task` and `yolov8n.pt` are in the same folder as `CV_Focus_Monitor.py`
2. Run the script:
   ```bash
   python CV_Focus_Monitor.py
   ```
3. A camera window opens — look naturally at the camera during calibration (~90 frames / ~3 seconds)
4. After calibration, the system detects faces and labels each student as **FOCUSED** or **NOT FOCUSED**
5. Press `ESC` to stop the session

---

### Step 3 — Open the Web Dashboard

1. Make sure either the ML or CV Python module is running (API active on port 5050)
2. Open `index.html` in your browser (double-click or drag into Chrome/Edge)
3. The green banner confirms: *"Connected to Neurolytics API — live sensor data active"*
4. The dashboard auto-refreshes every 800 ms — no additional server required

> 💡 **CORS Note:** The Python backend adds `Access-Control-Allow-Origin: *` headers automatically. If the dashboard shows a connection error, verify the Python script is running and port 5050 is not blocked by a firewall.

---

## 6. Dashboard Guide

| Panel | What It Shows |
|-------|--------------|
| **Environment Banner** | Top colored bar: 🟢 Focused · 🟡 Half Focus · 🔴 Not Focused. Shows ML confidence %. |
| **Summary Bar** | Four stat cards: environment status, temperature, humidity, and light level |
| **Focus History Chart** | Sparkline of the last 60 environment focus readings |
| **IoT Sensor Cards** | Live values for Temperature, Humidity, Light, Noise, and Motion. Red warning badge when out of range. |
| **Alert Log** | Scrollable list of the last 60 alerts — yellow for IoT threshold breaches, red for classification alerts |

### Environment Thresholds

| Sensor | Min (Optimal) | Max (Optimal) | Unit |
|--------|:-------------:|:-------------:|------|
| Temperature | 20 | 24 | °C |
| Humidity | 57 | 63 | % relative humidity |
| Light Level | 360 | 1600 | lux (ADC 0–4095) |
| Noise Level | 2000 | 2500 | ADC value (0–4095) |
| Motion (PIR) | 1 (present) | — | Binary: 1 = detected, 0 = absent |

---

## 7. Machine Learning Model

The ML module uses a **Random Forest Classifier** trained on synthetic data generated from the domain thresholds.

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest Classifier (scikit-learn) |
| **Features (inputs)** | temperature, light_level, motion, humidity, noise_level |
| **Classes (outputs)** | `0` = Not Focused · `1` = Half Focus · `2` = Focused |
| **Trees / Estimators** | 300 |
| **Preprocessing** | StandardScaler (wrapped in sklearn Pipeline) |
| **Evaluation** | 5-fold cross-validation accuracy reported at training time |
| **Fallback** | Rule-based predictor if scikit-learn is unavailable |
| **Model file** | `neurolytics_data/env_model.pkl` (auto-created on first run) |

```bash
# Force retrain the model
python Neurolytics-ML.py --retrain

# Run the built-in accuracy smoke-test
python Neurolytics-ML.py --smoke-test
```

---

## 8. Computer Vision Module

The CV module detects engagement at the individual student level using facial landmarks and object detection.

### 8.1 Detection Conditions

| Condition | Alert Delay | Detection Method |
|-----------|:-----------:|-----------------|
| **Eyes Closed** | 7 sec | Eye-opening ratio via MediaPipe landmarks vs calibrated baseline |
| **Head Down** | 10 sec | Nose Y − inner eye Y ratio vs calibrated baseline + 0.03 offset |
| **Looking Away** | 7 sec | Nose X offset > 0.12 from center, or nose Y < 0.35 |
| **Face Hidden** | 4 sec | Less than 50% of landmarks within valid frame coordinates |
| **Eyes Covered** | 1 sec | Both eye apertures < 0.003 (nearly zero) |
| **Gaze Distracted** | 8 sec | Iris gaze ratio < 0.35 (left) or > 0.65 (right), head not turned |
| **Holding Object** | 5 sec | YOLOv8 detects phone, book, bottle, laptop, etc. in frame |

### 8.2 Calibration

On startup the CV module collects **90 frames** of the user looking naturally at the camera. This sets personalised baseline values for eye-open ratio and head position, improving accuracy for different face shapes and distances.

---

## 9. Troubleshooting

| Problem | Solution |
|---------|----------|
| Dashboard shows "Cannot reach API" | Ensure the Python script is running and port 5050 is not blocked by a firewall |
| "Access is denied" on COM port | Close Arduino IDE Serial Monitor — only one program can use the COM port at a time |
| DHT sensor returns NaN | Check DHT wiring. The backend retains the last valid value automatically |
| `scikit-learn` not found | Run `pip install scikit-learn joblib`. The module falls back to rule-based classification until installed |
| MediaPipe model file missing | Download `face_landmarker.task` from the MediaPipe Model Zoo and place it next to `CV_Focus_Monitor.py` |
| YOLOv8 model not found | Run the CV script once with internet access — Ultralytics auto-downloads `yolov8n.pt` |
| Camera not opening | Ensure no other app is using the webcam. The script uses `cv2.VideoCapture(0)` (index 0 = default camera) |
| Port not detected automatically | Pass it explicitly: `--port COM6` (Windows) or `--port /dev/ttyUSB0` (Linux/macOS) |

---

## 10. Quick-Start Checklist

| # | Module | Step |
|---|--------|------|
| 1 | Both | Install Python 3.9+ |
| 2 | Both | `pip install flask flask-cors numpy scikit-learn joblib pyserial` |
| 3 | CV | `pip install opencv-python mediapipe ultralytics pandas` |
| 4 | ML | Connect ESP32-S3 via USB and upload `ESP32_Firmware.ino` via Arduino IDE |
| 5 | CV | Place `face_landmarker.task` and `yolov8n.pt` next to `CV_Focus_Monitor.py` |
| 6 | ML | Close Arduino Serial Monitor, then run `python Neurolytics-ML.py --port COM6` |
| 7 | CV | Run `python CV_Focus_Monitor.py` |
| 8 | Both | Open `index.html` in Chrome or Edge |
| 9 | Both | Confirm the green "Connected" banner in the dashboard header ✅ |

---

<div align="center">

**Smart Hall · Neurolytics Team · IAU · Saudi Vision 2030**

*Built with Python · Flask · scikit-learn · MediaPipe · YOLOv8 · Arduino / ESP32-S3 · HTML5/CSS3/JS*

</div>
