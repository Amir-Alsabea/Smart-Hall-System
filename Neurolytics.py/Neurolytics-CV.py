import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import face_recognition
import time
import threading
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, jsonify
from flask_cors import CORS
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ================================================================
#  CONFIG
# ================================================================
DATASET_DIR = r"C:\Users\zezom\Desktop\VScode\Neurolytics\student_dataset"

MODEL_PATH = r"C:\Users\zezom\Desktop\VScode\Neurolytics\face_landmarker.task"
YOLO_MODEL  = "yolov8n.pt"
LOG_FILE    = "focus_log.csv"
WINDOW_NAME = "Neurolytics - Attendance & Focus Monitor"
API_PORT    = 5050

ALERT_EYES_CLOSED_SEC  = 7
ALERT_HEAD_DOWN_SEC    = 10
ALERT_LOOK_AWAY_SEC    = 7
ALERT_FACE_HIDDEN_SEC  = 4
ALERT_OBJECT_SEC       = 5
ALERT_EYES_COVERED_SEC = 1
ALERT_GAZE_AWAY_SEC    = 8

UPDATE_HZ = 30

HUD_COLOR_FOCUS = (0, 220, 100)
HUD_COLOR_ALERT = (0, 80, 255)

CALIBRATION_FRAMES = 90

HANDHELD_OBJECTS = {
    "cell phone", "remote", "book", "bottle", "cup",
    "scissors", "knife", "fork", "spoon", "mouse",
    "keyboard", "laptop",
}

GAZE_LEFT_THRESH  = 0.35
GAZE_RIGHT_THRESH = 0.65

# ================================================================
#  LOAD REGISTERED STUDENTS (face_recognition)
# ================================================================
def load_registered_students():
    known_face_encodings = []
    known_face_names = []

    print("\n--- [LOG] Loading student photos from C:/student_dataset... ---")
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Folder '{DATASET_DIR}' not found!")
        return [], []

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if os.path.isdir(folder_path):
            print(f"Loading database photos for: {folder_name}")
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if len(encodings) > 0:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(folder_name)
                except Exception:
                    continue

    print(f"--- [LOG] Success! Loaded profiles for {len(set(known_face_names))} students. ---\n")
    return known_face_encodings, known_face_names

# ================================================================
#  SHARED STATE (written by camera thread, read by Flask thread)
# ================================================================
state_lock = threading.Lock()

shared = {
    "students": {},
    "summary":  {"total": 0, "focused": 0, "not_focused": 0, "focus_pct": 0},
    "alerts":   [],
    "attendance": []
}

# ================================================================
#  FLASK API
# ================================================================
flask_app = Flask(__name__)
CORS(flask_app)

@flask_app.route("/api/state")
def api_state():
    with state_lock:
        import copy
        return jsonify(copy.deepcopy(shared))

def run_flask():
    flask_app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)

# ================================================================
#  MEDIAPIPE + YOLO INIT
# ================================================================
BaseOptions           = mp.tasks.BaseOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
)

yolo_model = YOLO(YOLO_MODEL)
cap        = cv2.VideoCapture(0)

# ================================================================
#  CSV LOG
# ================================================================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "student_id", "status", "duration_sec"])

def log_event(student_id, stat, duration):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            student_id, stat, f"{duration:.2f}",
        ])

# ================================================================
#  PER-STUDENT STATE
# ================================================================
CONDITION_KEYS = [
    "eyes_closed", "head_down", "look_away", "no_face",
    "holding_obj", "eyes_covered", "gaze_away",
]

STATUS_LABELS = {
    "eyes_closed":  "Sleeping",
    "head_down":    "Head Down",
    "look_away":    "Looking Away",
    "no_face":      "Face Hidden",
    "holding_obj":  "Holding Object",
    "eyes_covered": "Eyes Covered",
    "gaze_away":    "Gaze Distracted",
}

def new_student_state():
    return {
        "timers":        {k: None  for k in CONDITION_KEYS},
        "active":        {k: False for k in CONDITION_KEYS},
        "focus_time":    0.0,
        "distract_time": 0.0,
        "focus_score":   100.0,
        "last_seen":     time.time(),
    }

students_state = {}

calibrated       = False
baseline_eye     = 0.022
baseline_head    = 0.06
calibration_data = []

# ================================================================
#  CONDITION ENGINE
# ================================================================
def update_condition(stu_id, key, detected, limit):
    st  = students_state[stu_id]
    now = time.time()

    if detected:
        if st["timers"][key] is None:
            st["timers"][key] = now
        elapsed = now - st["timers"][key]
        if elapsed >= limit and not st["active"][key]:
            st["active"][key] = True
            log_event(stu_id, key, elapsed)
            push_alert(stu_id, STATUS_LABELS[key])
    else:
        if st["active"][key]:
            log_event(stu_id, key + "_end",
                      now - st["timers"][key] if st["timers"][key] else 0)
        st["timers"][key] = None
        st["active"][key] = False

    return st["active"][key]

def push_alert(student_id, message):
    # Do not trigger alerts for unidentified profiles
    if "Unknown" in student_id:
        return
        
    entry = {
        "student": f"Student {student_id}",
        "message": message,
        "time":    datetime.now().strftime("%H:%M:%S"),
    }
    with state_lock:
        shared["alerts"].insert(0, entry)
        shared["alerts"] = shared["alerts"][:50]

# ================================================================
#  FACE HELPERS
# ================================================================
def is_eyes_closed(lms, thresh):
    return abs(lms[159].y - lms[145].y) < thresh

def head_down_ratio(lms):
    return lms[1].y - lms[33].y

def is_eyes_covered(lms):
    return (abs(lms[159].y - lms[145].y) < 0.003 and
            abs(lms[386].y - lms[374].y) < 0.003)

def get_gaze_ratio(lms):
    try:
        l_ratio = (lms[468].x - lms[33].x)  / (abs(lms[133].x - lms[33].x)  + 1e-6)
        r_ratio = (lms[473].x - lms[362].x) / (abs(lms[263].x - lms[362].x) + 1e-6)
        return (l_ratio + r_ratio) / 2.0
    except IndexError:
        return None

def estimate_head_angles(lms):
    yaw_deg   = round((lms[1].x - 0.5) * 120)
    pitch_deg = round((lms[1].y - 0.45) * 80)
    return f"{yaw_deg:+d}°", f"{pitch_deg:+d}°"

# ================================================================
#  CALIBRATION
# ================================================================
def calibrate_user(frame):
    global baseline_eye, baseline_head, calibrated
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
        int(time.time() * 1000),
    )
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        calibration_data.append((
            abs(lms[159].y - lms[145].y),
            lms[1].y - lms[33].y,
        ))
    if len(calibration_data) >= CALIBRATION_FRAMES and not calibrated:
        eye_vals, head_vals = zip(*calibration_data)
        baseline_eye  = np.mean(eye_vals)  * 0.70
        baseline_head = np.mean(head_vals) + 0.03
        calibrated = True

# ================================================================
#  MAIN CAMERA LOOP (Attendance + Focus combined)
# ================================================================
def camera_loop(known_encodings, known_names):
    global calibrated

    session_start       = time.time()
    already_present     = set()
    process_this_frame  = True

    fr_face_locations = []
    fr_face_encodings = []
    id_labels         = {}

    print("--- Live Camera is Starting ---")
    print("Press [ESC] to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Calibration phase ────────────────────────────────────
        if not calibrated:
            calibrate_user(frame)
            bar_len = int((len(calibration_data) / CALIBRATION_FRAMES) * w)
            cv2.rectangle(frame, (0, h - 25), (bar_len, h), (0, 220, 100), -1)
            cv2.putText(frame, "Calibrating... Look at camera naturally",
                        (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # ── Face Recognition (Attendance Tracking) ───────────────
        if process_this_frame:
            small_frame    = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small      = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            fr_face_locations = face_recognition.face_locations(rgb_small)
            fr_face_encodings = face_recognition.face_encodings(rgb_small, fr_face_locations)
            
            id_labels.clear()

            for idx, (face_enc, face_loc) in enumerate(zip(fr_face_encodings, fr_face_locations)):
                display_text = "Unknown Student"
                if len(known_encodings) > 0:
                    matches       = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.5)
                    face_distance = face_recognition.face_distance(known_encodings, face_enc)
                    if len(face_distance) > 0:
                        best_idx = np.argmin(face_distance)
                        if matches[best_idx]:
                            display_text = known_names[best_idx]

                id_labels[idx] = display_text 

                # Register attendance and render properties only for identified students
                if display_text != "Unknown Student":
                    if "_" in display_text:
                        student_real_name, student_id_str = display_text.split("_", 1)
                        label = f"{student_real_name} ({student_id_str})"
                    else:
                        student_real_name = display_text
                        student_id_str    = "No ID"
                        label             = student_real_name

                    if display_text not in already_present:
                        already_present.add(display_text)
                        print(f"[ATTENDANCE MARKED] Present: {student_real_name} (ID: {student_id_str})")
                        
                        with state_lock:
                            shared["attendance"].append({
                                "name": student_real_name,
                                "id": student_id_str,
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "status": "Present"
                            })

        process_this_frame = not process_this_frame

        # ── MediaPipe Face Detection ──────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(time.time() * 1000),
        )
        detected_faces = result.face_landmarks or []

        # ── YOLO (Handheld Object Detection) ──────────────────────
        yolo_res       = yolo_model(frame, verbose=False)
        holding_object = False
        holding_name   = ""

        for r in yolo_res:
            for box in r.boxes:
                name = yolo_model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if name in HANDHELD_OBJECTS:
                    holding_object = True
                    holding_name   = name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 1)
                    cv2.putText(frame, name.upper(), (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

        # ── Per-Face Focus Analytics Processing ────────────────────
        seen_this_frame = set()

        for face_idx, lms in enumerate(detected_faces):
            nose_x_px = int(lms[1].x * w)
            nose_y_px = int(lms[1].y * h)
            matched_stu_id = None

            for idx, loc in enumerate(fr_face_locations):
                top, right, bottom, left = loc
                top *= 4; right *= 4; bottom *= 4; left *= 4
                if (left - 20) <= nose_x_px <= (right + 20) and (top - 20) <= nose_y_px <= (bottom + 20):
                    matched_stu_id = id_labels.get(idx)
                    break

            stu_id = matched_stu_id if matched_stu_id and matched_stu_id != "Unknown Student" else f"Unknown_{face_idx + 1}"
            seen_this_frame.add(stu_id)

            if stu_id not in students_state:
                students_state[stu_id] = new_student_state()

            st = students_state[stu_id]
            st["last_seen"] = time.time()

            xs  = [int(p.x * w) for p in lms]
            ys  = [int(p.y * h) for p in lms]
            fx1 = max(min(xs) - 10, 0)
            fy1 = max(min(ys) - 10, 0)
            fx2 = min(max(xs) + 10, w)
            fy2 = min(max(ys) + 10, h)

            # Draw the facial mesh vertices
            for lm in lms:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 220, 100), -1)

            visible_ratio = sum(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in lms) / len(lms)
            nose_x        = lms[1].x
            nose_y        = lms[1].y
            head_turned   = abs(nose_x - 0.5) > 0.12 or nose_y < 0.35
            gaze_ratio    = get_gaze_ratio(lms)
            gaze_off      = (
                gaze_ratio is not None and
                (gaze_ratio < GAZE_LEFT_THRESH or gaze_ratio > GAZE_RIGHT_THRESH) and
                not head_turned
            )

            update_condition(stu_id, "eyes_closed",  is_eyes_closed(lms, baseline_eye),    ALERT_EYES_CLOSED_SEC)
            update_condition(stu_id, "head_down",    head_down_ratio(lms) > baseline_head, ALERT_HEAD_DOWN_SEC)
            update_condition(stu_id, "look_away",    head_turned,                          ALERT_LOOK_AWAY_SEC)
            update_condition(stu_id, "no_face",      visible_ratio < 0.5,                  ALERT_FACE_HIDDEN_SEC)
            update_condition(stu_id, "eyes_covered", is_eyes_covered(lms),                 ALERT_EYES_COVERED_SEC)
            update_condition(stu_id, "gaze_away",    gaze_off,                             ALERT_GAZE_AWAY_SEC)
            update_condition(stu_id, "holding_obj",  holding_object,                       ALERT_OBJECT_SEC)

            active_keys   = [k for k in CONDITION_KEYS if st["active"][k]]
            active_labels = []
            for k in active_keys:
                if k == "holding_obj":
                    active_labels.append(f"Holding: {holding_name or 'object'}")
                else:
                    active_labels.append(STATUS_LABELS[k])

            is_focused  = not active_labels
            status_text = "FOCUSED" if is_focused else "NOT FOCUSED"
            hud_color   = HUD_COLOR_FOCUS if is_focused else HUD_COLOR_ALERT

            if is_focused:
                st["focus_time"]    += 1 / UPDATE_HZ
            else:
                st["distract_time"] += 1 / UPDATE_HZ

            total = st["focus_time"] + st["distract_time"] + 1e-6
            st["focus_score"] = max(0.0, min(100.0, 100.0 * st["focus_time"] / total))

            # Handle visualization rendering exclusively for recognized profiles
            if "Unknown" not in stu_id:
                if "_" in stu_id:
                    real_name, id_str = stu_id.split("_", 1)
                    display_name = f"{real_name} ({id_str})"
                else:
                    display_name = stu_id

                cv2.rectangle(frame, (fx1, fy1 - 18), (fx2, fy1), (25, 25, 25), -1)
                cv2.putText(frame,
                            f"{display_name}: {status_text}  {st['focus_score']:.0f}%",
                            (fx1 + 4, fy1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), hud_color, 1)

                for i, lbl in enumerate(active_labels[:3]):
                    cv2.putText(frame, f"! {lbl}",
                                (fx1, fy2 + 14 + i * 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 255), 1)

                yaw_str, pitch_str = estimate_head_angles(lms)
                with state_lock:
                    shared["students"][stu_id] = {
                        "status":  "FOCUSED" if is_focused else "NOT FOCUSED",
                        "reasons": active_labels,
                        "yaw":     yaw_str,
                        "pitch":   pitch_str,
                        "score":   round(st["focus_score"]),
                    }

        # ── Handle Students Who Left the Frame ────────────────────
        gone = set(students_state.keys()) - seen_this_frame
        for stu_id in gone:
            update_condition(stu_id, "no_face", True, ALERT_FACE_HIDDEN_SEC)
            for k in ("eyes_closed", "head_down", "look_away", "eyes_covered", "gaze_away", "holding_obj"):
                update_condition(stu_id, k, False, 0)
            if (students_state[stu_id]["timers"]["no_face"] and
                    time.time() - students_state[stu_id]["last_seen"] > 10):
                del students_state[stu_id]
                with state_lock:
                    shared["students"].pop(stu_id, None)

        # ── Summary Metrics + Global HUD Strip ────────────────────
        all_students = list(shared["students"].values())
        total_n      = len(all_students)
        focused_n    = sum(1 for s in all_students if s["status"] == "FOCUSED")
        not_foc_n    = total_n - focused_n
        pct          = round(focused_n / total_n * 100) if total_n else 0

        with state_lock:
            shared["summary"] = {
                "total":       total_n,
                "focused":     focused_n,
                "not_focused": not_foc_n,
                "focus_pct":   pct,
            }

        cv2.rectangle(frame, (0, 0), (w, 28), (20, 20, 20), -1)
        cv2.putText(frame,
                    f"Students: {total_n} | Focused: {focused_n} | Not Focused: {not_foc_n} | Rate: {pct}% | Present: {len(already_present)}",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 100), 1)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    log_event("all", "SessionEnd", time.time() - session_start)
    print(f"\nSession ended. Total attendance recorded: {len(already_present)} student(s).")

# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    known_encodings, known_names = load_registered_students()
    if len(known_encodings) == 0:
        print("Error: No photos found in database. Exiting.")
        exit()

    print(f"Flask API → http://localhost:{API_PORT}/api/state")
    threading.Thread(target=run_flask, daemon=True).start()

    print("Camera starting — look naturally for calibration. Press ESC to quit.")
    camera_loop(known_encodings, known_names)
