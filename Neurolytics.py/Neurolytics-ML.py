"""
Usage
-----
  python Neurolytics-ML.py --port COM6          # live ESP32-S3
  python Neurolytics-ML.py --port COM6 --hz 2   # 2 readings/sec
  python Neurolytics-ML.py --no-arduino          # demo / no hardware
  python Neurolytics-ML.py --retrain             # retrain ML model
  python Neurolytics-ML.py --smoke-test          # accuracy self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Optional deps ──────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠️  scikit-learn / joblib not installed — rule-based fallback active.\n"
          "    pip install scikit-learn joblib")

try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False
    print("⚠️  pyserial not installed — Arduino connection unavailable.\n"
          "    pip install pyserial")

try:
    from flask import Flask, jsonify
    FLASK_OK = True
except ImportError:
    FLASK_OK = False
    print("⚠️  Flask not installed — dashboard API unavailable.\n"
          "    pip install flask")


# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

LABEL_MAP    = {0: "Not Focused", 1: "Half Focus", 2: "Focused"}
MODEL_PATH   = "neurolytics_data/env_model.pkl"
HISTORY_SIZE = 300          # readings kept for session stats
API_PORT     = 5051         # dashboard polls http://localhost:5051/api/state
MAX_ALERTS   = 100          # max alerts kept in memory


# ══════════════════════════════════════════════════════════════════════════
# Training-data generator
# ══════════════════════════════════════════════════════════════════════════

def _generate_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic labelled dataset derived from the domain rules.

    Focused      : temp 20-24 | light 360-1600 | motion 1
                   humidity 57-63 | sound 2000-2500
    Half Focus   : one value just outside a boundary, or motion=0
                   with otherwise borderline readings
    Not Focused  : clearly out-of-range value, or motion=0 + multiple bad readings
    """
    rng = np.random.RandomState(seed)
    X, y = [], []

    def add(t, li, mo, h, s, label):
        X.append([float(t), float(li), float(mo), float(h), float(s)])
        y.append(label)

    # ── Label 2 : Focused ──────────────────────────────────────────────
    for _ in range(700):
        add(rng.uniform(20,24), rng.uniform(360,1600), 1,
            rng.uniform(57,63), rng.uniform(2000,2500), 2)
    for t  in [20,21,22,23,24]:          add(t,  1600, 1, 60, 2300, 2)
    for li in [360,500,700,1000,1400]:   add(22, li,   1, 57, 2000, 2)
    for h  in [57,58,59,60,61,62,63]:    add(20, 800,  1, h,  2100, 2)
    for s  in [2000,2100,2200,2300,2400,2500]: add(23, 900, 1, 60, s, 2)

    # ── Label 1 : Half Focus ───────────────────────────────────────────
    for row in [(18,1700,0,64,1800),(19,1750,1,65,1900),
                (25,300, 0,55,2600),(26,250, 1,54,2700)]:
        add(*row, 1)
    for _ in range(150):
        t = rng.choice([rng.uniform(18,19.99), rng.uniform(25,26)])
        add(t, rng.uniform(360,1600), 1, rng.uniform(57,63), rng.uniform(2000,2500), 1)
    for _ in range(150):
        li = rng.choice([rng.uniform(201,359), rng.uniform(1601,1999)])
        add(rng.uniform(20,24), li, 1, rng.uniform(57,63), rng.uniform(2000,2500), 1)
    for _ in range(150):
        h = rng.choice([rng.uniform(51,56), rng.uniform(64,69)])
        add(rng.uniform(20,24), rng.uniform(360,1600), 1, h, rng.uniform(2000,2500), 1)
    for _ in range(150):
        s = rng.choice([rng.uniform(1701,1999), rng.uniform(2501,2999)])
        add(rng.uniform(20,24), rng.uniform(360,1600), 1, rng.uniform(57,63), s, 1)
    for _ in range(150):
        add(rng.uniform(18,26), rng.uniform(250,1750), 0,
            rng.uniform(54,65), rng.uniform(1800,2700), 1)

    # ── Label 0 : Not Focused ──────────────────────────────────────────
    for _ in range(150):
        t = rng.choice([rng.uniform(5,17.9), rng.uniform(26.1,40)])
        add(t, rng.uniform(360,1600), 1, rng.uniform(57,63), rng.uniform(2000,2500), 0)
    for _ in range(150):
        li = rng.choice([rng.uniform(0,200), rng.uniform(2000,3000)])
        add(rng.uniform(20,24), li, 1, rng.uniform(57,63), rng.uniform(2000,2500), 0)
    for _ in range(150):
        h = rng.choice([rng.uniform(10,50), rng.uniform(70,95)])
        add(rng.uniform(20,24), rng.uniform(360,1600), 1, h, rng.uniform(2000,2500), 0)
    for _ in range(150):
        s = rng.choice([rng.uniform(0,1700), rng.uniform(3000,4095)])
        add(rng.uniform(20,24), rng.uniform(360,1600), 1, rng.uniform(57,63), s, 0)
    for _ in range(150):
        t  = rng.choice([rng.uniform(5,17.9), rng.uniform(27,40)])
        li = rng.uniform(0, 200)
        add(t, li, 0, rng.uniform(10,50), rng.uniform(0,1700), 0)
    for _ in range(100):
        add(rng.uniform(28,40), rng.uniform(0,150), 0,
            rng.uniform(75,95), rng.uniform(3000,4095), 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════
# Model training
# ══════════════════════════════════════════════════════════════════════════

def train_model(save_path: str = MODEL_PATH):
    if not SKLEARN_OK:
        return None
    X, y = _generate_data()
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    model.fit(X, y)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"✅ Model trained  |  CV: {cv.mean():.1%} ± {cv.std():.1%}")
    print(f"   Samples: {len(y)}  "
          f"(Focused={sum(y==2)}, Half={sum(y==1)}, NotFocused={sum(y==0)})")
    print(f"   Saved → {save_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════
# Rule-based fallback
# ══════════════════════════════════════════════════════════════════════════

def _rule_predict(temp, light, motion, humidity, sound) -> tuple[int, str, dict]:
    hard_bad = (
        temp < 18 or temp > 26
        or light <= 200 or light >= 2000
        or humidity <= 50 or humidity >= 70
        or sound <= 1700 or sound >= 3000
    )
    if hard_bad:
        return 0, LABEL_MAP[0], {"Not Focused": 1.0}

    borderline = (
        (18 <= temp < 20 or 25 < temp <= 26)
        or (201 <= light < 360  or 1600 < light < 2000)
        or (51 <= humidity < 57 or 63 < humidity < 70)
        or (1701 <= sound < 2000 or 2500 < sound < 3000)
        or motion == 0
    )
    if motion == 0 and not borderline:
        return 0, LABEL_MAP[0], {"Not Focused": 1.0}
    if borderline:
        return 1, LABEL_MAP[1], {"Half Focus": 1.0}
    return 2, LABEL_MAP[2], {"Focused": 1.0}


# ══════════════════════════════════════════════════════════════════════════
# EnvClassifier
# ══════════════════════════════════════════════════════════════════════════

class EnvClassifier:
    def __init__(self, model_path: str = MODEL_PATH, retrain: bool = False):
        self._model = None
        if not SKLEARN_OK:
            return
        p = Path(model_path)
        if not retrain and p.exists():
            try:
                self._model = joblib.load(p)
                print(f"✅ Classifier loaded ← {p}")
                return
            except Exception as e:
                print(f"⚠️  Could not load model ({e}) — retraining …")
        self._model = train_model(model_path)

    def predict(self, temperature, light, motion, humidity, sound
                ) -> tuple[int, str, dict]:
        if self._model is None:
            return _rule_predict(temperature, light, motion, humidity, sound)
        feat      = np.array([[temperature, light, motion, humidity, sound]],
                             dtype=np.float32)
        label_int = int(self._model.predict(feat)[0])
        proba     = self._model.predict_proba(feat)[0]
        conf      = {LABEL_MAP[i]: round(float(p), 3) for i, p in enumerate(proba)}
        return label_int, LABEL_MAP[label_int], conf

    def predict_from_dict(self, env: dict) -> tuple[int, str, dict]:
        return self.predict(
            temperature = env.get("temperature", 22.0),
            light       = env.get("light_level",  800),
            motion      = env.get("motion",          1),
            humidity    = env.get("humidity",      60.0),
            sound       = env.get("noise_level",  2200),
        )


# ══════════════════════════════════════════════════════════════════════════
# ArduinoReader
# ══════════════════════════════════════════════════════════════════════════

class ArduinoReader:
    """
    Background thread that reads JSON lines from the ESP32-S3 over USB-serial.

    Expected line:
        {"temp":22.5,"noise":2300,"light":1200,"motion":1,"humidity":60.0}
    """
    DEFAULT = {
        "temperature": 22.0,
        "noise_level": 2200,
        "light_level":  800,
        "motion":          1,
        "humidity":      60.0,
    }

    def __init__(self, port: str | None = None, baud: int = 115200):
        self.port      = port
        self.baud      = baud
        self.connected = False
        self._data     = dict(self.DEFAULT)
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._ser      = None

    def connect(self) -> bool:
        if not SERIAL_OK:
            print("⚠️  pyserial unavailable — using sensor defaults.")
            return False
        try:
            if self.port is None:
                self.port = self._auto_detect()
            if self.port is None:
                print("⚠️  No ESP32 port found — using sensor defaults.")
                return False
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)           # let ESP32 reset after DTR toggle
            self.connected = True
            print(f"✅ ESP32-S3 connected on {self.port} @ {self.baud} baud")
            threading.Thread(target=self._read_loop, daemon=True).start()
            return True
        except serial.SerialException as e:
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                print(f"⚠️  {self.port} is in use by another program "
                      f"(close Arduino IDE Serial Monitor) — using defaults.")
            else:
                print(f"⚠️  ESP32 connect error: {e}")
            return False
        except Exception as e:
            print(f"⚠️  ESP32 connect error: {e}")
            return False

    def _auto_detect(self) -> str | None:
        keywords = ("arduino", "ch340", "cp210", "usb serial", "esp32", "silicon labs")
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if any(k in desc for k in keywords):
                return p.device
        return None

    def _read_loop(self):
        while not self._stop.is_set():
            try:
                raw = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                d = json.loads(raw)
                if d.get("error"):
                    continue
                with self._lock:
                    self._data = {
                        "temperature": float(d.get("temp",     self._data["temperature"])),
                        "noise_level":   int(d.get("noise",    self._data["noise_level"])),
                        "light_level":   int(d.get("light",    self._data["light_level"])),
                        "motion":        int(d.get("motion",   self._data["motion"])),
                        "humidity":    float(d.get("humidity", self._data["humidity"])),
                    }
            except (json.JSONDecodeError, ValueError):
                pass
            except Exception:
                time.sleep(0.3)

    @property
    def data(self) -> dict:
        with self._lock:
            return dict(self._data)

    def stop(self):
        self._stop.set()
        if self._ser and self._ser.is_open:
            self._ser.close()


# ══════════════════════════════════════════════════════════════════════════
# DataLogger
# ══════════════════════════════════════════════════════════════════════════

class DataLogger:
    HEADERS = [
        "timestamp", "session_time_s",
        "focus_label", "focus_int",
        "conf_focused", "conf_half_focus", "conf_not_focused",
        "temperature", "humidity", "noise_level", "light_level", "motion",
    ]

    def __init__(self, csv_path: str):
        self.csv_path      = csv_path
        self.session_start = time.time()
        self._rows: list[dict] = []
        self._file   = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.HEADERS,
                                      extrasaction="ignore")
        self._writer.writeheader()
        print(f"📝 Logging → {csv_path}")

    def log(self, env: dict, label_int: int, label_str: str, conf: dict):
        row = {
            "timestamp":        datetime.now().isoformat(),
            "session_time_s":   round(time.time() - self.session_start, 2),
            "focus_label":      label_str,
            "focus_int":        label_int,
            "conf_focused":     conf.get("Focused",     ""),
            "conf_half_focus":  conf.get("Half Focus",  ""),
            "conf_not_focused": conf.get("Not Focused", ""),
            "temperature":      env.get("temperature",  ""),
            "humidity":         env.get("humidity",     ""),
            "noise_level":      env.get("noise_level",  ""),
            "light_level":      env.get("light_level",  ""),
            "motion":           env.get("motion",       ""),
        }
        self._writer.writerow(row)
        self._rows.append(row)

    def save_summary(self, json_path: str):
        if not self._rows:
            return
        dur    = round(time.time() - self.session_start)
        counts = {"Focused": 0, "Half Focus": 0, "Not Focused": 0}
        for r in self._rows:
            lbl = r.get("focus_label", "")
            if lbl in counts:
                counts[lbl] += 1
        total = len(self._rows)
        summary = {
            "duration_sec":   dur,
            "total_readings": total,
            "focus_counts":   counts,
            "focus_pct":      {k: round(v / total * 100, 1) if total else 0
                               for k, v in counts.items()},
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 50)
        print("  Session Summary — Neurolytics Env Monitor")
        print("=" * 50)
        print(f"  Duration     : {dur // 60:02d}:{dur % 60:02d}")
        print(f"  Total reads  : {total}")
        for lbl, cnt in counts.items():
            pct = round(cnt / total * 100, 1) if total else 0
            bar = "█" * int(pct / 5)
            print(f"  {lbl:<13}: {cnt:>4}  ({pct:5.1f}%)  {bar}")
        print(f"\n  CSV  → {self.csv_path}")
        print(f"  JSON → {json_path}")
        print("=" * 50)

    def close(self):
        self._file.flush()
        self._file.close()


# ══════════════════════════════════════════════════════════════════════════
# Shared application state  (written by monitor thread, read by Flask)
# ══════════════════════════════════════════════════════════════════════════

class AppState:
    """Thread-safe container for the latest reading + alert log."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._latest = {
            "env":       dict(ArduinoReader.DEFAULT),
            "label_int": 2,
            "label_str": "Focused",
            "conf":      {},
        }
        self._alerts: list[dict] = []          # newest first

    def update(self, env: dict, label_int: int, label_str: str, conf: dict):
        with self._lock:
            self._latest = {
                "env":       env,
                "label_int": label_int,
                "label_str": label_str,
                "conf":      conf,
            }

    def push_alert(self, source: str, message: str):
        """Add an alert (called from monitor thread)."""
        entry = {
            "student": source,
            "message": message,
            "time":    datetime.now().strftime("%H:%M:%S"),
            "type":    "iot",
        }
        with self._lock:
            self._alerts.insert(0, entry)
            if len(self._alerts) > MAX_ALERTS:
                self._alerts.pop()

    def snapshot(self) -> dict:
        with self._lock:
            lat   = self._latest
            env   = lat["env"]
            label = lat["label_str"]
            conf  = lat["conf"]
            alerts = list(self._alerts)

        # Build the JSON the dashboard expects
        focused_count    = 1 if label == "Focused" else 0
        not_focused_count = 1 if label == "Not Focused" else 0
        half_count        = 1 if label == "Half Focus"  else 0

        return {
            "summary": {
                "total":      1,
                "focused":    focused_count,
                "not_focused": not_focused_count + half_count,
                "focus_pct":  100 if label == "Focused" else (50 if label == "Half Focus" else 0),
            },
            "students": {},          # CV removed — no student cards
            "sensors": {
                "temperature": env.get("temperature", 22.0),
                "humidity":    env.get("humidity",    60.0),
                "light":       env.get("light_level",  800),
                "noise":       env.get("noise_level", 2200),
                "motion":      env.get("motion",         1),
            },
            "env_focus": {
                "label":       label,
                "int":         lat["label_int"],
                "conf":        conf,
            },
            "alerts": alerts,
        }


# ══════════════════════════════════════════════════════════════════════════
# Flask API  (runs in a daemon thread)
# ══════════════════════════════════════════════════════════════════════════

def start_api(state: AppState, port: int = API_PORT):
    if not FLASK_OK:
        print("⚠️  Flask unavailable — dashboard API not started.")
        return

    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)         # silence noisy Flask request logs

    app = Flask(__name__)

    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"]  = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/api/state")
    def api_state():
        return jsonify(state.snapshot())

    @app.route("/api/health")
    def api_health():
        return jsonify({"status": "ok", "port": port})

    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False),
        daemon=True,
        name="flask-api",
    )
    t.start()
    print(f"🌐 Dashboard API → http://localhost:{port}/api/state")


# ══════════════════════════════════════════════════════════════════════════
# Alert generator  (runs inside monitor loop, fires every 3 s max)
# ══════════════════════════════════════════════════════════════════════════

# Thresholds that trigger IoT alerts sent to the dashboard
_ALERT_THRESHOLDS = {
    "temperature": {"max": 30,  "min": None, "unit": "°C",  "icon": "🌡"},
    "humidity":    {"max": 70,  "min": 40,   "unit": "%",   "icon": "💧"},
    "light_level": {"max": 1600,"min": 200,  "unit": "lx",  "icon": "☀️"},
    "noise_level": {"max": 3000,"min": None, "unit": "ADC", "icon": "🔊"},
}

_ALERT_INTERVAL = 3.0          # seconds between repeated alerts for same sensor
_last_alert_ts: dict[str, float] = {}


def _check_and_push_alerts(env: dict, label_str: str, state: AppState):
    now = time.time()

    # Environmental focus label alert
    key = "env_focus"
    if label_str != "Focused":
        if now - _last_alert_ts.get(key, 0) >= _ALERT_INTERVAL:
            msg = (f"⚠️ Environment: {label_str} — "
                   f"T={env.get('temperature',0):.1f}°C "
                   f"H={env.get('humidity',0):.0f}% "
                   f"L={env.get('light_level',0)}lx")
            state.push_alert("IoT Classifier", msg)
            _last_alert_ts[key] = now

    # Per-sensor threshold alerts
    for field, thresh in _ALERT_THRESHOLDS.items():
        val = env.get(field, 0)
        breach_msg = None

        if thresh["max"] is not None and val > thresh["max"]:
            breach_msg = (f"{thresh['icon']} {field.replace('_',' ').title()} too high: "
                          f"{val} {thresh['unit']} (max {thresh['max']})")
        elif thresh["min"] is not None and val < thresh["min"]:
            breach_msg = (f"{thresh['icon']} {field.replace('_',' ').title()} too low: "
                          f"{val} {thresh['unit']} (min {thresh['min']})")

        if breach_msg and now - _last_alert_ts.get(field, 0) >= _ALERT_INTERVAL:
            state.push_alert("IoT Sensor", breach_msg)
            _last_alert_ts[field] = now

    # Motion absent alert
    if env.get("motion", 1) == 0:
        if now - _last_alert_ts.get("motion", 0) >= _ALERT_INTERVAL:
            state.push_alert("PIR Sensor", "🚶 No motion detected in classroom")
            _last_alert_ts["motion"] = now


# ══════════════════════════════════════════════════════════════════════════
# Console display helpers
# ══════════════════════════════════════════════════════════════════════════

_CLR = {
    "Focused":     "\033[92m",
    "Half Focus":  "\033[93m",
    "Not Focused": "\033[91m",
}
_RST = "\033[0m"

_ADVICE = {
    "Focused":     "✅ Environment is optimal for learning.",
    "Half Focus":  "⚠️  Environment is suboptimal — check temp/light/humidity/sound.",
    "Not Focused": "🚫 Environment is poor — one or more sensors out of range.",
}


# ══════════════════════════════════════════════════════════════════════════
# NeurolyticsEnv — main orchestrator
# ══════════════════════════════════════════════════════════════════════════

class NeurolyticsEnv:
    def __init__(
        self,
        arduino_port: str | None = None,
        no_arduino:   bool  = False,
        retrain:      bool  = False,
        save_data:    bool  = True,
        data_dir:     str   = "neurolytics_data",
        poll_hz:      float = 1.0,
        api_port:     int   = API_PORT,
    ):
        self.poll_hz       = poll_hz
        self.session_start = time.time()
        self._history: deque[int] = deque(maxlen=HISTORY_SIZE)

        # Shared state (thread-safe, read by Flask)
        self.state = AppState()

        # Arduino
        self.arduino = ArduinoReader(port=arduino_port)
        if not no_arduino:
            self.arduino.connect()

        # Classifier
        self.clf = EnvClassifier(retrain=retrain)

        # Flask API
        start_api(self.state, port=api_port)

        # Logger
        self.logger    = None
        self.json_path = None
        if save_data:
            dp = Path(data_dir)
            dp.mkdir(exist_ok=True)
            sid            = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path       = dp / f"env_{sid}.csv"
            self.json_path = dp / f"env_{sid}_summary.json"
            self.logger    = DataLogger(str(csv_path))

        print(f"✅ NeurolyticsEnv ready  |  poll={poll_hz} Hz\n")

    # ── Single poll ────────────────────────────────────────────────────────
    def poll(self) -> dict:
        env = self.arduino.data
        label_int, label_str, conf = self.clf.predict_from_dict(env)
        self._history.append(label_int)

        # Update shared state for Flask
        self.state.update(env, label_int, label_str, conf)

        # Push alerts (throttled to _ALERT_INTERVAL seconds)
        _check_and_push_alerts(env, label_str, self.state)

        if self.logger:
            self.logger.log(env, label_int, label_str, conf)

        return {"env": env, "label_int": label_int, "label_str": label_str, "conf": conf}

    # ── Live console loop ──────────────────────────────────────────────────
    def run(self):
        interval = 1.0 / max(self.poll_hz, 0.1)
        print("  Press Ctrl-C to stop.\n")
        print(f"  {'TIME':>7}  {'LABEL':<13}  "
              f"{'TEMP':>6}  {'HUM':>5}  {'LIGHT':>6}  "
              f"{'SOUND':>5}  MOT  CONF")
        print("  " + "─" * 72)
        _last_label = None

        try:
            while True:
                t0     = time.time()
                result = self.poll()
                env    = result["env"]
                lbl    = result["label_str"]
                conf   = result["conf"]
                color  = _CLR.get(lbl, "")
                sess   = round(time.time() - self.session_start)
                pct    = f"{conf.get(lbl, 1.0):.0%}" if conf else "rule"

                print(
                    f"  {sess:>7}s  "
                    f"{color}{lbl:<13}{_RST}  "
                    f"{env.get('temperature',0):>5.1f}°  "
                    f"{env.get('humidity',0):>4.1f}%  "
                    f"{env.get('light_level',0):>6}lx  "
                    f"{env.get('noise_level',0):>5}  "
                    f"  {env.get('motion',0)}  "
                    f"{pct}"
                )
                if lbl != _last_label:
                    print(f"\n  {_ADVICE[lbl]}\n")
                    _last_label = lbl

                elapsed = time.time() - t0
                time.sleep(max(0, interval - elapsed))

        except KeyboardInterrupt:
            print("\n\n  Stopping …")
        finally:
            self.close()

    # ── Teardown ───────────────────────────────────────────────────────────
    def close(self):
        if self.logger:
            self.logger.save_summary(str(self.json_path))
            self.logger.close()
        self.arduino.stop()
        print("✅ NeurolyticsEnv closed.")


# ══════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════

def _run_smoke_test():
    print("\n── Smoke test ─────────────────────────────────────────────")
    clf = EnvClassifier()
    tests = [
        (22, 1600, 1, 60, 2300, "Focused"),
        (22,  360, 1, 57, 2000, "Focused"),
        (20,  800, 1, 60, 2100, "Focused"),
        (18, 1700, 0, 64, 1800, "Half Focus"),
        (25,  300, 0, 55, 2600, "Half Focus"),
        (15,  800, 1, 60, 2200, "Not Focused"),
        (22,  100, 1, 60, 2200, "Not Focused"),
        (22,  800, 1, 45, 2200, "Not Focused"),
        (22,  800, 1, 60, 1500, "Not Focused"),
        (22,  800, 0, 60, 2200, "Not Focused"),
    ]
    ok = 0
    for t, li, mo, h, s, exp in tests:
        pi, ps, conf = clf.predict(t, li, mo, h, s)
        tick = "✅" if ps == exp else "❌"
        pct  = f"{conf.get(ps, 1.0):.2f}" if conf else "rule"
        print(f"  {tick}  [{ps:<13}]  expected={exp:<13}  conf={pct}  "
              f"T={t} L={li} Mo={mo} H={h} S={s}")
        ok += ps == exp
    print(f"\n  Passed {ok}/{len(tests)}\n")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neurolytics Environmental Focus Monitor")
    parser.add_argument("--port",       default=None,
                        help="ESP32-S3 serial port (e.g. COM6 or /dev/ttyUSB0)")
    parser.add_argument("--baud",       type=int, default=115200)
    parser.add_argument("--hz",         type=float, default=1.0,
                        help="Sensor readings per second (default: 1)")
    parser.add_argument("--api-port",   type=int, default=API_PORT,
                        help=f"Flask API port (default: {API_PORT})")
    parser.add_argument("--no-arduino", action="store_true",
                        help="Skip serial connection — use sensor defaults")
    parser.add_argument("--retrain",    action="store_true",
                        help="Force ML model retraining then exit")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run accuracy self-test then exit")
    parser.add_argument("--no-save",    action="store_true",
                        help="Disable CSV/JSON logging")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        sys.exit(0)

    if args.retrain:
        train_model()
        sys.exit(0)

    monitor = NeurolyticsEnv(
        arduino_port = args.port,
        no_arduino   = args.no_arduino,
        retrain      = False,
        save_data    = not args.no_save,
        poll_hz      = args.hz,
        api_port     = args.api_port,
    )
    monitor.run()
