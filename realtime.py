#!/usr/bin/env python3
"""
realtime.py  —  Real-Time Sign Language Translator
Requires : train_model.py to have been run first (model file present)
Controls :
    SPACE  — capture current prediction and queue it for speech
    C      — clear word buffer
    Q      — quit
"""

import time
import threading
import collections
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import pyttsx3

# ── Config ────────────────────────────────────────────────────
MODEL_PATH  = "sign_language_cnn.keras"
IMG_SIZE    = 28
LABELS      = list("ABCDEFGHIKLMNOPQRSTUVWXY")
CONF_THRESH = 0.70          # minimum confidence to display prediction
SMOOTH_N    = 7             # frames to vote over (temporal smoothing)
HAND_PAD    = 30            # px padding around detected hand bbox

# ── TTS (non-blocking thread) ─────────────────────────────────
_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", 160)
_tts_queue: list[str] = []
_tts_lock  = threading.Lock()

def _tts_worker():
    while True:
        with _tts_lock:
            if _tts_queue:
                text = _tts_queue.pop(0)
                _tts_engine.say(text)
                _tts_engine.runAndWait()
        time.sleep(0.05)

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text: str):
    with _tts_lock:
        _tts_queue.append(text)


# ── Load model ────────────────────────────────────────────────
def load_model():
    if not __import__("os").path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run  python train_model.py  first.")
        raise SystemExit(1)
    print(f"Loading model from {MODEL_PATH}...")
    return tf.keras.models.load_model(MODEL_PATH)


# ── Predict from ROI ──────────────────────────────────────────
def predict(model, roi_bgr):
    gray  = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    inp   = gray.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    return LABELS[idx], conf


# ── Hand ROI from MediaPipe landmarks ─────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_hand_roi(frame, hand_landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - HAND_PAD)
    y1 = max(0, int(min(ys)) - HAND_PAD)
    x2 = min(w, int(max(xs)) + HAND_PAD)
    y2 = min(h, int(max(ys)) + HAND_PAD)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ── Main loop ─────────────────────────────────────────────────
def main():
    model  = load_model()
    cap    = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vote_buf   = collections.deque(maxlen=SMOOTH_N)
    word_buf   = []
    fps_times  = collections.deque(maxlen=30)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print("\n  [SPACE] Add letter to word  |  [C] Clear  |  [Q] Quit\n")

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pred_letter = "—"
        confidence  = 0.0
        bbox        = None

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                roi, bbox = get_hand_roi(frame, hand_lm)
                if roi is not None and roi.size > 0:
                    t_inf = time.perf_counter()
                    ltr, conf = predict(model, roi)
                    latency_ms = (time.perf_counter() - t_inf) * 1000

                    vote_buf.append(ltr)
                    # majority vote for smoothing
                    counts      = collections.Counter(vote_buf)
                    pred_letter = counts.most_common(1)[0][0]
                    confidence  = conf
                break   # only process first hand

        # ── FPS ───────────────────────────────────────────────
        fps_times.append(time.perf_counter())
        fps = len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-6) if len(fps_times) > 1 else 0

        # ── Draw ──────────────────────────────────────────────
        h, w = frame.shape[:2]

        if bbox:
            x1, y1, x2, y2 = bbox
            col = (0, 220, 0) if confidence >= CONF_THRESH else (0, 160, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        # Side panel
        panel = np.zeros((h, 200, 3), dtype=np.uint8)
        cv2.putText(panel, "SIGN TRANSLATOR", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.line(panel, (0, 38), (200, 38), (80, 80, 80), 1)

        cv2.putText(panel, "LETTER", (8, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.putText(panel, pred_letter, (8, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 120), 3)

        cv2.putText(panel, f"Conf: {confidence*100:.0f}%", (8, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 0) if confidence >= CONF_THRESH else (0, 120, 255), 1)

        cv2.putText(panel, f"FPS : {fps:.0f}", (8, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.line(panel, (0, 180), (200, 180), (80, 80, 80), 1)
        cv2.putText(panel, "WORD", (8, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        word_disp = "".join(word_buf[-12:])           # last 12 chars
        cv2.putText(panel, word_disp, (8, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)

        cv2.putText(panel, "[SPC] add letter", (8, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
        cv2.putText(panel, "[C] clear word",   (8, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
        cv2.putText(panel, "[Q] quit",          (8, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        combined = np.hstack([frame, panel])
        cv2.imshow("Sign Language Translator", combined)

        # ── Key handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if pred_letter != "—" and confidence >= CONF_THRESH:
                word_buf.append(pred_letter)
                speak(pred_letter)
        elif key == ord("c"):
            if word_buf:
                speak("".join(word_buf))
            word_buf.clear()

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
