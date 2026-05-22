#!/usr/bin/env python3
"""
realtime.py  —  Real-Time Sign Language Translator
Requires : train_model.py to have been run first (model file present)
Controls :
    SPACE  — capture current prediction and queue it for speech
    C      — clear word buffer + speak it
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
# A-Y excluding J (idx 9) and Z (idx 25).
# These 24 letters match the Sign Language MNIST label mapping used in training.
LABELS      = list("ABCDEFGHIKLMNOPQRSTUVWXY")
CONF_THRESH = 0.65          # minimum confidence to display prediction
SMOOTH_N    = 10            # frames to vote over (temporal smoothing)
HAND_PAD    = 40            # px padding around detected hand bbox

# CLAHE for contrast equalization — MUST match train_model.py preprocessing
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

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
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run  python train_model.py  first.")
        raise SystemExit(1)
    print(f"Loading model from {MODEL_PATH}...")
    return tf.keras.models.load_model(MODEL_PATH)


# ── Predict from ROI ──────────────────────────────────────────
def preprocess_roi(roi_bgr):
    """
    Preprocessing pipeline that EXACTLY mirrors what train_model.py does:
      1. Convert to grayscale
      2. Resize to 28×28
      3. CLAHE equalization (matches training)
      4. Normalize to [0, 1]
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    gray = _clahe.apply(gray)                           # contrast equalization
    return gray.astype('float32') / 255.0


def predict(model, roi_bgr):
    img  = preprocess_roi(roi_bgr)
    inp  = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    return LABELS[idx], conf, probs


# ── Hand ROI from MediaPipe landmarks ─────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def get_hand_roi(frame, hand_landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks]
    ys = [lm.y * h for lm in hand_landmarks]
    x1 = max(0, int(min(xs)) - HAND_PAD)
    y1 = max(0, int(min(ys)) - HAND_PAD)
    x2 = min(w, int(max(xs)) + HAND_PAD)
    y2 = min(h, int(max(ys)) + HAND_PAD)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ── Confidence bar helper ─────────────────────────────────────
def draw_conf_bar(panel, x, y, w, h, value, color):
    cv2.rectangle(panel, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * value)
    if filled > 0:
        cv2.rectangle(panel, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (120, 120, 120), 1)


# ── Main loop ─────────────────────────────────────────────────
def main():
    model = load_model()
    cap   = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        raise SystemExit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

    vote_buf  = collections.deque(maxlen=SMOOTH_N)
    word_buf  = []
    fps_times = collections.deque(maxlen=30)
    last_probs = np.zeros(len(LABELS))

    base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    print("\n  [SPACE] Add letter to word  |  [C] Clear & speak  |  [Q] Quit\n")

    PANEL_W = 220

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = detector.detect(mp_image)

        pred_letter = "—"
        confidence  = 0.0
        bbox        = None

        if results.hand_landmarks:
            for hand_lm in results.hand_landmarks:
                roi, bbox = get_hand_roi(frame, hand_lm)
                if roi is not None and roi.size > 0:
                    ltr, conf, probs = predict(model, roi)
                    last_probs = probs

                    vote_buf.append(ltr)
                    counts      = collections.Counter(vote_buf)
                    pred_letter = counts.most_common(1)[0][0]
                    confidence  = conf
                break   # only first hand

        # ── FPS ───────────────────────────────────────────────
        fps_times.append(time.perf_counter())
        fps = (len(fps_times) /
               (fps_times[-1] - fps_times[0] + 1e-6)) if len(fps_times) > 1 else 0

        # ── Draw ──────────────────────────────────────────────
        h, w = frame.shape[:2]

        if bbox:
            x1, y1, x2, y2 = bbox
            col = (0, 220, 0) if confidence >= CONF_THRESH else (0, 120, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

            # Show preprocessed thumb in corner of frame for debugging
            roi_thumb = preprocess_roi(frame[y1:y2, x1:x2])
            thumb = (roi_thumb * 255).astype(np.uint8)
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            thumb_bgr = cv2.resize(thumb_bgr, (56, 56))
            frame[4:60, 4:60] = thumb_bgr
            cv2.rectangle(frame, (4, 4), (60, 60), (200, 200, 200), 1)
            cv2.putText(frame, "model view", (4, 73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # Side panel
        panel = np.zeros((h, PANEL_W, 3), dtype=np.uint8)

        # Header
        cv2.putText(panel, "SIGN TRANSLATOR", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.line(panel, (0, 38), (PANEL_W, 38), (80, 80, 80), 1)

        # Predicted letter
        cv2.putText(panel, "LETTER", (8, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        letter_col = (0, 255, 120) if confidence >= CONF_THRESH else (80, 80, 80)
        cv2.putText(panel, pred_letter, (8, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, letter_col, 3)

        # Confidence bar
        bar_col = (0, 220, 0) if confidence >= CONF_THRESH else (0, 120, 255)
        draw_conf_bar(panel, 8, 122, PANEL_W - 16, 12, confidence, bar_col)
        cv2.putText(panel, f"{confidence*100:.0f}%", (8, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_col, 1)

        cv2.putText(panel, f"FPS: {fps:.0f}", (PANEL_W - 60, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Top-3 predictions
        cv2.line(panel, (0, 158), (PANEL_W, 158), (60, 60, 60), 1)
        cv2.putText(panel, "TOP-3", (8, 173),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
        top3_idx = np.argsort(last_probs)[::-1][:3]
        for rank, idx in enumerate(top3_idx):
            lbl  = LABELS[idx]
            prob = last_probs[idx]
            yy   = 190 + rank * 18
            cv2.putText(panel, f"{lbl}: {prob*100:.1f}%", (8, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 220, 100) if rank == 0 else (140, 140, 140), 1)

        # Word buffer
        cv2.line(panel, (0, 250), (PANEL_W, 250), (80, 80, 80), 1)
        cv2.putText(panel, "WORD", (8, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
        word_disp = "".join(word_buf[-12:])
        cv2.putText(panel, word_disp, (8, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)

        # Controls hint
        cv2.line(panel, (0, 320), (PANEL_W, 320), (60, 60, 60), 1)
        cv2.putText(panel, "[SPC] add letter", (8, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(panel, "[C] speak & clear", (8, 356),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(panel, "[Q] quit",           (8, 372),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

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
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
