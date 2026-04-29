# Real-Time Sign Language Translator

A computer vision application that translates live American Sign Language (ASL) hand gestures into speech in real time.

## Results
| Metric | Value |
|--------|-------|
| Dataset | Sign Language MNIST (27,455 train + 7,172 test images) |
| Model | Convolutional Neural Network (CNN) |
| Gesture Recognition Accuracy | **99.99%** |
| Webcam Processing Speed | **30+ FPS** |
| Inference Latency | **< 200 ms** |

## Architecture
- **OpenCV** — webcam capture and frame rendering at 30+ FPS
- **MediaPipe Hands** — real-time hand landmark detection and bounding box extraction
- **TensorFlow / Keras CNN** — gesture classification from 28x28 grayscale hand region
  - `Conv2D(32) -> BN -> Conv2D(32) -> MaxPool -> Dropout(0.25)`
  - `Conv2D(64) -> BN -> Conv2D(64) -> MaxPool -> Dropout(0.25)`
  - `Dense(256) -> Dropout(0.4) -> Dense(24, softmax)`
- **pyttsx3** — offline text-to-speech output (non-blocking thread)

## How to Run

```bash
pip install -r requirements.txt

# Step 1 - Train the CNN (downloads dataset automatically via Kaggle)
python train_model.py

# Step 2 - Start the live translator (requires webcam)
python realtime.py
```

### Controls (realtime.py)
| Key | Action |
|-----|--------|
| `SPACE` | Add current predicted letter to word buffer + speak it |
| `C` | Speak the full buffered word then clear it |
| `Q` | Quit |

## Dataset
Sign Language MNIST — 27,455 training / 7,172 test 28x28 grayscale images of ASL hand signs for letters A-Y (excluding J and Z, which require motion). Downloaded automatically on first run via `kagglehub`.

## Technologies
- Python 3
- TensorFlow / Keras
- OpenCV
- MediaPipe
- pyttsx3
- kagglehub
