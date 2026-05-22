import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pandas as pd

# ── Reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config ────────────────────────────────────────────────────
IMG_SIZE   = 28
NUM_CLASSES = 24
BATCH_SIZE  = 64
EPOCHS      = 60
MODEL_PATH  = "sign_language_cnn.keras"

def remap_labels(labels):
    """
    Sign Language MNIST raw labels: 0-25 (letters A-Z), but J (9) and Z (25)
    are excluded because they require motion. So the dataset has 24 classes:
    labels 0-8 → classes 0-8, labels 10-25 → classes 9-23.
    """
    remapped = []
    for l in labels:
        if l < 9:
            remapped.append(l)
        elif l == 9:
            # J should not appear in dataset, but just in case skip
            remapped.append(-1)
        else:
            remapped.append(l - 1)   # 10→9, 11→10, ..., 25→24 — but 25 not in data
    return np.array(remapped)


def preprocess_roi(image_28x28_gray):
    """
    Simulate the preprocessing that realtime.py will apply to webcam ROIs.
    Applies CLAHE for contrast equalization and normalizes.
    Since we can't apply CLAHE easily in keras layers, we do it in numpy
    and apply augmentation on top.
    """
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(image_28x28_gray)
    return eq.astype('float32') / 255.0


def load_data():
    train_path = os.path.join("sing language mnist", "sign_mnist_train.csv")
    test_path  = os.path.join("sing language mnist", "sign_mnist_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[ERROR] CSV files not found in 'sing language mnist' folder.")
        raise SystemExit(1)

    print(f"Reading {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Reading {test_path}...")
    test_df  = pd.read_csv(test_path)

    y_train_raw = train_df['label'].values
    x_train_raw = train_df.drop('label', axis=1).values.reshape(-1, IMG_SIZE, IMG_SIZE).astype('uint8')

    y_test_raw  = test_df['label'].values
    x_test_raw  = test_df.drop('label', axis=1).values.reshape(-1, IMG_SIZE, IMG_SIZE).astype('uint8')

    print(f"Raw train: {x_train_raw.shape}, unique labels: {np.unique(y_train_raw)}")

    # Remap labels
    y_train = remap_labels(y_train_raw)
    y_test  = remap_labels(y_test_raw)

    # Remove any -1 labels (J class which shouldn't appear)
    valid_train = y_train >= 0
    valid_test  = y_test  >= 0
    x_train_raw, y_train = x_train_raw[valid_train], y_train[valid_train]
    x_test_raw,  y_test  = x_test_raw[valid_test],   y_test[valid_test]

    # Apply CLAHE contrast equalization to match realtime preprocessing
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    x_train = np.array([clahe.apply(img).astype('float32') / 255.0 for img in x_train_raw])
    x_test  = np.array([clahe.apply(img).astype('float32') / 255.0 for img in x_test_raw])

    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test  = x_test.reshape(-1,  IMG_SIZE, IMG_SIZE, 1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test  = tf.keras.utils.to_categorical(y_test,  num_classes=NUM_CLASSES)

    print(f"Preprocessed train: {x_train.shape}, test: {x_test.shape}")
    print(f"Remapped classes: {NUM_CLASSES}  (A-Y excluding J and Z)")
    return x_train, y_train, x_test, y_test


def build_model():
    """
    Deeper CNN with BatchNorm and L2 regularization to improve
    generalization to real-world webcam conditions.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def build_augmentation_pipeline():
    """
    Data augmentation that simulates webcam conditions:
    - small rotations (hand tilt)
    - small translations (hand not perfectly centered)
    - small zoom variations
    - horizontal flip (mirrored hand)
    """
    aug = tf.keras.Sequential([
        layers.RandomRotation(0.1),           # ±36 degrees
        layers.RandomTranslation(0.1, 0.1),   # ±10% shift
        layers.RandomZoom((-0.1, 0.1)),        # ±10% zoom
        layers.RandomFlip("horizontal"),       # mirror hand
    ], name="augmentation")
    return aug


def train():
    x_train, y_train, x_test, y_test = load_data()

    # Build augmentation + model pipeline
    aug = build_augmentation_pipeline()
    base_model = build_model()

    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x       = aug(inputs, training=True)   # augmentation only active during training
    outputs = base_model(x)
    model   = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True,
                      verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True,
                        verbose=1),
    ]

    print(f"\nTraining model (up to {EPOCHS} epochs, batch {BATCH_SIZE})...")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[DONE] Final test accuracy: {acc*100:.2f}%  (loss: {loss:.4f})")
    print(f"   Model saved to: {MODEL_PATH}")

    return history


if __name__ == '__main__':
    train()
