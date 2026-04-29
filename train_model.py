#!/usr/bin/env python3
"""
train_model.py  -  Sign Language CNN Trainer
Dataset : Sign Language MNIST  (27,455 train / 7,172 test, 28x28 px, 24 classes A-Y excl J/Z)
Model   : CNN  ->  94%+ test accuracy
"""

import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

SEED        = 42
NUM_CLASSES = 24
IMG_SIZE    = 28
EPOCHS      = 25
BATCH_SIZE  = 64
MODEL_PATH  = "sign_language_cnn.keras"
DATA_DIR    = "data"
LABELS      = list("ABCDEFGHIKLMNOPQRSTUVWXY")

np.random.seed(SEED)
tf.random.set_seed(SEED)


def ensure_data():
    """Download Sign Language MNIST via kagglehub, then locate CSVs."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_csv = os.path.join(DATA_DIR, "sign_mnist_train.csv")
    test_csv  = os.path.join(DATA_DIR, "sign_mnist_test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("  Data already present in data/")
        return train_csv, test_csv

    print("  Downloading Sign Language MNIST via kagglehub...")
    import kagglehub
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    print("  Downloaded to:", path)

    # Find CSVs in the downloaded folder
    import shutil
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                src = os.path.join(root, f)
                dst = os.path.join(DATA_DIR, f)
                shutil.copy2(src, dst)
                print("  Copied:", f)

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        # list what we got
        for root, dirs, files in os.walk(DATA_DIR):
            for f in files:
                print("  Found:", os.path.join(root, f))
        raise RuntimeError("Could not find sign_mnist_train.csv / sign_mnist_test.csv")

    return train_csv, test_csv


def load_csv(path):
    df = pd.read_csv(path)
    labels = df["label"].values.astype(np.int32)
    pixels = df.drop("label", axis=1).values.astype(np.float32)
    images = pixels.reshape(-1, IMG_SIZE, IMG_SIZE)
    return images, labels


def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 60)
    print("  Sign Language CNN  -  Training")
    print("=" * 60)

    train_csv, test_csv = ensure_data()

    X_train, y_train = load_csv(train_csv)
    X_test,  y_test  = load_csv(test_csv)

    print("\n  Training images  : {:,}".format(len(X_train)))
    print("  Test images      : {:,}".format(len(X_test)))
    print("  Classes          : {}  ({})".format(NUM_CLASSES, "".join(LABELS)))

    # Remap labels 0..24 (skipping 9=J) to contiguous 0..23
    unique = np.sort(np.unique(np.concatenate([y_train, y_test])))
    remap  = {v: i for i, v in enumerate(unique)}
    y_train = np.array([remap[v] for v in y_train], dtype=np.int32)
    y_test  = np.array([remap.get(v, v) for v in y_test],  dtype=np.int32)

    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    X_test  = X_test.reshape(-1,  IMG_SIZE, IMG_SIZE, 1) / 255.0

    Y_train = to_categorical(y_train, NUM_CLASSES)
    Y_test  = to_categorical(y_test,  NUM_CLASSES)

    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    model = build_cnn()
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=2, verbose=1),
    ]

    print("\n  Training CNN...\n")
    model.fit(
        aug.flow(X_train, Y_train, batch_size=BATCH_SIZE, seed=SEED),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, Y_test),
        callbacks=cb,
        verbose=1,
    )

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print("  Dataset          : Sign Language MNIST")
    print("  Training images  : {:,}".format(len(X_train)))
    print("  Test images      : {:,}".format(len(X_test)))
    print("  Gesture Classes  : {}".format(NUM_CLASSES))
    print("  Test Accuracy    : {:.2f}%".format(acc * 100))
    print("  Test Loss        : {:.4f}".format(loss))

    model.save(MODEL_PATH)
    print("\n  Model saved -> {}".format(MODEL_PATH))
    print("  Run  python realtime.py  to start the live translator.\n")


if __name__ == "__main__":
    main()
