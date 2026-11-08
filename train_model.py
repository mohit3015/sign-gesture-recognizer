"""Step 2: Train the CNN on the collected dataset."""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import GESTURES, IMG_SIZE

DATASET_DIR = "dataset"
EPOCHS = 40
BATCH_SIZE = 32

# ---------- 1. Load data ----------
X, y = [], []
for label, g in enumerate(GESTURES):
    folder = os.path.join(DATASET_DIR, g)
    for f in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        X.append(img)
        y.append(label)

X, y = np.array(X), np.array(y)
print("Total images:", len(X), "| per class:", np.bincount(y))

y_cat = tf.keras.utils.to_categorical(y, len(GESTURES))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42)

# ---------- 2. Augmentation (rotation, scale, flip) ----------
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

# ---------- 3. CNN ----------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(GESTURES), activation="softmax"),
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---------- 4. Train ----------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint("sign_model.h5", monitor="val_accuracy", save_best_only=True),
]
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ---------- 5. Evaluate (report ke Results chapter ke liye) ----------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc * 100:.2f}%  |  Test loss: {loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification report:\n",
      classification_report(y_true, y_pred, target_names=GESTURES))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

model.save("sign_model.h5")

# training graphs
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.savefig("training_graphs.png", dpi=150)
print("Saved: sign_model.h5, training_graphs.png")