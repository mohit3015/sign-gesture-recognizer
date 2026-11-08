"""Shared helpers: MediaPipe hand detection + hand-region preprocessing."""
import cv2
import numpy as np
import mediapipe as mp

IMG_SIZE = 64
GESTURES = ["Hello", "Stop", "Thank You", "Rock", "Perfect"]  # class order matters

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def create_hands(max_hands=1):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )


def get_hand_crop(frame, results, padding=25):
    """Crop a square region around the detected hand and preprocess it.

    Returns (processed_image, bbox) or (None, None) if no hand found.
    processed_image: float32, shape (64, 64, 3), values 0-1
    """
    if not results.multi_hand_landmarks:
        return None, None

    h, w, _ = frame.shape
    lm = results.multi_hand_landmarks[0].landmark
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # make the box square so the hand is not stretched when resized
    side = max(x_max - x_min, y_max - y_min) + 2 * padding
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x2, y2 = min(cx + side // 2, w), min(cy + side // 2, h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))        # resize to fixed size
    crop = crop.astype("float32") / 255.0                # normalize 0-1
    return crop, (x1, y1, x2, y2)