"""Step 3: Real-time sign language recognition with text + speech output."""
import time
import queue
import threading
from collections import deque, Counter

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import create_hands, get_hand_crop, mp_draw, mp_hands, GESTURES

MODEL_PATH = "sign_model.h5"
CONF_THRESHOLD = 0.80     # minimum confidence to accept a prediction
STABLE_FRAMES = 10        # same gesture must appear this many frames in a row
SPEAK_COOLDOWN = 2.5      # seconds before speaking again
ENABLE_SPEECH = True

# ---------- Speech (separate thread so video never freezes) ----------
speech_queue = queue.Queue()


def speech_worker():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
    except Exception as e:
        print("TTS unavailable:", e)
        return
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()


if ENABLE_SPEECH:
    threading.Thread(target=speech_worker, daemon=True).start()

# ---------- Main ----------
model = load_model(MODEL_PATH)
hands = create_hands()
cap = cv2.VideoCapture(0)

history = deque(maxlen=STABLE_FRAMES)
current_text = "..."
last_spoken, last_spoken_time = None, 0
prev_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera disconnected")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    crop, bbox = get_hand_crop(frame, results)

    if crop is not None:
        probs = model.predict(np.expand_dims(crop, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        history.append(idx if conf >= CONF_THRESHOLD else -1)

        # stability check: same label for all last N frames
        if len(history) == STABLE_FRAMES:
            label, count = Counter(history).most_common(1)[0]
            if label != -1 and count == STABLE_FRAMES:
                current_text = GESTURES[label]
                now = time.time()
                if (ENABLE_SPEECH and current_text != last_spoken
                        and now - last_spoken_time > SPEAK_COOLDOWN):
                    speech_queue.put(current_text)
                    last_spoken, last_spoken_time = current_text, now

        cv2.putText(frame, f"{GESTURES[idx]} ({conf * 100:.0f}%)",
                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    else:
        history.clear()
        last_spoken = None   # allow same gesture to be spoken again after hand leaves
        cv2.putText(frame, "No hand detected", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # FPS
    now = time.time()
    fps = 1 / max(now - prev_time, 1e-6)
    prev_time = now

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
    cv2.putText(frame, f"Output: {current_text}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()