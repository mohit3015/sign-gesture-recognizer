"""Step 1: Collect dataset from webcam.

Controls:
  n  -> next gesture class
  p  -> previous gesture class
  s  -> start / stop saving images (auto-capture)
  q  -> quit
"""
import os
import cv2
from utils import create_hands, get_hand_crop, mp_draw, mp_hands, GESTURES, IMG_SIZE

DATASET_DIR = "dataset"
SAMPLES_PER_CLASS = 400

for g in GESTURES:
    os.makedirs(os.path.join(DATASET_DIR, g), exist_ok=True)

cap = cv2.VideoCapture(0)
hands = create_hands()
idx, saving = 0, False

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = GESTURES[idx]
    folder = os.path.join(DATASET_DIR, gesture)
    count = len(os.listdir(folder))

    crop, bbox = get_hand_crop(frame, results)
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        if saving and count < SAMPLES_PER_CLASS:
            img = (crop * 255).astype("uint8")
            cv2.imwrite(os.path.join(folder, f"{count:04d}.jpg"), img)
            count += 1

    if count >= SAMPLES_PER_CLASS:
        saving = False

    cv2.putText(frame, f"Gesture: {gesture}  ({count}/{SAMPLES_PER_CLASS})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "SAVING" if saving else "PAUSED  (s=save n/p=class q=quit)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if saving else (255, 255, 255), 2)
    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        saving = not saving
    elif key == ord("n"):
        idx, saving = (idx + 1) % len(GESTURES), False
    elif key == ord("p"):
        idx, saving = (idx - 1) % len(GESTURES), False

cap.release()
cv2.destroyAllWindows()