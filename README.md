# 🖐️ Real-Time Sign Language Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-blue?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![TTS](https://img.shields.io/badge/pyttsx3-Speech%20Synthesis-yellow)](https://pypi.org/project/pyttsx3/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

An assistive Human-Computer Interaction (HCI) computer vision system that captures hand gestures from live webcam input, classifies them using a custom Convolutional Neural Network (CNN), and provides both **real-time on-screen text** and **synthesized voice speech output**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Supported Gestures](#-supported-gestures)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [CNN Model Architecture](#-cnn-model-architecture)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🔍 Project Overview

Sign language is the primary communication medium for millions of hearing and speech-impaired individuals. However, the lack of widespread fluency among the general public creates communication barriers in everyday scenarios.

This project delivers a real-time, vision-based sign language translation system:
1. Detects and tracks 21 hand landmarks in real time using **Google MediaPipe**.
2. Dynamically isolates, crops, and squares the region of interest (ROI) around the active hand.
3. Classifies gestures using a custom **Convolutional Neural Network (CNN)** trained with geometric data augmentation.
4. Uses a temporal consistency buffer and confidence thresholding to prevent prediction flickering.
5. Emits translated speech asynchronously via a non-blocking background **Text-to-Speech (TTS)** engine.

---

## ✨ Key Features

- **MediaPipe Hand Landmark Tracking:** Tracks 21 3D hand coordinates with low latency, providing high tracking precision across varying hand orientations.
- **Aspect-Ratio Preserving Square Cropping:** Custom bounding box algorithm adds adaptive padding and forces square dimensions, preventing gesture stretching and distortion upon 64x64 resizing.
- **Deep CNN Classification:** Multi-stage feature extraction with Conv2D layers, Max-Pooling, and Dropout regularization.
- **Flicker-Free Temporal Stability:** Uses a `deque` sliding buffer requiring 10 consecutive matching frames above an **80% confidence threshold** before registering a gesture change.
- **Non-Blocking Multithreaded Speech:** Runs `pyttsx3` voice synthesis on an isolated worker thread through a `queue.Queue`, ensuring 30+ FPS video streams never freeze during audio playback.
- **Integrated Dataset Collection Tool:** Includes an interactive data acquisition pipeline to record custom gesture classes.

---

## ✋ Supported Gestures

The default model is trained on 5 fundamental communicative sign gestures:

| Gesture Name | Meaning / Context | Trigger Output |
| :--- | :--- | :--- |
| **Hello** | Casual Greeting | Text display + *"Hello"* audio |
| **Stop** | Halt / Attention command | Text display + *"Stop"* audio |
| **Thank You** | Gratitude expression | Text display + *"Thank You"* audio |
| **Rock** | Affirmative sign | Text display + *"Rock"* audio |
| **Perfect** | OK / Agreement sign | Text display + *"Perfect"* audio |

> *Easily expandable: Add new gesture labels in `utils.py` and run the collection script.*

---

## 🔄 System Architecture & Pipeline

```
[ Webcam Video Stream (OpenCV) ]
              │
              ▼
[ MediaPipe Hands (21 Landmark Detection) ]
              │
              ▼
[ Square Aspect-Ratio ROI Hand Cropping (64x64, 0-1 Norm) ]
              │
              ▼
[ Custom CNN Classifier (sign_model.h5) ]
              │
              ▼
[ Confidence Thresholding (> 80%) & 10-Frame Stability Filter ]
              │
       ┌──────┴─────────────────────────┐
       ▼                                ▼
[ Visual Feedback Overlay ]   [ Asynchronous Audio Speech ]
(BBox + Text Label on Frame)   (pyttsx3 Thread Queue Worker)
```

---

## 🏗️ CNN Model Architecture

The neural network is optimized for fast embedded inference while preserving subtle finger configuration features:

| Layer | Type | Specifications | Output Dimensions |
| :--- | :--- | :--- | :--- |
| **Input** | InputLayer | RGB image input | `(64, 64, 3)` |
| **Block 1** | Conv2D + MaxPool | 32 filters (3x3), ReLU, Same padding | `(32, 32, 32)` |
| **Block 2** | Conv2D + MaxPool | 64 filters (3x3), ReLU, Same padding | `(16, 16, 64)` |
| **Block 3** | Conv2D + MaxPool | 128 filters (3x3), ReLU, Same padding | `(8, 8, 128)` |
| **Flatten** | Flatten | Dense vector flattening | `(8192)` |
| **Dense 1** | Dense + Dropout | 128 units, ReLU, **Dropout = 0.50** | `(128)` |
| **Output** | Dense | 5 units (Target Classes), Softmax | `(5)` |

* **Loss Function:** `categorical_crossentropy`
* **Optimizer:** Adam (`adam`)
* **Callbacks:** `EarlyStopping` (patience=6), `ModelCheckpoint` (saving best `val_accuracy`)
* **Data Augmentation:** Rotation ($\pm 15^\circ$), Zoom ($10\%$), Width/Height shifts ($10\%$), Horizontal flip.

---

## 📁 Repository Structure

```
sign-gesture-recognizer/
│
├── dataset/                  # Collected dataset split into gesture subfolders
│   ├── Hello/
│   ├── Stop/
│   ├── Thank You/
│   ├── Rock/
│   └── Perfect/
│
├── utils.py                  # MediaPipe hand tracker & square crop preprocessor
├── collect_data.py           # Interactive webcam dataset collector tool
├── train_model.py            # CNN training, augmentation & evaluation script
├── recognize.py              # Real-time webcam inference + TTS voice engine
├── sign_model.h5             # Trained TensorFlow/Keras CNN model weights
├── training_graphs.png       # Training & validation accuracy/loss curves
├── requirements.txt          # Project dependencies
└── README.md                 # System documentation
```

---

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/mohit3015/sign-gesture-recognizer.git
cd sign-gesture-recognizer
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Live Real-Time Recognition (Ready to use)
Run the real-time inference app with your webcam:
```bash
python recognize.py
```
* Press **`q`** to quit the live feed.
* When a supported gesture is held steady for $\approx 10$ frames, the system will speak the gesture aloud.

---

### Option 2: Retraining or Adding Custom Gestures

1. **Edit gesture classes in `utils.py`:**
   ```python
   GESTURES = ["Hello", "Stop", "Thank You", "Rock", "Perfect", "YourGesture"]
   ```

2. **Collect custom training images:**
   ```bash
   python collect_data.py
   ```
   * Press **`s`** to start/pause saving frames ($\approx 400$ images per class).
   * Press **`n`** to advance to the next gesture class.

3. **Train the CNN model:**
   ```bash
   python train_model.py
   ```
   * Automatically trains with data augmentation, evaluates confusion matrix, and saves `sign_model.h5`.

---

## 👨‍💻 Author

**Mohit Kumar**  
*B.Tech CSE (AI & ML) | Computer Vision & Deep Learning Specialist*

- 🌐 **Portfolio:** [mohit3015.github.io/Mohit-Portfolio](https://mohit3015.github.io/Mohit-Portfolio/)
- 🐙 **GitHub:** [@mohit3015](https://github.com/mohit3015)
- 💼 **LinkedIn:** [Mohit Kumar](https://www.linkedin.com/in/mohit-kumar-9a33412b3/)
- 📧 **Email:** [mk5972345@gmail.com](mailto:mk5972345@gmail.com)

---
⭐ *If you find this project helpful, please consider starring the repository!*
