
# 🎭 Face Mask Detection AI (v1.0 Vision PRO)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-00C853.svg)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer--Vision-5C3EE8.svg)](https://opencv.org/)

A professional-grade Computer Vision system designed to detect and classify facial masks in real-time. This project leverages **MediaPipe** for ultra-fast face detection and **TensorFlow/Keras** for binary mask classification.

## 🚀 Innovation & Tech
Traditional face detection often fails when the face is occluded by a mask. This project uses a **multi-stage pipeline**:
1.  **Face Detection**: MediaPipe identifies face regions with high confidence even with occlusions.
2.  **Preprocessing**: Dynamic ROI (Region of Interest) cropping and normalization for CNN input.
3.  **Classification**: A specialized deep learning model trained to distinguish between "Masked" and "No Mask" faces.

## 🌐 Use Cases
*   **Public Safety**: Automated compliance monitoring in transport hubs.
*   **Healthcare**: Ensuring protocol adherence in clinical environments.
*   **Security**: Integrated biometric access control.

## 🛠️ Technical Deep Dive
*   **Real-time Stream**: Optimized for low-latency webcam processing (~10-15 FPS on standard hardware).
*   **API Integrity**: Backend ensures image validation before processing via FastAPI.
*   **UI/UX**: Cyberpunk-themed dark dashboard with live HUD (Heads-Up Display) overlays.

## 📦 Tech Stack
*   **Vision Core**: MediaPipe, OpenCV.
*   **Deep Learning**: TensorFlow, Keras.
*   **Infrastructure**: FastAPI (Asynchronous ML Inference).
*   **Analytics**: NumPy, Image ROI Processing.

## ⚙️ Quick Start
1. **Dependencies**: `pip install -r requirements.txt`
2. **Server**: `python main.py`
3. **Web Interface**: Open `frontend/index.html` and grant webcam permissions.

## ⚖️ Ethical AI
This project follows privacy-by-design principles:
*   No facial data is stored locally or on a server.
*   Processing occurs in volatile memory.
*   Transparent detection metrics provided to the user.
