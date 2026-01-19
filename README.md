
# Face Mask Detection AI

A real-world computer vision project that detects faces and classifies them as "Masked" or "No Mask" in real-time using a deep learning model.

## Features
- **Real-time Detection**: Uses MediaPipe for robust face detection.
- **Deep Learning**: MobileNetV2-based classification for high accuracy.
- **Webcam Integration**: Live feed directly in the browser.
- **Premium UI**: Dark-themed dashboard with live statistics and bounding box overlays.

## Setup & Installation

### 1. Backend
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8001`.

### 2. Frontend
Simply open `frontend/index.html` in your browser. Ensure the backend is running.

## Technology Stack
- **Python**: TensorFlow/Keras, OpenCV, MediaPipe.
- **API**: FastAPI, Uvicorn.
- **Frontend**: HTML5, CSS3, JavaScript.

## Ethical AI
This project is designed for educational and safety purposes. Ensure compliance with local privacy laws regarding face detection and data storage.

## License
MIT
