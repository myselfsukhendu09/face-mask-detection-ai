
import cv2
import numpy as np
import os
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import base64

app = FastAPI(title="Face Mask Detection AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load Mask Classifier (Placeholder for now, or use a simple heuristic if model missing)
# In a real scenario, you'd load a .h5 or .keras model here
MODEL_PATH = "models/mask_classifier.h5"
model = None

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Warning: mask_classifier.h5 not found. Using dummy classifier for demonstration.")

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = tf.keras.preprocessing.image.img_to_array(face_img)
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

@app.post("/detect")
async def detect_mask(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    detections = []
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            
            # Ensure coordinates are within bounds
            x, y = max(0, x), max(0, y)
            w, h = min(w, iw - x), min(h, ih - y)
            
            face_roi = img_rgb[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue

            # Classify mask
            label = "Unknown"
            confidence = 0.0
            
            if model:
                processed_face = preprocess_face(face_roi)
                preds = model.predict(processed_face)
                # Assuming index 0 is 'Mask' and index 1 is 'No Mask'
                mask_prob = preds[0][0]
                label = "Mask" if mask_prob > 0.5 else "No Mask"
                confidence = float(mask_prob if mask_prob > 0.5 else 1 - mask_prob)
            else:
                # Dummy logic: green-ish pixels in the mouth area = mask? 
                # (Just for demo if model is missing)
                label = "Mask (Demo)" 
                confidence = 0.95

            detections.append({
                "bbox": [x, y, w, h],
                "label": label,
                "confidence": confidence
            })
            
    return {"detections": detections}

@app.get("/")
async def root():
    return {"message": "Face Mask Detection API is active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
