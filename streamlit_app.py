
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from PIL import Image

# Set page config
st.set_page_config(page_title="Vision AI | Face Mask Detector", page_icon="🎭")

st.title("🎭 Face Mask Detection AI")
st.markdown("---")

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load Model
MODEL_PATH = "models/mask_classifier.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    st.warning("Model file (mask_classifier.h5) not found. Displaying face detection only.")

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = tf.keras.preprocessing.image.img_to_array(face_img)
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

img_file = st.camera_input("Take a photo to detect mask")

if img_file is not None:
    # Convert file to opencv image
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    results = face_detection.process(img_rgb)
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = cv2_img.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            
            x, y = max(0, x), max(0, y)
            w, h = min(w, iw - x), min(h, ih - y)
            
            face_roi = img_rgb[y:y+h, x:x+w]
            
            label = "Face Detected"
            color = (255, 255, 0) # Yellow for detection
            
            if model and face_roi.size > 0:
                processed_face = preprocess_face(face_roi)
                preds = model.predict(processed_face)
                mask_prob = preds[0][0]
                label = "Mask" if mask_prob > 0.5 else "No Mask"
                color = (0, 255, 0) if mask_prob > 0.5 else (255, 0, 0)
                
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 4)
            cv2.putText(img_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
    st.image(img_rgb, caption="Processed Image", use_container_width=True)
    
    if not results.detections:
        st.info("No face detected. Please adjust your position.")
