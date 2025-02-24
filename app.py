import streamlit as st
import os
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define paths
MODEL_PATH = "emotion_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Load the model
@st.cache_resource
def load_emotion_model():
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)
    return model_data["model"], model_data["label_encoder"]

model, label_encoder = load_emotion_model()

# Process video for emotion detection
def process_video(video_path, output_path="output_video.mp4", max_frames=400):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face = np.expand_dims(face, axis=0) / 255.0

            predictions = model.predict(face)
            pred_class = np.argmax(predictions)

            # **Fix: Ensure pred_label is a string**
            pred_label = str(label_encoder.inverse_transform([pred_class])[0])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, pred_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    return output_path

# Home Page
def home_page():
    st.markdown(
        "<h1 style='text-align: center;'>📊 Customer Sentiment Detection 🤖</h1>", 
        unsafe_allow_html=True
    )

    # Display Image
    st.image("customer_emotion.jpg", use_container_width=True)

    # App Description
    st.markdown(
        """
        <div style="text-align: center;">
        <p>Ever wondered what your customers *really* think? 🤔</p>
        <p>Our AI-powered emotion detector analyzes video recordings to understand customer sentiment in real-time! 🎥</p>
        <p>Perfect for businesses looking to enhance experiences & improve customer relationships! 🚀</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation Button
    if st.button("🎯 Click here to better understand your customers!"):
        st.session_state.page = "emotion_detector"

# Emotion Detection Page
def emotion_detector():
    st.markdown(
        "<h1 style='text-align: center;'>🎥 Emotion Detector</h1>", 
        unsafe_allow_html=True
    )

    # Back to Home Button
    if st.button("🏠 Go Home"):
        st.session_state.page = "home"

    st.markdown(
        "<p style='text-align: center;'>📂 Upload a video and let AI analyze customer emotions!</p>", 
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_video:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        if st.button("🎯 Predict Now"):
            output_video_path = process_video(video_path)

            # Display the processed video
            st.video(output_video_path)

            # Download button
            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=file,
                    file_name="analyzed_video.mp4",
                    mime="video/mp4"
                )

# Page Navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home_page()
else:
    emotion_detector()

