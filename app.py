import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import tempfile

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["Real", "Fake"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('test_model.h5')

@st.cache_resource
def load_mtcnn():
    return MTCNN()

def preprocess_video(video_file):
    frames_list = []
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_file.read())
        video_reader = cv2.VideoCapture(tmpfile.name)
        detector = load_mtcnn()
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_sample = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)

        for frame_number in frames_to_sample:
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video_reader.read()
            if not success:
                break

            faces = detector.detect_faces(frame)
            if faces:
                x, y, width, height = faces[0]['box']
                face_frame = frame[y:y+height, x:x+width]
                resized_frame = cv2.resize(face_frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                normalized_frame = resized_frame / 255.0
                frames_list.append(normalized_frame)
            
            if len(frames_list) == SEQUENCE_LENGTH:
                break

        video_reader.release()
    
    # If we couldn't extract enough frames, pad the list
    while len(frames_list) < SEQUENCE_LENGTH:
        frames_list.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    return np.array([frames_list])

# Load the model
model = load_model()

st.title('Deepfake Detection App')

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button('Detect Deepfake'):
        with st.spinner('Processing video...'):
            preprocessed_video = preprocess_video(uploaded_file)
            prediction = model.predict(preprocessed_video)
            result = CLASSES_LIST[int(prediction[0][0] > 0.5)]

        st.write(f"The video is predicted to be: {result}")
        st.write(f"Confidence: {prediction[0][0]:.2f}")

st.write("Note: This app uses a pre-trained model and may not be 100% accurate.")