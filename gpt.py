import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Gesture labels (must match your training order)
GESTURE_LABELS = [
    "call", "care", "dislike", "goodbye", "good luck", "heart", "hello", "help!",
    "illuminati", "like", "losers", "no", "okay", "peace", "protest", "punch",
    "relax", "rock_on", "silent", "sorry", "stop", "water", "yes"
]

# Load your trained model once
@st.cache_resource
def load_gesture_model():
    model = load_model('gesture_model.h5')
    return model

model = load_gesture_model()

# Load .npy gesture data (1 sample per class assumed)
gesture_data = np.load("dataset")  # Make sure this file exists

# Dictionary: gesture_name → index
gesture_dict = {name: idx for idx, name in enumerate(GESTURE_LABELS)}

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to extract 126-length feature vector from landmarks
def extract_landmark_features(results):
    features = np.zeros(126)
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                features[hand_idx * 63 + lm_idx * 3] = lm.x
                features[hand_idx * 63 + lm_idx * 3 + 1] = lm.y
                features[hand_idx * 63 + lm_idx * 3 + 2] = lm.z
    return features

# Plot sample skeleton from .npy
def plot_sample_gesture(gesture_name):
    index = gesture_dict.get(gesture_name)
    if index is None:
        return None

    sample = gesture_data[index].reshape(42, 3)  # 126 → (42, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Example: {gesture_name}")
    ax.axis('off')

    # Separate hands
    left = sample[:21]
    right = sample[21:]

    def draw_hand(hand, color):
        xs, ys = hand[:, 0], hand[:, 1]
        ax.scatter(xs, ys, color=color)
        for i in range(len(xs)):
            ax.text(xs[i], ys[i], str(i), fontsize=6, color='black')

    draw_hand(left, 'blue')
    draw_hand(right, 'green')

    return fig

# Video frame processor class
class GestureRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        self.prediction = "Waiting for hand..."
        self.confidence = 0.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_landmark_features(results)
            features = features.reshape(1, -1)

            pred_probs = model.predict(features)
            pred_idx = np.argmax(pred_probs)
            self.prediction = GESTURE_LABELS[pred_idx]
            self.confidence = pred_probs[0][pred_idx]

            # Display prediction on frame
            text = f"{self.prediction} ({self.confidence*100:.1f}%)"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            self.prediction = "No hands detected"
            self.confidence = 0.0
            cv2.putText(img, self.prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        return img

# Streamlit UI
st.title("Real-Time Hand Gesture Recognition")
st.write("Using MediaPipe Hands + MLP model")

# Reverse Gesture Lookup UI
st.sidebar.markdown("### 🔁 Reverse Gesture Lookup")
selected_gesture = st.sidebar.selectbox("Select a gesture", GESTURE_LABELS)

if selected_gesture:
    fig = plot_sample_gesture(selected_gesture)
    if fig:
        st.sidebar.pyplot(fig)
    else:
        st.sidebar.warning("Could not find gesture data.")

# Webcam stream
webrtc_streamer(key="gesture-recognition", video_transformer_factory=GestureRecognitionTransformer)
