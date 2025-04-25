import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import queue
from streamlit_autorefresh import st_autorefresh

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

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmark_features(results):
    features = np.zeros(126)
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                features[hand_idx * 63 + lm_idx * 3] = lm.x
                features[hand_idx * 63 + lm_idx * 3 + 1] = lm.y
                features[hand_idx * 63 + lm_idx * 3 + 2] = lm.z
    return features

# Thread-safe queue for predictions
prediction_queue = queue.Queue()

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

            pred_probs = model.predict(features, verbose=0)
            pred_idx = np.argmax(pred_probs)
            gesture = GESTURE_LABELS[pred_idx]
            confidence = pred_probs[0][pred_idx]

            # Send prediction to main thread safely
            prediction_queue.put((gesture, confidence, time.strftime("%H:%M:%S")))

            # Display prediction on frame
            text = f"{gesture} ({confidence*100:.1f}%)"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        return img

# Initialize session state for history and metrics
if 'history' not in st.session_state:
    st.session_state.history = []

if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'total_predictions': 0,
        'avg_confidence': 0.0
    }

# Reverse gesture lookup function
def search_gestures(query):
    return [g for g in GESTURE_LABELS if query.lower() in g.lower()]

# Auto-refresh every 1 second for live updates (max 100 times)
st_autorefresh(interval=1000, limit=100, key="refresh")

st.title("Real-Time Hand Gesture Recognition")
st.write("Using MediaPipe Hands + MLP model")

# Sidebar: Reverse Gesture Lookup
st.sidebar.header("Search Gestures")
search_term = st.sidebar.text_input("Enter gesture name to search:")
if search_term:
    matches = search_gestures(search_term)
    if matches:
        st.sidebar.success(f"Found {len(matches)} matching gesture(s):")
        for m in matches:
            st.sidebar.write(f"- {m}")
            # Uncomment and add images if available
            # st.sidebar.image(f"images/{m}.png", width=100)
    else:
        st.sidebar.error("No matching gestures found.")

# Sidebar placeholders for history and metrics
st.sidebar.header("Prediction History (Last 5)")
history_placeholder = st.sidebar.empty()

st.sidebar.header("Performance Metrics")
metrics_placeholder = st.sidebar.empty()

# Run webcam streamer with audio disabled
webrtc_ctx = webrtc_streamer(
    key="gesture-recognition",
    video_transformer_factory=GestureRecognitionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# Process predictions from queue and update session state
if webrtc_ctx.video_transformer:
    while not prediction_queue.empty():
        gesture, confidence, timestamp = prediction_queue.get()
        st.session_state.history.append((gesture, confidence, timestamp))
        st.session_state.metrics['total_predictions'] += 1
        total = st.session_state.metrics['total_predictions']
        prev_avg = st.session_state.metrics['avg_confidence']
        st.session_state.metrics['avg_confidence'] = (prev_avg * (total - 1) + confidence) / total

# Update sidebar with latest history and metrics
history_text = ""
for gesture, conf, ts in reversed(st.session_state.history[-5:]):
    history_text += f"{ts} - {gesture} ({conf*100:.1f}%)\n"
