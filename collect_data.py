import cv2
import mediapipe as mp
import numpy as np
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = 'dataset'
GESTURES = [ "call", "care", "dislike", "goodbye", "good luck", "heart", "hello", "help!",
    "illuminati", "like", "losers", "no", "okay", "peace", "protest", "punch",
    "relax", "rock_on", "silent", "sorry", "stop", "water", "yes"]  # Your 22 gestures

SAMPLES_PER_GESTURE = 1000

def extract_features(multi_hand_landmarks):
    features = np.zeros(126)
    if multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks[:2]):
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                base_idx = hand_idx * 63 + lm_idx * 3
                features[base_idx] = lm.x
                features[base_idx + 1] = lm.y
                features[base_idx + 2] = lm.z
    return features

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

cap = cv2.VideoCapture(0)

try:
    for gesture in GESTURES:
        print(f"\nPrepare to record gesture: {gesture}")
        print("Get ready... Starting in 5 seconds")
        time.sleep(5)
        print(f"Press 's' to START/STOP recording for gesture '{gesture}'. Press ESC to exit.")

        gesture_path = os.path.join(DATA_PATH, gesture)
        if not os.path.exists(gesture_path):
            os.makedirs(gesture_path)

        collected = 0
        collecting = False  # Control flag for start/stop

        while collected < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Draw landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show collection status on frame
            status_text = f"Gesture: {gesture} | Collected: {collected}/{SAMPLES_PER_GESTURE} | "
            status_text += "Recording..." if collecting else "Paused - Press 's' to start"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if collecting else (0, 0, 255), 2)

            cv2.imshow("PalmX Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key to quit
                print("Data collection interrupted by user.")
                raise KeyboardInterrupt

            elif key == ord('s'):
                collecting = not collecting
                print("Recording started." if collecting else "Recording paused.")

            # Save data only if collecting and landmarks detected
            if collecting and results.multi_hand_landmarks:
                features = extract_features(results.multi_hand_landmarks)
                np.save(os.path.join(gesture_path, f"{collected}.npy"), features)
                collected += 1

    print("\nData collection completed successfully!")

except KeyboardInterrupt:
    print("\nData collection stopped.")

finally:
    cap.release()
    cv2.destroyAllWindows()
