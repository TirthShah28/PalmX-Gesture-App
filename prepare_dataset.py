import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = 'dataset'  # Folder where your gesture folders are
GESTURES = [
    "call", "care", "dislike", "goodbye", "good luck", "heart", "hello", "help!",
    "illuminati", "like", "losers", "no", "okay", "peace", "protest", "punch",
    "relax", "rock_on", "silent", "sorry", "stop", "water", "yes"
]

features = []
labels = []

print("Loading dataset...")

for idx, gesture in enumerate(GESTURES):
    gesture_folder = os.path.join(DATA_PATH, gesture)
    files = os.listdir(gesture_folder)
    print(f"Loading {len(files)} samples for gesture '{gesture}'")
    for file in files:
        data = np.load(os.path.join(gesture_folder, file))
        features.append(data)
        labels.append(idx)

features = np.array(features)
labels = np.array(labels)

print(f"Total samples: {features.shape[0]}")

# One-hot encode labels
labels_cat = to_categorical(labels, num_classes=len(GESTURES))

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels_cat, test_size=0.2, random_state=42, stratify=labels)

# Save prepared datasets
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Dataset prepared and saved.")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
