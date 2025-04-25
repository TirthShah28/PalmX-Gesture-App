import numpy as np
import os
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Configuration
DATA_PATH = 'dataset'  # Folder containing gesture subfolders with .npy files
GESTURES = [
    "call", "care", "dislike", "goodbye", "good luck", "heart", "hello", "help!",
    "illuminati", "like", "losers", "no", "okay", "peace", "protest", "punch",
    "relax", "rock_on", "silent", "sorry", "stop", "water", "yes"
]
SAMPLES_PER_GESTURE = 1000  # Adjust if different
INPUT_FEATURES = 126        # 2 hands × 21 landmarks × 3 coords

def load_dataset():
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
    print(f"Total samples loaded: {features.shape[0]}")
    return features, labels

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load data
    X, y = load_dataset()
    
    # One-hot encode labels
    y_cat = to_categorical(y, num_classes=len(GESTURES))
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    
    # Build model
    model = build_model(INPUT_FEATURES, len(GESTURES))
    
    # Setup early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop]
    )
    
    # Save model
    model.save('gesture_model.h5')
    print("Model trained and saved as gesture_model.h5")

if __name__ == "__main__":
    main()
