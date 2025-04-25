import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('gesture_model.h5')

# Create a dummy input sample (shape: 1 sample, 126 features)
dummy_input = np.zeros((1, 126), dtype=np.float32)

# Predict
predictions = model.predict(dummy_input)
print("Prediction shape:", predictions.shape)
print("Predicted probabilities:", predictions)
