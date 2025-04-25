import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Replace with your actual gesture labels in the correct order
GESTURE_LABELS = [
    "call", "care", "dislike", "goodbye", "good luck", "heart", "hello", "help!",
    "illuminati", "like", "losers", "no", "okay", "peace", "protest", "punch",
    "relax", "rock_on", "silent", "sorry", "stop", "water", "yes"
]

def main():
    # Load the trained model
    print("Loading model...")
    model = load_model('gesture_model.h5')

    # Load test dataset
    print("Loading test data...")
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')  # one-hot encoded labels

    # Convert one-hot labels to class indices
    y_true = np.argmax(y_test, axis=1)

    # Get model predictions (probabilities)
    print("Predicting on test data...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Classification report: precision, recall, f1-score per class
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=GESTURE_LABELS)
    print(report)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Plot confusion matrix heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_LABELS,
                yticklabels=GESTURE_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
