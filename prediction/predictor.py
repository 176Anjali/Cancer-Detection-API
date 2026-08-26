import os
import sys
import numpy as np
import tensorflow as tf

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from prediction.preprocess import preprocess_image
from prediction.labels import get_prediction

# -------------------------------------------------------
# Model Path
# -------------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final",
    "leukemia_classifier.keras"
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

print("Loading Leukemia Detection Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully!\n")

# -------------------------------------------------------
# Predict Function
# -------------------------------------------------------

def predict_image(image_path):
    """
    Predict leukemia class from a single image.
    """

    image = preprocess_image(image_path)

    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions)

    confidence = float(np.max(predictions) * 100)

    result = get_prediction(predicted_index)

    return {
        "prediction": result["class"],
        "confidence": round(confidence, 2),
        "message": result["message"]
    }


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print("\nPrediction Result")
    print("-" * 40)

    print("Prediction :", result["prediction"])
    print("Confidence :", f"{result['confidence']} %")
    print("Message    :", result["message"])