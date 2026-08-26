import os
import sys

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from prediction.predictor import predict_image


def predict(image_path: str):
    """
    Predict cancer type from uploaded image.
    """

    result = predict_image(image_path)

    return {
        "success": True,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "message": result["message"]
    }