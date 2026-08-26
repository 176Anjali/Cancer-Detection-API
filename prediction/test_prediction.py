import os
import sys

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from prediction.predictor import predict_image

# -------------------------------------------------------
# Test Prediction
# -------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Leukemia Image Prediction")
    print("=" * 60)

    image_path = input("\nEnter image path: ").strip()

    if not os.path.exists(image_path):
        print("\nError: Image not found!")
        exit()

    result = predict_image(image_path)

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']:.2f}%")
    print(f"Message    : {result['message']}")