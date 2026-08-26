import os
import sys
import numpy as np
import tensorflow as tf

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from configs.config import IMAGE_SIZE


# -------------------------------------------------------
# Image Preprocessing
# -------------------------------------------------------

def preprocess_image(image_path):
    """
    Loads and preprocesses a single image for prediction.
    """

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    # Normalize (same as training)
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    image = preprocess_image(image_path)

    print("\nImage Loaded Successfully!")
    print("Image Shape :", image.shape)