import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)

from configs.config import IMAGE_SIZE, BATCH_SIZE

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final",
    "leukemia_classifier.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Loading Trained Model")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel Loaded Successfully!")
print(model.summary())