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

print("\n" + "=" * 60)
print("Loading Test Dataset")
print("=" * 60)

from preprocessing.data_loader import load_datasets

_, _, test_ds, class_names = load_datasets()

print("\nClasses:")
print(class_names)

print(f"\nTotal Classes : {len(class_names)}")

print("\n" + "=" * 60)
print("Predicting Test Images")
print("=" * 60)

# Predict probabilities
predictions = model.predict(test_ds)

# Convert probabilities to class indices
y_pred = np.argmax(predictions, axis=1)

# Collect actual labels
y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds],
    axis=0
)

print("\nPrediction Completed!")
print(f"Total Predictions : {len(y_pred)}")
print(f"Total Actual Labels : {len(y_true)}")

print("\n" + "=" * 60)
print("Model Evaluation")
print("=" * 60)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n" + "=" * 60)
print("Keras Evaluation")
print("=" * 60)

results = model.evaluate(test_ds, verbose=1)

print("\nMetrics Names:", model.metrics_names)
print("Results:", results)