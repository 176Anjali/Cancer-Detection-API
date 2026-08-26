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
    f1_score,
)

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from configs.config import IMAGE_SIZE, BATCH_SIZE

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

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

print("Model Path :", MODEL_PATH)
print("Model Exists :", os.path.exists(MODEL_PATH))

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel Loaded Successfully!")

print("\n" + "=" * 60)
print("Loading Test Dataset")
print("=" * 60)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = test_ds.class_names

print("\nClasses :", class_names)

# -------------------------------------------------------
# SAME preprocessing as training
# -------------------------------------------------------

normalizer = tf.keras.layers.Rescaling(1.0 / 255)

test_ds = test_ds.map(
    lambda x, y: (normalizer(x), y)
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------

print("\n" + "=" * 60)
print("Evaluating Model")
print("=" * 60)

results = model.evaluate(
    test_ds,
    verbose=1
)

print("\nLoss :", results[0])
print("Accuracy :", results[1])

# -------------------------------------------------------
# Predictions
# -------------------------------------------------------

print("\n" + "=" * 60)
print("Predicting")
print("=" * 60)

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds],
    axis=0
)

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

print("\nAccuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 Score :", f1)

# -------------------------------------------------------
# Classification Report
# -------------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.json"),
    "w"
) as f:
    json.dump(report, f, indent=4)

# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

df.to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)

plt.figure(figsize=(8,6))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

print("\nEvaluation Completed Successfully!")