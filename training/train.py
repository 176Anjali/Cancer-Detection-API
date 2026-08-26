import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------------
# Add Project Root
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from preprocessing.data_loader import load_datasets
from training.model import build_model
from training.callbacks import CALLBACKS
from training.metrics import METRICS

from configs.config import (
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH
)

# -------------------------------------------------
# Create folders
# -------------------------------------------------

os.makedirs("models/final", exist_ok=True)
os.makedirs("models/history", exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

train_ds, val_ds, test_ds, class_names = load_datasets()

print("\nClasses :", class_names)

# -------------------------------------------------
# Build Model
# -------------------------------------------------

model = build_model()

# -------------------------------------------------
# Compile Model
# -------------------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=METRICS

)

print("\nModel Compiled Successfully")

# -------------------------------------------------
# Train
# -------------------------------------------------

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=CALLBACKS

)

print("\nFinal Training Accuracy :", history.history["accuracy"][-1])
print("Final Validation Accuracy :", history.history["val_accuracy"][-1])

# -------------------------------------------------
# Save Final Model
# -------------------------------------------------

model.save(MODEL_PATH)

print("\nModel Saved Successfully")

# -------------------------------------------------
# Evaluate
# -------------------------------------------------

results = model.evaluate(test_ds)

print("\nTest Results")

for metric, value in zip(model.metrics_names, results):

    print(f"{metric} : {value:.4f}")

# -------------------------------------------------
# Accuracy Graph
# -------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.grid()

plt.savefig("models/history/accuracy.png")

plt.close()

# -------------------------------------------------
# Loss Graph
# -------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.grid()

plt.savefig("models/history/loss.png")

plt.close()

print("\nTraining Completed Successfully")