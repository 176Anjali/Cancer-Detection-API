Thursday 6:06 PM

Awesome! 🚀 You've successfully completed Day 2. From now on, we're entering the Machine Learning phase of the project.

🗓️ Day 3: Build the Leukemia Classification Model

Today, we'll create a production-ready training pipeline instead of writing everything in one file.

🎯 Goal of Day 3

Train a MobileNetV2 model that classifies:

all_benign
all_early
all_pre
all_pro

At the end of today, you'll have:

models/
    leukemia_classifier.keras
Step 1: Create the Project Structure

Create a new folder:

training/

Inside it:

training/
│
├── __init__.py
├── model.py
├── train.py
├── callbacks.py
├── metrics.py
└── evaluate.py
Step 2: Update Project Structure

Your project should now look like:

Cancer-Detection-API/
│
├── app/
│
├── configs/
│     config.py
│
├── dataset/
│
├── models/
│
├── preprocessing/
│     data_loader.py
│
├── scripts/
│
├── training/
│     model.py
│     train.py
│     callbacks.py
│     metrics.py
│     evaluate.py
│
├── requirements.txt
│
└── README.md
Step 3: Update config.py

Replace it with:

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.0001

NUM_CLASSES = 4

SEED = 42

MODEL_NAME = "MobileNetV2"

MODEL_PATH = "models/leukemia_classifier.keras"
Step 4: Install Missing Packages

Run:

pip install scikit-learn matplotlib
Step 5: Build the Model

We'll use

✅ MobileNetV2

because

Fast
Lightweight
High Accuracy
Excellent for medical images
Easy to expand to all 8 cancers later
Today's Files

We'll build them in this order:

1. model.py          ⭐⭐⭐⭐⭐
2. callbacks.py
3. metrics.py
4. train.py
5. evaluate.py
Don't write everything at once

We're going to build each file carefully.

We'll first create

training/model.py

This is the heart of the entire project.

It will contain

MobileNetV2
Transfer Learning
Dropout
Batch Normalization
Dense Layers
Softmax Output
⭐ Today's Milestone

By the end of Day 3 you should be able to run:

python training/train.py

and see output similar to:

Epoch 1/20

accuracy: 0.89

val_accuracy: 0.92

and after training finishes:

models/
    leukemia_classifier.keras

will be created automatically.

📌 One Important Recommendation Before We Start

Since you're building this as a professional API project, I suggest one small improvement now that will save you work later.

Instead of saving just the model, let's organize the project like this:

models/
│
├── checkpoints/
├── final/
└── history/
checkpoints/ → best model during training
final/ → final production model used by the API
history/ → training history, graphs, and metrics

This structure scales well when you later add the other 7 cancer types.

✅ Next

We'll start with training/model.py, where we'll build a clean MobileNetV2 transfer learning architecture for your leukemia classifier.

ok understood next

Perfect. From now on, I'm going to guide you like you're working in a real AI company. We'll build one file at a time, test it, commit it, and then move on.

🗓️ Day 3 - Step 1: Build training/model.py
🎯 Goal

Create a Transfer Learning Model using MobileNetV2.

Instead of training from scratch, we'll use a pretrained model (ImageNet) and teach it to classify the 4 leukemia classes.

The architecture will be:

Input Image (224x224x3)
        │
        ▼
Data Augmentation
        │
        ▼
MobileNetV2 (Pretrained)
        │
        ▼
Global Average Pooling
        │
        ▼
Batch Normalization
        │
        ▼
Dropout
        │
        ▼
Dense(128)
        │
        ▼
Dropout
        │
        ▼
Dense(4, Softmax)
Step 1

Open

training/model.py

Paste this complete code.

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2

from configs.config import IMAGE_SIZE, NUM_CLASSES


def build_model():

    # ----------------------------
    # Input Layer
    # ----------------------------

    inputs = layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # ----------------------------
    # Data Augmentation
    # ----------------------------

    data_augmentation = tf.keras.Sequential([

        layers.RandomFlip("horizontal"),

        layers.RandomRotation(0.10),

        layers.RandomZoom(0.10),

        layers.RandomContrast(0.10)

    ])

    x = data_augmentation(inputs)

    # ----------------------------
    # MobileNetV2 Base Model
    # ----------------------------

    base_model = MobileNetV2(

        weights="imagenet",

        include_top=False,

        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    )

    # Freeze pretrained weights
    base_model.trainable = False

    x = base_model(x, training=False)

    # ----------------------------
    # Classification Head
    # ----------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.30)(x)

    x = layers.Dense(

        128,

        activation="relu"

    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(

        NUM_CLASSES,

        activation="softmax"

    )(x)

    model = models.Model(inputs, outputs)

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()
Step 2

Run:

python training/model.py
Expected output

You'll see TensorFlow download MobileNetV2 weights the first time (internet required).

Then you'll see a model summary ending with something like:

Total params: ...
Trainable params: ...
Non-trainable params: ...

The key point is:

Non-trainable parameters → MobileNetV2 backbone (frozen)
Trainable parameters → Your custom classification head
Why are we freezing the base model?

Your dataset has 20,000 images, which is good, but training all 2+ million MobileNetV2 parameters immediately can lead to:

Longer training times
Overfitting
More CPU/GPU usage

We'll first train only the classifier head.

Later (Day 4), we'll unfreeze the top MobileNetV2 layers for fine-tuning and improve accuracy.

After this step

We'll create:

training/callbacks.py

This will automatically:

Save the best model
Stop training when it stops improving
Reduce the learning rate when validation performance plateaus

These are standard practices in professional deep learning projects.

Complete model.py, run it, and paste the output (or any errors). Then we'll continue to the next file.

(venv) d:\Cancer-Detection-API\training>python -u "d:\Cancer-Detection-API\training\model.py"
2026-08-24 11:26:19.057186: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0.
2026-08-24 11:26:20.090588: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0.
Traceback (most recent call last):
  File "d:\Cancer-Detection-API\training\model.py", line 7, in <module>
    from configs.config import IMAGE_SIZE, NUM_CLASSES
ModuleNotFoundError: No module named 'configs'

This is the same import issue you had with data_loader.py. The good news is it's easy to fix.

Option 1 (Recommended): Add the project root to sys.path

At the top of training/model.py, replace the imports with this:

import os
import sys

# Add project root to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2

from configs.config import IMAGE_SIZE, NUM_CLASSES


def build_model():

    # ----------------------------
    # Input Layer
    # ----------------------------

    inputs = layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # ----------------------------
    # Data Augmentation
    # ----------------------------

    data_augmentation = tf.keras.Sequential([

        layers.RandomFlip("horizontal"),

        layers.RandomRotation(0.10),

        layers.RandomZoom(0.10),

        layers.RandomContrast(0.10)

    ])

    x = data_augmentation(inputs)

    # ----------------------------
    # MobileNetV2 Base Model
    # ----------------------------

    base_model = MobileNetV2(

        weights="imagenet",

        include_top=False,

        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)

    )

    # Freeze pretrained weights
    base_model.trainable = False

    x = base_model(x, training=False)

    # ----------------------------
    # Classification Head
    # ----------------------------

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.30)(x)

    x = layers.Dense(

        128,

        activation="relu"

    )(x)

    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(

        NUM_CLASSES,

        activation="softmax"

    )(x)

    model = models.Model(inputs, outputs)

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()