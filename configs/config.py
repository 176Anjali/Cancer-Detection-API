import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.0001

NUM_CLASSES = 4

SEED = 42

MODEL_NAME = "MobileNetV2"

MODEL_PATH = os.path.join(BASE_DIR, "models", "final", "leukemia_classifier.keras")