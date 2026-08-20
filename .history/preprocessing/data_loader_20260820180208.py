import os
import tensorflow as tf
from configs.config import IMAGE_SIZE, BATCH_SIZE, SEED

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")


def load_dataset(path, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int"
    )


train_ds = load_dataset(TRAIN_DIR, shuffle=True)
val_ds = load_dataset(VAL_DIR)
test_ds = load_dataset(TEST_DIR)

# Save class names before mapping
CLASS_NAMES = train_ds.class_names

# Normalize images
normalizer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalizer(x), y))
val_ds = val_ds.map(lambda x, y: (normalizer(x), y))
test_ds = test_ds.map(lambda x, y: (normalizer(x), y))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

print("\nDatasets Loaded Successfully")
print("Classes:", CLASS_NAMES)

for images, labels in train_ds.take(1):
    print("Images Shape :", images.shape)
    print("Labels Shape :", labels.shape)