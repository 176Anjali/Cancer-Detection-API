import os
import sys
import tensorflow as tf

# -------------------------------------------------------
# Add Project Root
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from configs.config import IMAGE_SIZE, BATCH_SIZE, SEED

# -------------------------------------------------------
# Dataset Paths
# -------------------------------------------------------

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

# -------------------------------------------------------
# Dataset Loader
# -------------------------------------------------------

def load_dataset(path, shuffle=False):

    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int"
    )


# -------------------------------------------------------
# Main Loader Function
# -------------------------------------------------------

def load_datasets():

    train_ds = load_dataset(TRAIN_DIR, shuffle=True)

    val_ds = load_dataset(VAL_DIR)

    test_ds = load_dataset(TEST_DIR)

    class_names = train_ds.class_names

    normalizer = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalizer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalizer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalizer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# -------------------------------------------------------
# Test Loader
# -------------------------------------------------------

if __name__ == "__main__":

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("\nDatasets Loaded Successfully")
    print("Classes:", class_names)

    for images, labels in train_ds.take(1):
        print("Images Shape :", images.shape)
        print("Labels Shape :", labels.shape)