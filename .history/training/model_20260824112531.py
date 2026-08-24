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