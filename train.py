
import tensorflow as tf
import numpy as np
import os

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------------------
# PROJECT CONFIGURATION
# --------------------------------

PROJECT_NAME = "Fake Currency Detection System"

DATASET_PATH = "dataset"
MODEL_PATH = "currency_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

print("=" * 50)
print(PROJECT_NAME)
print("CNN Model Training Started")
print("=" * 50)

# --------------------------------
# CHECK DATASET
# --------------------------------

fake_folder = Path(DATASET_PATH) / "fake"
genuine_folder = Path(DATASET_PATH) / "genuine"

if not fake_folder.exists() or not genuine_folder.exists():
    raise FileNotFoundError(
        "Dataset folders missing! Create dataset/fake "
        "and dataset/genuine first."
    )

fake_count = len(list(fake_folder.glob("*")))
genuine_count = len(list(genuine_folder.glob("*")))

print("Fake images:", fake_count)
print("Genuine images:", genuine_count)

if fake_count < 2 or genuine_count < 2:
    raise ValueError(
        "Add at least 2 images in each class to begin training."
    )

# --------------------------------
# LOAD DATASET
# --------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fake", "genuine"]
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["fake", "genuine"]
)

print("Class Names:", train_dataset.class_names)
print("0 = Fake")
print("1 = Genuine")

# --------------------------------
# IMAGE NORMALIZATION
# --------------------------------

normalization_layer = layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# --------------------------------
# CNN MODEL ARCHITECTURE
# --------------------------------

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    # Data Augmentation
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),

    # First Convolution Block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Second Convolution Block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Third Convolution Block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Fourth Convolution Block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D((2, 2)),

    # Flatten Layer
    layers.Flatten(),

    # Fully Connected Layers
    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    # Output Layer
    layers.Dense(1, activation="sigmoid")
])

# --------------------------------
# COMPILE MODEL
# --------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")

model.summary()

# --------------------------------
# TRAINING CALLBACKS
# --------------------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]

# --------------------------------
# START MODEL TRAINING
# --------------------------------

print("\nTraining Started...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --------------------------------
# SAVE TRAINED MODEL
# --------------------------------

model.save(MODEL_PATH)

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("Model Saved:", MODEL_PATH)

print(
    "Final Training Accuracy:",
    round(history.history["accuracy"][-1] * 100, 2),
    "%"
)

print(
    "Final Validation Accuracy:",
    round(history.history["val_accuracy"][-1] * 100, 2),
    "%"
)

print("\nYour CNN Model is Ready!")
