import tensorflow as tf
import pandas as pd

# Image configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

# Dataset CSV files
TRAIN_CSV = "train_data.csv"
VAL_CSV = "validation_data.csv"
TEST_CSV = "test_data.csv"


# Function to load image and label paths
def load_dataset(csv_file):

    df = pd.read_csv(csv_file)

    image_paths = df["image_path"].values
    labels = df["label"].values.astype("float32")

    return image_paths, labels


# Function to preprocess an individual image
def preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH]
    )

    # Normalize pixel values
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# Function to create TensorFlow dataset
def create_dataset(csv_file, shuffle=False):

    image_paths, labels = load_dataset(csv_file)

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=42
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Create datasets
print("Preparing CNN datasets...")

train_dataset = create_dataset(
    TRAIN_CSV,
    shuffle=True
)

val_dataset = create_dataset(
    VAL_CSV
)

test_dataset = create_dataset(
    TEST_CSV
)

print("\nCNN datasets created successfully!")

# Inspect one batch
for images, labels in train_dataset.take(1):

    print("\nBatch image shape:", images.shape)
    print("Batch label shape:", labels.shape)

    print(
        "Pixel value range:",
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )

    print(
        "Sample labels:",
        labels[:10].numpy()
    )


# ==========================================
# CUSTOM CNN ARCHITECTURE
# ==========================================

from tensorflow.keras import layers, models

def build_cnn_model():

    model = models.Sequential([

        # Input layer
        layers.Input(shape=(224, 224, 3)),

        # First convolutional block
        layers.Conv2D(
            32, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third convolutional block
        layers.Conv2D(
            128, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classification layers
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        # Output: malignant-class probability
        layers.Dense(1, activation="sigmoid")
    ])

    return model


# Build CNN model
cnn_model = build_cnn_model()

# Display architecture
cnn_model.summary()


# ==========================================
# CNN MODEL COMPILATION
# ==========================================

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=np.concatenate([
        pd.read_csv(TRAIN_CSV)["label"].values
    ])
)

class_weights = {
    0: class_weights_array[0],
    1: class_weights_array[1]
}

print("\nCLASS WEIGHTS:")
print(class_weights)

# Compile CNN model
cnn_model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("\nCNN model compiled successfully!")


# ==========================================
# TRAINING CALLBACKS
# ==========================================

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# Create folder for saved models
import os

os.makedirs("saved_models", exist_ok=True)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Save the best model
model_checkpoint = ModelCheckpoint(
    filepath="saved_models/best_cnn_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

print("\nTraining callbacks configured successfully!")


# ==========================================
# CNN MODEL TRAINING
# ==========================================

print("\nStarting CNN training...")

history = cnn_model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=20,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]

)

print("\nCNN training completed!")

# Save final training history
import pandas as pd

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "cnn_training_history.csv",
    index=False
)

print("Training history saved successfully!")
