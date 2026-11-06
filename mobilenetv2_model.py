
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# CONFIGURATION
# ==========================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_CSV = "train_data.csv"
VAL_CSV = "validation_data.csv"

MODEL_PATH = "saved_models/best_mobilenetv2.keras"

print("MobileNetV2 configuration loaded successfully!")


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def load_dataset(csv_file):

    df = pd.read_csv(csv_file)

    image_paths = df["image_path"].values
    labels = df["label"].values.astype("float32")

    return image_paths, labels


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

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


# ==========================================
# CREATE TENSORFLOW DATASETS
# ==========================================

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

    return dataset, labels


print("\nPreparing MobileNetV2 datasets...")

train_dataset, y_train = create_dataset(
    TRAIN_CSV,
    shuffle=True
)

val_dataset, y_val = create_dataset(
    VAL_CSV
)

print("Training images:", len(y_train))
print("Validation images:", len(y_val))

print("\nMobileNetV2 datasets prepared successfully!")



# ==========================================
# BUILD MOBILENETV2 MODEL
# ==========================================

print("\nBuilding MobileNetV2 model...")

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Add custom classification layers
inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

mobilenet_model = models.Model(
    inputs,
    outputs
)

print("\nMobileNetV2 model created successfully!")

mobilenet_model.summary()


# ==========================================
# COMPILE MOBILENETV2 MODEL
# ==========================================

mobilenet_model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("\nMobileNetV2 compiled successfully!")



# ==========================================
# TRAINING CALLBACKS
# ==========================================

os.makedirs("saved_models", exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

print("\nMobileNetV2 callbacks configured successfully!")


# ==========================================
# MOBILENETV2 MODEL TRAINING
# ==========================================

print("\nStarting MobileNetV2 training...")

# Calculate class weights
classes = np.unique(y_train.astype(int))

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.astype(int)
)

class_weights = dict(
    zip(classes.tolist(), weights.tolist())
)

print("\nClass weights:", class_weights)

# Train model
history = mobilenet_model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

# Save training history
history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "mobilenetv2_training_history.csv",
    index=False
)

print("\nMobileNetV2 training completed!")
print("Training history saved successfully!")
