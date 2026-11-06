
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
EPOCHS = 20

TRAIN_CSV = "train_data.csv"
VAL_CSV = "validation_data.csv"


# ==========================================
# DATA LOADING
# ==========================================

def load_dataset(csv_file):

    df = pd.read_csv(csv_file)

    image_paths = df["image_path"].values
    labels = df["label"].values.astype("float32")

    return image_paths, labels


def preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH]
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


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


print("Preparing datasets...")

train_dataset, y_train = create_dataset(
    TRAIN_CSV,
    shuffle=True
)

val_dataset, y_val = create_dataset(
    VAL_CSV
)

print("Training and validation datasets ready!")


# ==========================================
# CLASS WEIGHTS
# ==========================================

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


# ==========================================
# CNN ARCHITECTURE
# ==========================================

def build_cnn_model():

    model = models.Sequential([

        layers.Input(shape=(224, 224, 3)),

        layers.Conv2D(
            32, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    return model


cnn_model = build_cnn_model()

cnn_model.summary()


# ==========================================
# MODEL COMPILATION
# ==========================================

cnn_model.compile(

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

print("\nCNN v3 compiled successfully!")


# ==========================================
# CALLBACKS
# ==========================================

os.makedirs("saved_models", exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath="saved_models/best_cnn_model_v3.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ==========================================
# TRAINING
# ==========================================

print("\nStarting CNN v3 training...")

history = cnn_model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

print("\nCNN v3 training completed!")


# ==========================================
# SAVE TRAINING HISTORY
# ==========================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "cnn_training_history_v3.csv",
    index=False
)

print("CNN v3 training history saved successfully!")
