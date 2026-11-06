
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

IMG_SIZE = 224
BATCH_SIZE = 16

MODEL_PATH = "saved_models/best_mobilenetv2.keras"
VAL_CSV = "test_data.csv"


# Load validation data
df = pd.read_csv(VAL_CSV)

image_paths = df["image_path"].values
y_true = df["label"].values.astype("int32")


# Preprocess images
def preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


# Create validation dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, y_true)
)

dataset = dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# Load best saved model
print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# Predict probabilities
print("\nEvaluating validation dataset...")

y_prob = model.predict(dataset).ravel()

y_pred = (y_prob >= 0.5).astype(int)


# Calculate metrics
print("\nMOBILENETV2 VALIDATION RESULTS")
print("=" * 45)

print("Accuracy:",
      round(accuracy_score(y_true, y_pred), 4))

print("Precision:",
      round(precision_score(y_true, y_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_true, y_pred, zero_division=0), 4))

print("F1-score:",
      round(f1_score(y_true, y_pred, zero_division=0), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_true, y_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0
    )
)
