
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Configuration
MODEL_PATH = "saved_models/best_mobilenetv2.keras"
TEST_CSV = "test_data.csv"
IMG_SIZE = 224
BATCH_SIZE = 16

# Load test dataset
df = pd.read_csv(TEST_CSV)

paths = df["image_path"].values
labels = df["label"].values.astype(int)

# Image preprocessing
def preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label

dataset = tf.data.Dataset.from_tensor_slices(
    (paths, labels)
)

dataset = dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

# Load existing model
print("Loading MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Generate probabilities
print("Predicting test images...")

probabilities = model.predict(
    dataset.map(lambda x, y: x),
    verbose=1
).flatten()

# Compare thresholds
print("\nTEST THRESHOLD COMPARISON")
print("=========================")

for threshold in [0.20, 0.30, 0.50]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    print(f"\nThreshold: {threshold}")

    print(
        "Accuracy:",
        round(accuracy_score(labels, predictions), 4)
    )

    print(
        "Malignant Precision:",
        round(precision_score(labels, predictions, zero_division=0), 4)
    )

    print(
        "Malignant Recall:",
        round(recall_score(labels, predictions, zero_division=0), 4)
    )

    print(
        "F1-score:",
        round(f1_score(labels, predictions, zero_division=0), 4)
    )

    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions))

print(
    "\nTest ROC-AUC:",
    round(roc_auc_score(labels, probabilities), 4)
)

# Save test probabilities
results = pd.DataFrame({
    "actual_label": labels,
    "malignant_probability": probabilities
})

results.to_csv(
    "mobilenetv2_test_probabilities.csv",
    index=False
)

print("\nTest probability results saved successfully.")
