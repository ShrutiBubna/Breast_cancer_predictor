
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
VAL_CSV = "validation_data.csv"
IMG_SIZE = 224
BATCH_SIZE = 16

# Load validation dataset
df = pd.read_csv(VAL_CSV)

paths = df["image_path"].values
labels = df["label"].values.astype(int)

# Preprocess images
def preprocess_image(path, label):

    image = tf.io.read_file(path)

    image = tf.io.decode_png(
        image, channels=3
    )

    image = tf.image.resize(
        image, (IMG_SIZE, IMG_SIZE)
    )

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

# Predict probabilities
print("Predicting validation images...")

probabilities = model.predict(
    dataset.map(lambda x, y: x),
    verbose=1
).flatten()

# Analyze probability distribution
results = pd.DataFrame({
    "actual_label": labels,
    "malignant_probability": probabilities
})

print("\nPROBABILITY DISTRIBUTION")
print("========================")

for label, name in [(0, "Benign"), (1, "Malignant")]:

    values = results.loc[
        results["actual_label"] == label,
        "malignant_probability"
    ]

    print(f"\n{name} images:")
    print("Count:", len(values))
    print("Mean:", round(values.mean(), 4))
    print("Minimum:", round(values.min(), 4))
    print("Maximum:", round(values.max(), 4))
    print("Median:", round(values.median(), 4))

# Compare different thresholds on validation data
print("\nTHRESHOLD COMPARISON")
print("====================")

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels, predictions
    )

    precision = precision_score(
        labels, predictions, zero_division=0
    )

    recall = recall_score(
        labels, predictions, zero_division=0
    )

    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    cm = confusion_matrix(
        labels, predictions
    )

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Malignant Precision: {precision:.4f}")
    print(f"Malignant Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# ROC-AUC is independent of the classification threshold
print(
    "\nValidation ROC-AUC:",
    round(roc_auc_score(labels, probabilities), 4)
)

# Save probability results
results.to_csv(
    "mobilenetv2_validation_probabilities.csv",
    index=False
)

print("\nProbability analysis saved successfully.")
