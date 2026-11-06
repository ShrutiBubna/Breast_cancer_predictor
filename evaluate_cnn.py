
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

# ==========================================
# CONFIGURATION
# ==========================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

TEST_CSV = "test_data.csv"
MODEL_PATH = "saved_models/best_cnn_model.keras"


# ==========================================
# LOAD TEST DATA
# ==========================================

print("Loading independent test dataset...")

df = pd.read_csv(TEST_CSV)

image_paths = df["image_path"].values
y_true = df["label"].values.astype("int32")

print("Total test images:", len(image_paths))
print("Test label distribution:")
print(pd.Series(y_true).value_counts().sort_index())


# ==========================================
# IMAGE PREPROCESSING
# Same as cnn_model.py
# ==========================================

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

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ==========================================
# CREATE TEST DATASET
# ==========================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, y_true)
)

test_dataset = test_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==========================================
# LOAD TRAINED CNN MODEL
# ==========================================

print("\nLoading best saved CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# GENERATE PREDICTIONS
# ==========================================

print("\nEvaluating independent test data...")

y_prob = model.predict(test_dataset).ravel()
# ==========================================
# DIAGNOSTIC CHECK
# ==========================================

print("\nTEST DATA LABEL CHECK")

print("\nSample image paths and labels:")
print(df[["image_path", "label"]].head(10).to_string(index=False))

print("\nImage folder names against labels:")
print(
    df.assign(
        folder=df["image_path"].str.replace("\\", "/", regex=False)
        .str.lower()
        .str.extract(r"/(benign|malignant)/")[0]
    ).groupby(["folder", "label"]).size()
)

print("\nAverage predicted probability by actual class:")

for label in [0, 1]:
    print(
        f"Class {label}:",
        round(float(y_prob[y_true == label].mean()), 4)
    )

print("\nPredicted probability range:")
print("Minimum:", float(y_prob.min()))
print("Maximum:", float(y_prob.max()))

y_pred = (y_prob >= 0.5).astype(int)


# ==========================================
# EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 50)
print("INDEPENDENT TEST SET EVALUATION")
print("=" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign (0)", "Malignant (1)"],
        zero_division=0
    )
)

print("\nTest evaluation completed successfully!")
