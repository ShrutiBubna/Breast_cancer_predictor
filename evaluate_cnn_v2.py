
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16

TEST_CSV = "test_data.csv"
MODEL_PATH = "saved_models/best_cnn_model_v2.keras"


# Load the independent test data
df = pd.read_csv(TEST_CSV)

image_paths = df["image_path"].values
y_true = df["label"].values.astype("int32")

print("Test images:", len(image_paths))
print("Benign images:", np.sum(y_true == 0))
print("Malignant images:", np.sum(y_true == 1))


# Preprocess images
def preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    # Supports PNG and JPEG images
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


# Create test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, y_true)
)

test_dataset = test_dataset.map(
    preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# Load the saved best CNN v2 model
print("\nLoading CNN v2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# Predict probabilities on the test set
print("\nEvaluating independent test data...")

y_prob = model.predict(test_dataset).ravel()

# Convert probabilities into class predictions
y_pred = (y_prob >= 0.5).astype(int)


# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)


# Display results
print("\n" + "=" * 45)
print("CNN V2 - INDEPENDENT TEST RESULTS")
print("=" * 45)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0
    )
)

print("\nTest evaluation completed successfully!")

# ==========================================
# PREDICTION PROBABILITY ANALYSIS
# ==========================================

results = pd.DataFrame({
    "Actual_Label": y_true,
    "Malignant_Probability": y_prob,
    "Predicted_Label": y_pred
})

print("\nPROBABILITY ANALYSIS")
print("=" * 45)

print("\nProbability summary by actual class:")

print(
    results.groupby("Actual_Label")[
        "Malignant_Probability"
    ].describe()
)

print("\nFirst 15 predictions:")
print(results.head(15))

# Save results for further analysis
results.to_csv(
    "cnn_v2_test_predictions.csv",
    index=False
)

print("\nPrediction analysis saved successfully!")