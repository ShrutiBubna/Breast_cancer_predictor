
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

IMG_SIZE = 224
BATCH_SIZE = 16

MODEL_PATH = "saved_models/best_cnn_model_v2.keras"

# Load model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


def load_dataset(csv_file):

    df = pd.read_csv(csv_file)

    paths = df["image_path"].values
    labels = df["label"].values.astype("int32")

    def preprocess(path, label):

        image = tf.io.read_file(path)

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            [IMG_SIZE, IMG_SIZE]
        )

        image = tf.cast(image, tf.float32) / 255.0

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels


for name, csv_file in [
    ("TRAINING", "train_data.csv"),
    ("VALIDATION", "validation_data.csv"),
    ("TESTING", "test_data.csv")
]:

    dataset, y_true = load_dataset(csv_file)

    probabilities = model.predict(
        dataset,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\n" + "=" * 45)
    print(name, "RESULTS")
    print("=" * 45)

    print(
        "Accuracy:",
        round(accuracy_score(y_true, predictions), 4)
    )

    print(
        "Benign recall:",
        round(
            recall_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Malignant recall:",
        round(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            ),
            4
        )
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(y_true, probabilities),
            4
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))
