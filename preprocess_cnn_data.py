
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# Image dimensions
IMG_SIZE = (224, 224)

# Function to load and preprocess images
def load_images(csv_file):

    df = pd.read_csv(csv_file)

    images = []
    labels = []

    for _, row in df.iterrows():

        image_path = Path(row["image_path"])

        # Open image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Resize image
        image = image.resize(IMG_SIZE)

        # Convert image to NumPy array
        image_array = np.array(image, dtype=np.float32)

        # Normalize pixel values
        image_array = image_array / 255.0

        images.append(image_array)
        labels.append(row["label"])

    return np.array(images), np.array(labels)


# Load training data
print("Loading training images...")
X_train, y_train = load_images("train_data.csv")

# Load validation data
print("Loading validation images...")
X_val, y_val = load_images("validation_data.csv")

# Load testing data
print("Loading testing images...")
X_test, y_test = load_images("test_data.csv")


# Display dataset information
print("\nPREPROCESSING RESULTS")
print("=" * 40)

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Validation images:", X_val.shape)
print("Validation labels:", y_val.shape)

print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)

print("\nMinimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())

print("\nImage preprocessing completed successfully!")
