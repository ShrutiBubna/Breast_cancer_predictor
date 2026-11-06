
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load datasets
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Select 3 benign and 3 malignant images from each dataset
samples = []

for name, df in [
    ("Training", train_df),
    ("Testing", test_df)
]:

    for label in [0, 1]:

        selected = df[df["label"] == label].head(3)

        for _, row in selected.iterrows():

            samples.append({
                "dataset": name,
                "class": "Benign" if label == 0 else "Malignant",
                "path": row["image_path"]
            })

# Display images
fig, axes = plt.subplots(2, 6, figsize=(16, 7))

for i, sample in enumerate(samples):

    image = Image.open(sample["path"]).convert("RGB")

    row = 0 if sample["dataset"] == "Training" else 1
    col = (0 if sample["class"] == "Benign" else 3) + (
        i % 3
    )

    axes[row, col].imshow(image)
    axes[row, col].set_title(
        f'{sample["dataset"]}\n{sample["class"]}'
    )
    axes[row, col].axis("off")

plt.tight_layout()
plt.show()
