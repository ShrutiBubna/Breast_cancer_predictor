
import pandas as pd
import numpy as np
from PIL import Image

datasets = {
    "Training": "train_data.csv",
    "Testing": "test_data.csv"
}

for dataset_name, csv_file in datasets.items():

    df = pd.read_csv(csv_file)

    print("\n", "=" * 45)
    print(dataset_name.upper())
    print("=" * 45)

    for label, class_name in [
        (0, "Benign"),
        (1, "Malignant")
    ]:

        class_df = df[df["label"] == label]

        rgb_values = []

        for path in class_df["image_path"]:

            image = Image.open(path).convert("RGB")
            image = image.resize((224, 224))

            array = np.asarray(image, dtype=np.float32)

            rgb_values.append(array.mean(axis=(0, 1)))

        rgb_values = np.array(rgb_values)

        print(f"\n{class_name} ({len(rgb_values)} images)")
        print("Mean Red:", round(rgb_values[:, 0].mean(), 2))
        print("Mean Green:", round(rgb_values[:, 1].mean(), 2))
        print("Mean Blue:", round(rgb_values[:, 2].mean(), 2))
