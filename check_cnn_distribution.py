
import pandas as pd

files = {
    "Training": "train_data.csv",
    "Validation": "validation_data.csv",
    "Testing": "test_data.csv"
}

for name, file in files.items():

    df = pd.read_csv(file)

    print("\n" + "=" * 40)
    print(name.upper(), "DATASET")
    print("=" * 40)

    counts = df["label"].value_counts().sort_index()

    print("Total images:", len(df))
    print("Benign images:", counts.get(0, 0))
    print("Malignant images:", counts.get(1, 0))

    print("\nClass percentages:")
    print(
        (df["label"].value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )
