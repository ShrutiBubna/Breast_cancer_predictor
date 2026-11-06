
from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

# Dataset path
DATASET_PATH = Path(
    "dataset_cancer_v1/classificacao_binaria/40X"
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Collect image information
records = []

for class_name, label in [("benign", 0), ("malignant", 1)]:

    class_path = DATASET_PATH / class_name

    for image_path in class_path.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parts = image_path.stem.split("-")
        group_id = "-".join(parts[:-2])

        records.append({
            "image_path": str(image_path),
            "label": label,
            "class_name": class_name,
            "group_id": group_id
        })

# Create DataFrame
df = pd.DataFrame(records)
# Inspect image filenames and extracted group IDs
print("\nSAMPLE GROUP IDENTIFIERS")
print(df[["image_path", "label", "group_id"]].head(15))

print("\nUNIQUE GROUPS PER CLASS")
print(
    df.groupby("label")["group_id"].nunique()
)

print("\nIMAGES PER GROUP")
print(
    df.groupby("group_id").size().describe()
)

print("Total images:", len(df))
print("Total groups:", df["group_id"].nunique())

# First split: 70% training, 30% remaining
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_indices, remaining_indices = next(
    split1.split(
        df,
        df["label"],
        groups=df["group_id"]
    )
)

train_df = df.iloc[train_indices].copy()
remaining_df = df.iloc[remaining_indices].copy()

# Second split: divide remaining data into
# 15% validation and 15% testing
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_indices, test_indices = next(
    split2.split(
        remaining_df,
        remaining_df["label"],
        groups=remaining_df["group_id"]
    )
)

val_df = remaining_df.iloc[val_indices].copy()
test_df = remaining_df.iloc[test_indices].copy()

# Verify group separation
train_groups = set(train_df["group_id"])
val_groups = set(val_df["group_id"])
test_groups = set(test_df["group_id"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

# Display results
print("\nDATASET SPLIT RESULTS")
print("=" * 45)

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:

    print(f"\n{name} set:")
    print("Images:", len(data))
    print("Groups:", data["group_id"].nunique())
    print("Benign:", (data["label"] == 0).sum())
    print("Malignant:", (data["label"] == 1).sum())

# Save split information
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("validation_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("\nGroup overlap verification: PASSED")
print("Split CSV files saved successfully!")
