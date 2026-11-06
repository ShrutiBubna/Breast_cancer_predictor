
from pathlib import Path

dataset_path = Path(
    "dataset_cancer_v1/classificacao_binaria/40X"
)

for class_name in ["benign", "malignant"]:

    folder = dataset_path / class_name

    print("\n", "=" * 55)
    print(class_name.upper(), "IMAGE FILENAMES")
    print("=" * 55)

    image_files = sorted(folder.glob("*"))

    for image_path in image_files[:10]:

        parts = image_path.stem.split("-")
        group_id = "-".join(parts[:-2])

        print("Filename :", image_path.name)
        print("Group ID :", group_id)
        print("-" * 40)

