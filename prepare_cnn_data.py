
from pathlib import Path
from collections import Counter

# Dataset location
DATASET_PATH = Path(
    "dataset_cancer_v1/classificacao_binaria/40X"
)

# Supported image formats
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Classes
classes = ["benign", "malignant"]

print("BREAKHIS DATASET INSPECTION")
print("=" * 40)

total_images = 0

for class_name in classes:

    class_path = DATASET_PATH / class_name

    if not class_path.exists():
        print(f"Folder not found: {class_path}")
        continue

    image_files = [
        file for file in class_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_count = len(image_files)
    total_images += image_count

    print(f"\nClass: {class_name}")
    print(f"Number of images: {image_count}")

    # Inspect a few filenames
    print("Sample filenames:")

    for image in image_files[:5]:
        print(" ", image.name)

print("\n" + "=" * 40)
print("TOTAL IMAGES:", total_images)
print("DATASET INSPECTION COMPLETED")


print("\nPATIENT GROUP INSPECTION")
print("=" * 40)

for class_name in classes:

    class_path = DATASET_PATH / class_name

    image_files = [
        file for file in class_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    patient_ids = set()

    for image in image_files:
        parts = image.stem.split("-")

        # Remove the final image number and magnification
        patient_id = "-".join(parts[:-2])

        patient_ids.add(patient_id)

    print(f"\nClass: {class_name}")
    print("Unique patient/specimen groups:", len(patient_ids))

    print("Sample group IDs:")

    for patient_id in sorted(patient_ids)[:5]:
        print(" ", patient_id)
