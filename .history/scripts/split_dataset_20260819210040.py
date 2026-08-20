import os
import shutil
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "dataset", "ALL")

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VALID_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_STATE = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


create_directory(TRAIN_DIR)
create_directory(VALID_DIR)
create_directory(TEST_DIR)

print("=" * 60)
print("Splitting Dataset")
print("=" * 60)

for class_name in sorted(os.listdir(SOURCE_DIR)):

    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    train_images, temp_images = train_test_split(
        images,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    valid_images, test_images = train_test_split(
        temp_images,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        create_directory(os.path.join(folder, class_name))

    for img in train_images:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    for img in valid_images:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(VALID_DIR, class_name, img)
        )

    for img in test_images:
        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(TEST_DIR, class_name, img)
        )

    print(f"\n{class_name}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(valid_images)}")
    print(f"Test       : {len(test_images)}")

print("\nDataset split completed successfully!")