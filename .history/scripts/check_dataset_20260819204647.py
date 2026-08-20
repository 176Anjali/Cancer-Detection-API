import os
from PIL import Image

# ----------------------------
# Change this path if necessary
# ----------------------------
DATASET_PATH = "../dataset/ALL"

# Supported image formats
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

total_images = 0
corrupted_images = []

print("=" * 60)
print("CHECKING DATASET")
print("=" * 60)

for class_name in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"\nClass: {class_name}")

    image_count = 0

    for filename in os.listdir(class_path):

        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        image_path = os.path.join(class_path, filename)

        try:
            img = Image.open(image_path)
            img.verify()  # Verify image integrity
            image_count += 1

        except Exception:
            corrupted_images.append(image_path)

    total_images += image_count

    print(f"Images : {image_count}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Images : {total_images}")
print(f"Corrupted Images : {len(corrupted_images)}")

if corrupted_images:
    print("\nCorrupted Files:")
    for img in corrupted_images:
        print(img)
else:
    print("\nNo corrupted images found.")