
import os
from PIL import Image

images_folder = "val2017"

total = 0
valid = 0
corrupted = []
unreadable = []
invalid_dimensions = []

for filename in os.listdir(images_folder):

    if not filename.lower().endswith(".jpg"):
        continue

    total += 1
    image_path = os.path.join(images_folder, filename)

    try:
        with Image.open(image_path) as img:
            img.verify()

        # Reopen to check dimensions
        with Image.open(image_path) as img:
            width, height = img.size

            if width <= 0 or height <= 0:
                invalid_dimensions.append(filename)
            else:
                valid += 1

    except OSError:
        corrupted.append(filename)

    except Exception:
        unreadable.append(filename)

print("\n===== IMAGE QUALITY REPORT =====")
print("Total images:", total)
print("Valid images:", valid)
print("Corrupted images:", len(corrupted))
print("Unreadable images:", len(unreadable))
print("Invalid dimensions:", len(invalid_dimensions))
print("================================")

print("\nFirst corrupted images:", corrupted[:10])
print("First unreadable images:", unreadable[:10])
print("First invalid dimensions:", invalid_dimensions[:10])
