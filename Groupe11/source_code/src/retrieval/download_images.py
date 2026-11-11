import json
import os
import urllib.request
from tqdm import tqdm

# 1. Load the IDs from Membre 1's splits.json
with open("outputs/splits.json", "r") as f:
    splits = json.load(f)

# We only need the Knowledge Base (KB) and Validation (Val) images for now.
# The test images are secret and only used at the very end.
ids_to_download = splits["kb_ids"] + splits["val_ids"]

# 2. Create the folder to save the images
save_dir = "data/raw/images"
os.makedirs(save_dir, exist_ok=True)

print(f"Downloading {len(ids_to_download)} images from COCO...")

# 3. Download loop
for img_id in tqdm(ids_to_download):
    # COCO images are stored in folders based on their ID
    # Example: ID 12345 is in train2017/000000012345.jpg
    img_id_str = str(img_id).zfill(12)
    url = f"http://images.cocodataset.org/train2017/{img_id_str}.jpg"
    filepath = os.path.join(save_dir, f"{img_id_str}.jpg")
    
    # Skip if we already downloaded it
    if os.path.exists(filepath):
        continue
        
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        print(f"\nFailed to download {img_id}: {e}")

print("\nDownload complete!")