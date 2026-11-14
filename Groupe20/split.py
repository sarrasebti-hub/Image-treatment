import json
import random
from collections import defaultdict
from pathlib import Path

# 1. SET UP FOLDERS
# Define where the original COCO text captions are stored
ANN_DIR = Path(r"C:\Users\DELL\Desktop\annotations_trainval2017\annotations")
# Define where we want to save our new list of IDs
OUT_DIR = Path("splits")
# Create the "splits" folder if it doesn't already exist on your computer
OUT_DIR.mkdir(exist_ok=True)

# 2. SET CONFIGURATION VARIABLES
# A fixed seed guarantees that every time you run this code, 
SEED = 42
N_KB = 10_000

# 3. LOAD THE TEXT DATA
# Open the giant text file containing all the COCO captions and load it into memory
with open(ANN_DIR / "captions_train2017.json", "r") as f:
    data = json.load(f)

# Create an empty dictionary where the keys will be Image IDs, 
# and the values will be a list of all captions for that ID
caps = defaultdict(list)

# Loop through every single annotation in the text file
for a in data["annotations"]:
    # Add the current caption to the list for this specific image_id
    caps[a["image_id"]].append(a["caption"])

# 5. FILTER OUT IMAGES WITH TOO FEW CAPTIONS
# Loop through our grouped data. Keep the image ID ('i') ONLY if the 
# number of captions ('len(c)') is 5 or more. 
# Images with 1, 2, 3, or 4 captions are completely ignored here!
valid_ids = sorted(i for i, c in caps.items() if len(c) >= 5)
print("Images with >=5 captions:", len(valid_ids))

# 6. RANDOMLY SELECT 10,000 IDs
# Lock in our random seed so the "random" choice is repeatable
random.seed(SEED)
# From our pool of 'valid_ids', randomly pick exactly 10,000 IDs and sort them
kb_ids = sorted(random.sample(valid_ids, N_KB))

# 7. SAVE THE FINAL LIST
# Create a new JSON file inside the "splits" folder
with open(OUT_DIR / "kb_train_ids.json", "w") as f:
    # Write just the 10,000 ID numbers into this file (no actual images are saved)
    json.dump(kb_ids, f)

# Print a success message so you know it finished
print("Saved", len(kb_ids), "knowledge-base IDs to splits/kb_train_ids.json")