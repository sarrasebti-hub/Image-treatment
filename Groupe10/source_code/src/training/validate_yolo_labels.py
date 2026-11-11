from pathlib import Path

LABELS_DIR = Path("data/processed/yolo/labels/train")

files = list(LABELS_DIR.glob("*.txt"))

invalid_files = []
total_boxes = 0

for file in files:
    content = file.read_text().strip()

    if not content:
        continue

    for line_number, line in enumerate(content.splitlines(), start=1):
        values = line.split()

        if len(values) != 5:
            invalid_files.append(f"{file.name}: line {line_number}")
            continue

        class_id = int(values[0])
        coordinates = [float(x) for x in values[1:]]

        if not 0 <= class_id < 80:
            invalid_files.append(f"{file.name}: invalid class ID")

        if not all(0 <= x <= 1 for x in coordinates):
            invalid_files.append(f"{file.name}: invalid coordinates")

        total_boxes += 1

print("===== LABEL VALIDATION =====")
print(f"Label files: {len(files)}")
print(f"Bounding boxes checked: {total_boxes}")
print(f"Invalid entries: {len(invalid_files)}")

for error in invalid_files[:10]:
    print(error)

print("Validation:", "PASSED" if not invalid_files else "FAILED")