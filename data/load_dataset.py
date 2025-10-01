import os
import csv

# Path to the main dataset folder
dataset_dir = r"D:\Waste Classification Dataset\Waste Classification Dataset\waste_dataset"  # <-- replace with your dataset folder path

# Folders inside dataset_dir: 'recyclable' and 'organic'
classes = ["recyclable", "organic"]

# Output CSV file
csv_file = os.path.join(dataset_dir, "waste_labels.csv")

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])  # header

    for label in classes:
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.exists(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                # Write relative path (or full path if you prefer)
                writer.writerow([os.path.join(label, img_file), label])

print(f"CSV file created at: {csv_file}")
