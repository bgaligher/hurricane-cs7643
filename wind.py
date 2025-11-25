import os
import json

# Helper function
def copy_files(source, destination):
    with open(source, "rb") as file_source:
        with open(destination, "wb") as file_dest:
            file_dest.write(file_source.read())

# Define folders
folders = ['train', 'tier3', 'test', 'hold']

dest_folder = "data_wind"
source_folder = "data"

# Make target folders
for folder in folders:
    wind_folder = os.path.join(dest_folder, f"wind_{folder}")
    os.makedirs(os.path.join(wind_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(wind_folder, "labels"), exist_ok=True)
    if folder != "tier3":
        os.makedirs(os.path.join(wind_folder, "targets"), exist_ok=True)

print("Extracting wind data from xView2 dataset...")

# Iterate over all images
for folder in folders:
    img_dir = os.path.join(source_folder, folder, "images")
    label_dir = os.path.join(source_folder, folder, "labels")
    targets_dir = os.path.join(source_folder, folder, "targets") if folder != "tier3" else None

    wind_folder = os.path.join(dest_folder, f"wind_{folder}")

    for img_filename in os.listdir(img_dir):
        if not img_filename.endswith(".png"):
            continue
        img_path = os.path.join(img_dir, img_filename)
        img_filename = os.path.basename(img_path)
        json_filename = img_filename.replace(".png", ".json")
        json_path = os.path.join(label_dir, json_filename)
        if targets_dir is not None:
            targets_filename = img_filename.replace(".png", "_target.png")
            targets_path = os.path.join(targets_dir, targets_filename)

        if not os.path.exists(json_path):
            continue

        # Load metadata
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Skip rest of loop if disaster type is not wind
        if data["metadata"].get("disaster_type") != "wind":
            continue

        # Copy files to wind folders
        copy_files(img_path, os.path.join(wind_folder, "images", img_filename))
        copy_files(json_path, os.path.join(wind_folder, "labels", json_filename))
        # Copy targets if they exist
        if targets_dir:
            copy_files(json_path, os.path.join(wind_folder, "targets", targets_filename))

print("Wind dataset copied successfully!")