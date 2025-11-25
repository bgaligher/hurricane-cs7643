import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Define folders and valid damage classes
folders = ['train', 'tier3', 'test', 'hold']

valid_classes = [
    'no-damage',
    'minor-damage',
    'major-damage',
    'destroyed',
    'unclassified'
]

# Initialize empty dicts
disaster_image_counts = {} 
damage_class_totals = {}
disaster_damage_breakdown = {}

for cls in valid_classes:
    damage_class_totals[cls] = 0

all_image_records = []  

print("Scanning dataset...")

# Iterate through data folders
for folder in folders:
    
    img_dir = os.path.join("data", folder, "images")
    json_dir = os.path.join("data", folder, "labels")

    # Iterate through post-disaster images
    for filename in os.listdir(img_dir):
        if not filename.endswith("post_disaster.png"):
            continue
        img_path = os.path.join(img_dir, filename)
        json_filename = filename.replace(".png", ".json")
        # Get json label path
        json_path = os.path.join(json_dir, json_filename)

        # Skip rest of iteration if json path doesn't exist
        if not os.path.exists(json_path):
            continue

        # Load metadata
        with open(json_path, "r") as f:
            data = json.load(f)

        # Get disaster type
        disaster_type = data["metadata"].get("disaster_type", "unknown")

        # Increment disaster img count
        if disaster_type not in disaster_image_counts:
            disaster_image_counts[disaster_type] = 0
        disaster_image_counts[disaster_type] += 1

        # Ensure per-disaster damage dict exists
        if disaster_type not in disaster_damage_breakdown:
            disaster_damage_breakdown[disaster_type] = {cls: 0 for cls in valid_classes}

        # Count classes in this image
        polygons = data["features"]["lng_lat"]
        class_counts = {cls: 0 for cls in valid_classes}

        # Iterate through polygons
        for p in polygons:
            subtype = p["properties"].get("subtype", "unclassified")
            if subtype not in valid_classes:
                subtype = "unclassified"
            class_counts[subtype] += 1

        # Update totals
        for cls in valid_classes:
            damage_class_totals[cls] += class_counts[cls]
            disaster_damage_breakdown[disaster_type][cls] += class_counts[cls]

        # Save to list
        all_image_records.append({
            "image_name": filename,
            "folder": folder,
            "disaster_type": disaster_type,
            **class_counts
        })

print("Scanning complete!")

# Convert list to df
df = pd.DataFrame(all_image_records)

# df for number of images per disaster type
df_disasters = pd.DataFrame(
    [{"disaster_type": d, "num_images": n} for d, n in disaster_image_counts.items()]
).set_index("disaster_type")

# df for number of objects by damage level
df_damage = pd.DataFrame(
    [{"damage_level": c, "num_objects": n} for c, n in damage_class_totals.items()]
).set_index("damage_level")

# df for damage levels by disaster type
df_per_disaster_damage = pd.DataFrame.from_dict(disaster_damage_breakdown, orient="index")

# Plot 1: num image pairs per disaster type
counts = df['disaster_type'].value_counts()
plt.figure()
counts.plot(kind='bar')
plt.title("Number of post-disaster images per disaster type")
plt.ylabel("Number of images")
plt.xlabel("Disaster type")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/img_totals.png')

# Plot 2: num polygons by damage level
plt.figure()
df[valid_classes].sum().plot(kind='bar')
plt.title("Number of polygons by damage level (all images)")
plt.ylabel("Number of polygons")
plt.xlabel("Damage level")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('images/obj_totals.png')

# Plot 3: polygon damage level distributions by disaster type
plt.figure()
df.groupby('disaster_type')[valid_classes].sum().plot(kind='bar', stacked=True)
plt.title("Polygon Damage Level Distribution by Disaster Type")
plt.ylabel("Polygon Count")
plt.xlabel("Disaster Type")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/damage_cats_by_dtype.png')

# Create 'has_damage' column for Plot 4
damage_cols = ['minor-damage', 'major-damage', 'destroyed']
df["has_damage"] = df[damage_cols].sum(axis=1) > 0

# Count images with damage by disaster type
damage_summary = df.groupby(["disaster_type", "has_damage"]).size().unstack(fill_value=0)
damage_summary.columns = ["no damage", "any damage"]

# Plot 4: num images with/without damage by disaster type
plt.figure()
damage_summary.plot(kind="bar", stacked=True)
plt.title("Number of Images With & Without Damage by Disaster Type")
plt.ylabel("Number of Images")
plt.xlabel("Disaster Type")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("images/damage_vs_nodamage_by_disaster.png")

# Print results to terminal
print("\nImages per disaster type:\n", df_disasters.sort_values("num_images", ascending=False))
print("\nImages with vs without damage per disaster type:\n", damage_summary)
print("\nTotal damage level distribution across all disasters:\n", df_damage)
print("\nDamage level distribution per disaster type:\n", df_per_disaster_damage)

# Save CSVs
df.to_csv("csv_exports/xview2_postdisaster_image_stats.csv", index=False)
df_disasters.to_csv("csv_exports/xview2_disaster_counts.csv")
df_per_disaster_damage.to_csv("csv_exports/xview2_disaster_damage_breakdown.csv")


# Filter dataset for wind-only disasters
df_wind = df[df["disaster_type"] == "wind"].copy() 
# Merge tier3 and train
df_wind["folder"] = df_wind["folder"].replace({"tier3": "train"})
print("\nWind image count by split:\n")
print(df_wind["folder"].value_counts())

df_wind["total_polygons"] = df_wind[valid_classes].sum(axis=1)
polygons_per_folder = df_wind.groupby("folder")["total_polygons"].sum()
print(polygons_per_folder)