import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

# 1. Find the project directory
project_dir = Path(__file__).resolve().parent
 
# 2. Locate the annotation file
json_path = (
    project_dir / "data" / "idd" /
    "annotations" / "instances.json"
)
 
# 3. Load the JSON file
with open(json_path, "r", encoding="utf-8") as file:
    coco = json.load(file)
 
print("COCO JSON loaded successfully!")
 
# 4. Read the three main sections
images = coco["images"]
categories = coco["categories"]
annotations = coco["annotations"]
 
# 5. Print dataset summary
print("\nDATASET SUMMARY")
print("Images:", len(images))
print("Categories:", len(categories))
print("Annotations:", len(annotations))
 
# 6. Print category information
print("\nAVAILABLE CATEGORIES")
for category in categories:
    print(category["id"], ":", category["name"])
 
# 7. Print image information
print("\nIMAGE INFORMATION")
for image in images:
    print(
        "Image ID:", image["id"],
        "| File:", image["file_name"],
        "| Width:", image["width"],
        "| Height:", image["height"]
    )
 
# 8. Print object annotations
print("\nOBJECT ANNOTATIONS")
for annotation in annotations:
    print(
        "Image ID:", annotation["image_id"],
        "| Category ID:", annotation["category_id"],
        "| Bounding Box:", annotation["bbox"]
    )


image_dir = project_dir / "data" / "idd" / "images"
output_dir = project_dir / "docs"
output_dir.mkdir(parents=True, exist_ok=True)
 
# Which image to draw: None = the first image that exists and has annotations.
# Set it to an image id (for example 2) to draw that image instead.
TARGET_IMAGE_ID = None
 
# Convert category IDs into category names
category_names = {
    category["id"]: category["name"]
    for category in categories
}
 
# Group annotations by image ID
annotations_by_image = defaultdict(list)
for annotation in annotations:
    annotations_by_image[annotation["image_id"]].append(annotation)
 
# Find an image that exists and has annotations
selected_image = None
for image_info in images:
    if TARGET_IMAGE_ID is not None and image_info["id"] != TARGET_IMAGE_ID:
        continue
    candidate = image_dir / image_info["file_name"]
    if candidate.exists() and annotations_by_image[image_info["id"]]:
        selected_image = image_info
        break
 
print("\nDRAWING")
if selected_image is None:
    print("No matching annotated image found. Check image names and paths.")
else:
    image_path = image_dir / selected_image["file_name"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
 
    for annotation in annotations_by_image[selected_image["id"]]:
        x, y, width, height = annotation["bbox"]
        x1, y1 = x, y
        x2, y2 = x + width, y + height
 
        label = category_names.get(
            annotation["category_id"], "Unknown"
        )
 
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )
        label_position = (x1, max(0, y1 - 15))
        draw.rectangle(draw.textbbox(label_position, label), fill="red")
        draw.text(label_position, label, fill="white")
 
    output_path = output_dir / f"coco_preview_{selected_image['id']}.jpg"
    image.save(output_path)
    print("Preview saved to:", output_path)
