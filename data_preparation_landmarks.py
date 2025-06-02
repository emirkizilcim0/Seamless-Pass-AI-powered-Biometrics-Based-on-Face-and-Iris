import os
import glob
from PIL import Image
import xml.etree.ElementTree as ET

# Base directory
BASE_DIR = "dataset-landmarks"
OUTPUT_XML = "training.xml"
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


# Define train and test folders separately
train_dirs = [
    os.path.join(BASE_DIR, "afw"),
    os.path.join(BASE_DIR, "ibug"),
    os.path.join(BASE_DIR, "helen", "trainset"),
    os.path.join(BASE_DIR, "lfpw", "trainset"),
]

test_dirs = [
    os.path.join(BASE_DIR, "helen", "testset"),
    os.path.join(BASE_DIR, "lfpw", "testset"),
]

def read_pts(pts_path):
    with open(pts_path, "r") as f:
        lines = f.read().strip().splitlines()
        lines = lines[3:-1]  # Skip header/footer
        return [tuple(map(float, line.strip().split())) for line in lines]

def find_image_file(pts_path):
    base = os.path.splitext(pts_path)[0]
    for ext in IMAGE_EXTS:
        img_path = base + ext
        if os.path.isfile(img_path):
            return img_path
    return None

def create_dlib_xml(dataset_root, output_file, data_dirs):
    dataset = ET.Element("dataset")
    images = ET.SubElement(dataset, "images")
    total = 0

    for data_dir in data_dirs:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".pts"):
                    pts_path = os.path.join(root, file)
                    img_path = find_image_file(pts_path)
                    if not img_path:
                        print(f"[!] No image found for {pts_path}")
                        continue

                    landmarks = read_pts(pts_path)
                    if len(landmarks) != 68:
                        print(f"Skipping {img_path}: {len(landmarks)} landmarks (expected 68)")
                        continue

                    xs = [x for x, y in landmarks]
                    ys = [y for x, y in landmarks]
                    left, top = int(min(xs)), int(min(ys))
                    right, bottom = int(max(xs)), int(max(ys))

                    box_width = right - left
                    box_height = bottom - top

                    image_el = ET.SubElement(images, "image", file=img_path)
                    box_el = ET.SubElement(image_el, "box", top=str(top), left=str(left),
                                           width=str(box_width), height=str(box_height))

                    for i, (x, y) in enumerate(landmarks):
                        ET.SubElement(box_el, "part", name=str(i), x=str(int(x)), y=str(int(y)))
                    total += 1

    tree = ET.ElementTree(dataset)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"{total} samples written to {output_file}")

# Run
# Create separate XML files
create_dlib_xml(BASE_DIR, "training.xml", train_dirs)
create_dlib_xml(BASE_DIR, "testing.xml", test_dirs)
