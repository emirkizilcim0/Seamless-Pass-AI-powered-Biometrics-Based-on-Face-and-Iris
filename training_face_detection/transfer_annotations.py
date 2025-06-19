import os

# Input annotation file
annotation_file = r"C:\Users\Emir\Desktop\training model\WIDERFace\wider_face_split\wider_face_val_bbx_gt.txt"

# Output directory for YOLO label files
output_dir = r"C:\Users\Emir\Desktop\training model\datasets\labels\val"
os.makedirs(output_dir, exist_ok=True)

# Check that the annotation file exists
assert os.path.exists(annotation_file), "Annotation file not found!"

with open(annotation_file, "r") as file:
    lines = file.readlines()

i = 0
img_width = 1024  # placeholder (you can change this if using real image dimensions)
img_height = 768
image_count = 0

while i < len(lines):
    line = lines[i].strip()
    if ".jpg" in line.lower():
        image_path = line
        image_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(output_dir, image_name)

        i += 1
        if i >= len(lines):
            break

        try:
            num_faces = int(lines[i].strip())
        except ValueError:
            print(f"[WARNING] Skipping invalid face count at line {i}: {lines[i]}")
            continue

        with open(label_path, "w") as out:
            for _ in range(num_faces):
                i += 1
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) < 4:
                    continue
                x, y, w, h = map(float, parts[:4])
                cx = (x + w / 2) / img_width
                cy = (y + h / 2) / img_height
                bw = w / img_width
                bh = h / img_height
                out.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"[INFO] Written {num_faces} faces to {label_path}")
        image_count += 1
        i += 1
    else:
        i += 1

print(f"[DONE] Total processed images: {image_count}")
