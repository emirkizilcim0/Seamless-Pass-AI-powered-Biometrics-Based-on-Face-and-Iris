import os

def normalize_label(label_file, img_width, img_height):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    with open(label_file, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            cls_id, x_center, y_center, width, height = map(float, parts)
            # Normalize bounding box values
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)
            # Write back the normalized label
            f.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')

# Directory paths
labels_dir = 'C:/Users/Emir\Desktop/training model/datasets/labels/val'   # /val , /train
image_dir = 'C:/Users/Emir\Desktop/training model/datasets/images/val'    # /val , /train

# Iterate over the labels and normalize
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        img_file = label_file.replace('.txt', '.jpg')  # Assuming images are jpg
        img_path = os.path.join(image_dir, img_file)
        if os.path.exists(img_path):
            # Get image dimensions
            from PIL import Image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            normalize_label(os.path.join(labels_dir, label_file), img_width, img_height)
