import os
import shutil

def copy_images(source_folder, destination_folder):
    """Copies all .jpg images from source to destination folder."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(".jpg"):
                # Construct full file path
                source_path = os.path.join(subdir, file)
                destination_path = os.path.join(destination_folder, file)
                
                # Copy the file to the destination
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")

# Paths to source and destination
source_train = "WIDERFace/WIDER_train/images"
destination_train = "datasets/images/train"

source_val = "WIDERFace/WIDER_val/images"
destination_val = "datasets/images/val"

# Copy the images
copy_images(source_train, destination_train)
copy_images(source_val, destination_val)

print("Image copying complete.")
