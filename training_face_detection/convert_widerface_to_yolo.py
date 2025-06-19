import os
import cv2

def convert_widerface_to_yolo(image_dir, annotation_dir, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all images in the image directory
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            # Get the corresponding annotation file
            annotation_file = os.path.join(annotation_dir, image_name.replace('.jpg', '.txt'))

            if not os.path.exists(annotation_file):
                continue  # Skip if annotation file doesn't exist

            # Load the image to get its dimensions
            image_path = os.path.join(image_dir, image_name)
            img = cv2.imread(image_path)
            h, w, _ = img.shape

            # Open the annotation file
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            # Open the YOLO annotation file
            yolo_annotation_file = os.path.join(output_dir, image_name.replace('.jpg', '.txt'))
            with open(yolo_annotation_file, 'w') as yolo_file:
                for line in lines:
                    # Parse the annotation: xmin, ymin, xmax, ymax
                    parts = line.strip().split(' ')
                    xmin, ymin, xmax, ymax = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

                    # Calculate YOLO format
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    width = xmax - xmin
                    height = ymax - ymin

                    # Normalize the coordinates
                    x_center /= w
                    y_center /= h
                    width /= w
                    height /= h

                    # Write the YOLO format to the file
                    yolo_file.write(f"0 {x_center} {y_center} {width} {height}\n")

if __name__ == '__main__':
    # Directories (modify these paths according to your system)
    image_dir = 'C:/Users/Emir/Desktop/training model/WIDERFace/WIDER_train/images/'
    annotation_dir = 'C:/Users/Emir/Desktop/training model/WIDERFace/WIDER_train/annotations/'
    output_dir = 'C:/Users/Emir/Desktop/training model/WIDERFacewider_face_split/'

    convert_widerface_to_yolo(image_dir, annotation_dir, output_dir)
