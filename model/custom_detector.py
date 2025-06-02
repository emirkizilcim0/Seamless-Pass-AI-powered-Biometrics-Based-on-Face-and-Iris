import torch
import cv2
import numpy as np
from models.face_detector_model import MyFaceDetectorModel  # Your own model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
face_detector_model = MyFaceDetectorModel()
face_detector_model.load_state_dict(torch.load("models/face_detector.pth", map_location=device))
face_detector_model.to(device)
face_detector_model.eval()

def detect_faces(image):
    """Run your custom face detector on an image and return face bounding boxes."""
    resized = cv2.resize(image, (224, 224))
    tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = face_detector_model(tensor)  # You define what this returns

    # Assume it returns list of [x1, y1, x2, y2] scaled to original image size
    return outputs  # List of bounding boxes
