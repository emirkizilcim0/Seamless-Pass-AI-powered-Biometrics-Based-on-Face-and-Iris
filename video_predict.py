import cv2
import os
import numpy as np
import argparse
from face_recognition import face_encodings, load_image_file, compare_faces, face_distance
from src.FaceAntiSpoofing import AntiSpoof
import face_recognition
import torch

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)
COLOR_AUTHENTICATED = (255, 0, 255)
COLOR_UNAUTHENTICATED = (0, 165, 255)

class FaceAuthenticator:
    def __init__(self, known_faces_folder="known_faces", tolerance=0.6):
        self.known_faces_folder = known_faces_folder
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.loaded_users = set()
        self.load_known_faces_from_folder()

    def load_known_faces_from_folder(self):
        print("Loading new known faces...")

        try:
            for root, dirs, files in os.walk(self.known_faces_folder):
                person_name = os.path.basename(root)
                if person_name in self.loaded_users:
                    continue

                for file in files:
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)
                        print(f"Processing: {image_path}")
                        image = load_image_file(image_path)
                        encodings = face_encodings(image)

                        if len(encodings) > 0:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_name)
                            print(f"Added encoding for: {person_name}")
                        else:
                            print(f"No face found in: {image_path}")

                self.loaded_users.add(person_name)  # 👈 Mark as loaded

            print(f"\nTotal known people: {len(set(self.known_face_names))}")
        except Exception as e:
            print(f"Error loading known faces: {e}")

    def authenticate_face(self, face_image):
        if not self.known_face_encodings:
            print("No known face encodings loaded")
            return None, False

        try:
            # Convert to RGB and ensure proper type
            rgb_image = face_image[:, :, ::-1].astype('uint8')

            # Get face locations - returns empty list if no faces found
            face_locations = face_recognition.face_locations(rgb_image)

            if not face_locations:  # Check if empty
                print("No faces detected in image")
                return None, False

            # Get encodings for the first face found
            face_encodings_list = face_recognition.face_encodings(
                rgb_image, 
                known_face_locations=[face_locations[0]]  # Use first face found
            )

            if not face_encodings_list:
                print("Could not generate face encoding")
                return None, False

            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encodings_list[0], 
                tolerance=self.tolerance
            )
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                face_encodings_list[0]
            )

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]
                print(f"Recognized: {name} (Confidence: {confidence:.2f})")
                return name, True

            print("Face not recognized")
            return None, False

        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return None, False

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(real_w, x + int(l*bbox_inc))
    y2 = min(real_h, y + int(l*bbox_inc))
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img,
                             y1-y, int(l*bbox_inc-y2+y),
                             x1-x, int(l*bbox_inc)-x2+x,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof, authenticator=None, threshold=0.7):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detector(img_rgb)
        boxes = results.xyxy[0].cpu().numpy()
        
        if len(boxes) == 0:
            return None
        
        bbox = boxes[0][:4].astype(int)
        face_crop = increased_crop(img_rgb, bbox, bbox_inc=1.5)
        
        if face_crop.size == 0 or min(face_crop.shape[:2]) < 20:
            return None
            
        face_crop = cv2.resize(face_crop, (160, 160))
        pred = anti_spoof([face_crop])[0]
        
        auth_name, is_authenticated = None, False
        if authenticator and np.argmax(pred) == 0 and pred[0][0] > threshold:
            auth_name, is_authenticated = authenticator.authenticate_face(face_crop)
            
        return bbox, np.argmax(pred), pred[0][0], auth_name, is_authenticated
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None


def check_zero_to_one(value):
    fvalue = float(value)
    if fvalue <= 0 or fvalue >= 1:
        raise argparse.ArgumentTypeError("%s is an invalid value" % value)
    return fvalue

def parse_args():
    parser = argparse.ArgumentParser(description="Face Recognition with Anti-Spoofing")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input video path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output video path")
    parser.add_argument("--model_path", "-m", type=str, default="saved_models/best.onnx", help="Anti-spoofing model path")
    parser.add_argument("--threshold", "-t", type=check_zero_to_one, default=0.7, help="Real face confidence threshold")
    parser.add_argument("--known_faces_folder", "-k", type=str, default="known_faces", help="Folder with known faces")
    parser.add_argument("--auth_threshold", "-a", type=float, default=0.5, help="Face recognition tolerance (lower is stricter)")
    return parser.parse_args()


if __name__ == "__main__":
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue

    parser = argparse.ArgumentParser(description="Face Recognition with Anti-Spoofing")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input video path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output video path")
    parser.add_argument("--model_path", "-m", type=str,
                       default="saved_models/bestest.onnx", 
                       help="Anti-spoofing model path")
    parser.add_argument("--threshold", "-t", type=check_zero_to_one, 
                       default=0.7, help="Real face confidence threshold")
    parser.add_argument("--known_faces_folder", "-k", type=str, 
                       default="known_faces", help="Folder with known faces")
    parser.add_argument("--auth_threshold", "-a", type=float, 
                       default=0.5, help="Face recognition tolerance (lower is stricter)")
    args = parser.parse_args()

    # Initialize components
    
    face_detector = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/face-detector-from-scratch14/weights/best.pt', force_reload=False)
    anti_spoof = AntiSpoof(args.model_path)
    authenticator = FaceAuthenticator(args.known_faces_folder, args.auth_threshold)

    # Initialize video capture
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error opening video stream")
        exit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(5) or 24

    print("\n🔹 System Information:")
    print(f"- Frame size: {frame_width}x{frame_height}")
    print(f"- FPS: {fps}")
    print(f"- Real face threshold: {args.threshold}")
    print(f"- Recognition tolerance: {args.auth_threshold}")
    print(f"- Known faces loaded: {len(authenticator.known_face_names)}")

    # Initialize video writer if output specified
    if args.output:
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # Calculate display parameters based on frame size
    rec_width = max(1, int(frame_width / 240))
    txt_offset = int(frame_height / 50)
    txt_width = max(1, int(frame_width / 480))
    font_scale = (frame_width * frame_height) / (640 * 480 * 100)  # Scale based on resolution

    print("\n🎥 Starting video processing... Press 'Q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(" Stream ended or error reading frame")
            break

        # Make prediction
        pred = make_prediction(frame, face_detector, anti_spoof, authenticator)

        if pred:
            (x1, y1, x2, y2), label, score, auth_name, is_authenticated = pred

            # Determine status and color
            if label == 0:  # Real face
                if score > args.threshold:
                    if is_authenticated and auth_name:
                        status = f"REAL: {auth_name} ({score:.2f})"
                        color = COLOR_AUTHENTICATED
                    else:
                        status = f"REAL: Unknown ({score:.2f})"
                        color = COLOR_UNAUTHENTICATED
                else:
                    status = "UNKNOWN"
                    color = COLOR_UNKNOWN
            else:  # Fake face
                status = f"FAKE: {score:.2f}"
                color = COLOR_FAKE

            # Draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, rec_width)
            cv2.putText(frame, status, (x1, y1 - txt_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, txt_width)

            # Additional authentication status
            if auth_name is not None:
                auth_status = "AUTHENTICATED" if is_authenticated else "UNKNOWN PERSON"
                auth_color = COLOR_AUTHENTICATED if is_authenticated else COLOR_UNAUTHENTICATED
                cv2.putText(frame, auth_status, (x1, y1 - 2 * txt_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, auth_color, txt_width)

        # Write to output if specified
        if args.output:
            out.write(frame)

        # Display frame
        cv2.imshow("Face Recognition with Anti-Spoofing", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    if args.output:
        out.release()
    cv2.destroyAllWindows()
    print("\nSystem stopped")