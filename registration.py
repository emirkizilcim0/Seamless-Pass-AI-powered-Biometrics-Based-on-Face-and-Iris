import cv2
import os
import numpy as np
from datetime import datetime
import dlib
from scipy.spatial import distance
import time
import re
import torch
import sys
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Add yolov5 directory to Python pathQ
yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# Now import YOLOv5 modules
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.dataloaders import letterbox
from yolov5.utils.torch_utils import select_device

# Directory where known face images will be stored
KNOWN_FACES_DIR = 'known_faces/'
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Initialize YOLOv5 model
device = select_device('0')  # Uses CUDA if available, otherwise CPU
model = attempt_load('yolov5/runs/train/face-detector-from-scratch14/weights/best.pt', device=device)
model.eval()

# Load dlib's landmark predictor
predictor = dlib.shape_predictor('my_shape_predictor_68.dat')

# Constants
EYE_AR_THRESH = 0.1  # Lower threshold for better eye closure detection
EYE_AR_CONSEC_FRAMES = 5  # More frames for more reliable detection
MOVEMENT_TIMEOUT = 5  # 5 seconds per movement
CAPTURE_INTERVAL = 0.5  # Capture image every 0.5 seconds during movement
MIN_EYE_CLOSE_PIXELS = 3  # Minimum distance between eyelids when closed

def register_user(name, face_images):
    """Save the user's face images to the database"""
    user_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    # Save all captured images with timestamps
    saved_images = []
    for idx, img in enumerate(face_images):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        img_filename = os.path.join(user_dir, f'{timestamp}_{idx}.jpg')
        cv2.imwrite(img_filename, img)
        saved_images.append(img_filename)

    print(f"User {name} registered with {len(saved_images)} images!")
    clean_user_directory(username=name)
    keep_last_per_movement(username=name)
    return True

def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio with improved reliability"""
    if len(eye_points) != 6:
        return 1.0  # Return high value (open) if we don't have enough points
        
    # Vertical distances
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    
    # Horizontal distance
    C = distance.euclidean(eye_points[0], eye_points[3])
    
    # Additional check for absolute closeness
    vertical_avg = (A + B) / 2
    if vertical_avg < MIN_EYE_CLOSE_PIXELS:
        return 0.0  # Definitely closed
    
    ear = (A + B) / (2.0 * C)
    return ear

# EAR calculation helper
def calculate_ear(eye_landmarks):
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

def get_iris_centers_mediapipe(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w = frame.shape[:2]

            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_eye = np.array([(int(face_landmarks.landmark[i].x * w), 
                                  int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices])
            right_eye = np.array([(int(face_landmarks.landmark[i].x * w), 
                                   int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices])

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            EAR_THRESHOLD = 0.2

            # Compute iris centers only for open eyes
            left_center = None
            right_center = None

            if left_ear >= EAR_THRESHOLD:
                left_iris_indices = [468, 469, 470, 471]
                left_iris_pts = [(int(face_landmarks.landmark[i].x * w),
                                  int(face_landmarks.landmark[i].y * h)) for i in left_iris_indices]
                left_center = np.mean(left_iris_pts, axis=0).astype(int)
                left_center = tuple(left_center)

            if right_ear >= EAR_THRESHOLD:
                right_iris_indices = [473, 474, 475, 476]
                right_iris_pts = [(int(face_landmarks.landmark[i].x * w),
                                   int(face_landmarks.landmark[i].y * h)) for i in right_iris_indices]
                right_center = np.mean(right_iris_pts, axis=0).astype(int)
                right_center = tuple(right_center)

            return left_center, right_center

    return None, None

def detect_eye_closure(landmarks, frame):
    """Robust eye closure detection using multiple methods"""
    if not landmarks or len(landmarks) < 48:
        return False, False
    
    # Get eye landmarks
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    # Calculate eye aspect ratios
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    
    # Method 1: EAR threshold - more sensitive threshold
    ear_closed_left = left_ear < EYE_AR_THRESH
    ear_closed_right = right_ear < EYE_AR_THRESH
    
    # Method 2: Check if upper and lower eyelids are very close
    def is_eyelids_touching(eye_points):
        if len(eye_points) < 6:
            return False
        # Distance between upper and lower eyelid points
        vertical_dist1 = distance.euclidean(eye_points[1], eye_points[5])
        vertical_dist2 = distance.euclidean(eye_points[2], eye_points[4])
        return min(vertical_dist1, vertical_dist2) < MIN_EYE_CLOSE_PIXELS
    
    touch_closed_left = is_eyelids_touching(left_eye)
    touch_closed_right = is_eyelids_touching(right_eye)
    
    # Method 3: Check if iris is not visible (for MediaPipe)
    left_iris, right_iris = get_iris_centers_mediapipe(frame)
    
    # Combine all methods - eye is considered closed if any method says it's closed
    left_closed = ear_closed_left or touch_closed_left or (left_iris is None)
    right_closed = ear_closed_right or touch_closed_right or (right_iris is None)
    
    return left_closed, right_closed

def detect_gaze_direction(left_iris, right_iris, eye_landmarks):
    """Detect gaze direction with improved accuracy"""
    if not left_iris or not right_iris or not eye_landmarks or len(eye_landmarks) < 6:
        return None
    
    # Get eye boundaries
    eye_left = min(eye_landmarks, key=lambda p: p[0])[0]
    eye_right = max(eye_landmarks, key=lambda p: p[0])[0]
    eye_width = max(eye_right - eye_left, 1)  # Avoid division by zero
    
    # Calculate iris position ratios
    left_ratio = (left_iris[0] - eye_left) / eye_width
    right_ratio = (right_iris[0] - eye_left) / eye_width
    
    # Average the ratios for both eyes
    gaze_ratio = (left_ratio + right_ratio) / 2
    
    # Determine gaze direction
    if gaze_ratio < 0.4:
        return "left"
    elif gaze_ratio > 0.6:
        return "right"
    else:
        return "center"

def get_face_landmarks(frame):
    img_size = 640
    img0 = frame.copy()
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)[0]

    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    faces = []
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                faces.append((x1, y1, x2, y2))

    if not faces:
        return None

    # Pick the largest face
    face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = face

    # Expand the box a bit
    margin = 20
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, frame.shape[1])
    y2 = min(y2 + margin, frame.shape[0])

    # Crop the face and convert to gray
    face_img = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Detect landmarks on the cropped face
    rect = dlib.rectangle(0, 0, x2 - x1, y2 - y1)
    shape = predictor(gray, rect)

    # Adjust landmark coordinates back to full frame
    landmarks = [(p.x + x1, p.y + y1) for p in shape.parts()]
    return landmarks

def draw_landmarks(frame, landmarks):
    """Draw facial landmarks on the frame."""
    if landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

def perform_registration_movements(cap, name):
    # Debugging
    print("\nPlease follow the movements for registration:")
    print("1. Close your left eye (5 seconds)")
    print("2. Close your right eye (5 seconds)")
    print("3. Look left (5 seconds)")
    print("4. Look right (5 seconds)")
    print("5. Neutral face (3 seconds)")
    print("Press 'q' to cancel registration")

    movements = [
        {"name": "close your left eye", "completed": False, "frames": 0, "images": [], "timer": 0, "max_time": 5},
        {"name": "close your right eye", "completed": False, "frames": 0, "images": [], "timer": 0, "max_time": 5},
        {"name": "look left", "completed": False, "frames": 0, "images": [], "timer": 0, "max_time": 5},
        {"name": "look right", "completed": False, "frames": 0, "images": [], "timer": 0, "max_time": 5},
        {"name": "neutral face", "completed": False, "frames": 0, "images": [], "timer": 0, "max_time": 3}
    ]

    current_movement = 0
    all_captured_images = []
    last_capture_time = time.time()
    movement_start_time = time.time()
    last_frame_time = time.time()

    movement_dir = os.path.join(KNOWN_FACES_DIR, name, name)
    os.makedirs(movement_dir, exist_ok=True)

    while current_movement < len(movements):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time

        landmarks = get_face_landmarks(frame)
        movement_info = movements[current_movement]
        movement_info["timer"] += frame_time
        
        if landmarks:
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Detect eye states
            left_closed, right_closed = detect_eye_closure(landmarks, frame)
            
            # Get iris positions and gaze direction
            left_iris, right_iris = get_iris_centers_mediapipe(frame)
            gaze_direction = detect_gaze_direction(left_iris, right_iris, left_eye + right_eye)
            
            # Draw landmarks and information
            draw_landmarks(frame, landmarks)
            if left_iris:
                cv2.circle(frame, left_iris, 2, (255, 0, 0), -1)
            if right_iris:
                cv2.circle(frame, right_iris, 2, (0, 0, 255), -1)
                
            # Display current state information
            cv2.putText(frame, f"Left Eye: {'Closed' if left_closed else 'Open'}", (10, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Right Eye: {'Closed' if right_closed else 'Open'}", (10, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Gaze: {gaze_direction if gaze_direction else 'Unknown'}", (10, 260), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Capture images at intervals
            if current_time - last_capture_time > CAPTURE_INTERVAL:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_filename = os.path.join(movement_dir, f"{current_movement}_{movement_info['name']}_{timestamp}.jpg")
                cv2.imwrite(img_filename, frame)
                movement_info["images"].append(frame.copy())
                all_captured_images.append(frame.copy())
                last_capture_time = current_time

            # Check movement completion
            if not movement_info["completed"]:
                if current_movement == 0:  # Close left eye
                    if not get_iris_centers_mediapipe(frame)[0]:
                        movement_info["frames"] += 1
                        if movement_info["frames"] >= EYE_AR_CONSEC_FRAMES:
                            movement_info["completed"] = True
                            current_movement += 1
                            movement_start_time = time.time()
                            print(f"Completed: {movement_info['name']}")
                    else:
                        movement_info["frames"] = 0

                elif current_movement == 1:  # Close right eye
                    if not get_iris_centers_mediapipe(frame)[1]:
                        movement_info["frames"] += 1
                        if movement_info["frames"] >= EYE_AR_CONSEC_FRAMES:
                            movement_info["completed"] = True
                            current_movement += 1
                            movement_start_time = time.time()
                            print(f"Completed: {movement_info['name']}")
                    else:
                        movement_info["frames"] = 0

                elif current_movement == 2:  # Look left
                    if gaze_direction == "left":
                        movement_info["completed"] = True
                        current_movement += 1
                        movement_start_time = time.time()
                        print(f"Completed: {movement_info['name']}")

                elif current_movement == 3:  # Look right
                    if gaze_direction == "right":
                        movement_info["completed"] = True
                        current_movement += 1
                        movement_start_time = time.time()
                        print(f"Completed: {movement_info['name']}")

                elif current_movement == 4:  # Neutral face
                    if (not left_closed and not right_closed and 
                        gaze_direction == "center" and
                        movement_info["timer"] >= movement_info["max_time"]):
                        movement_info["completed"] = True
                        current_movement += 1
                        print(f"Completed: {movement_info['name']}")

                # Timeout if movement takes too long
                if movement_info["timer"] >= movement_info["max_time"]:
                    movement_info["completed"] = True
                    current_movement += 1
                    movement_start_time = time.time()
                    print(f"Timeout: {movement_info['name']}")

        # Display movement instructions
        remaining_time = max(0, movement_info["max_time"] - movement_info["timer"])
        cv2.putText(frame, f"Current: {movement_info['name']} ({remaining_time:.1f}s)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display movement progress
        for i, mov in enumerate(movements):
            status = "✓" if mov["completed"] else "✗" if i < current_movement else "•"
            color = (0, 255, 0) if mov["completed"] else (0, 0, 255) if i < current_movement else (255, 255, 255)
            cv2.putText(frame, f"{status} {mov['name']} ({len(mov['images'])} images)", 
                       (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Registration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled")
            return False

    if all_captured_images:
        return register_user(name, all_captured_images)
    return False


def start_face_registration(name):  # Name is passed from the main app
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    #name = input("Enter the name for registration: ")
    #if not name:
    #    print("Error: Name cannot be empty!")
    #    return

    perform_registration_movements(cap, name)
    cap.release()
    cv2.destroyAllWindows()

def clean_user_directory(username):
    user_dir = os.path.join(KNOWN_FACES_DIR, username)
    nested_dir = os.path.join(user_dir, username)

    # Step 1: Delete .jpg files in /known_faces/user1/
    deleted_files = 0
    for file in os.listdir(user_dir):
        file_path = os.path.join(user_dir, file)
        if os.path.isfile(file_path) and file.lower().endswith(".jpg"):
            os.remove(file_path)
            deleted_files += 1

    # Step 2: Move .jpg files from /known_faces/user1/user1/ to /known_faces/user1/
    moved_files = 0
    if os.path.exists(nested_dir):
        for file in os.listdir(nested_dir):
            src = os.path.join(nested_dir, file)
            dst = os.path.join(user_dir, file)
            if os.path.isfile(src) and file.lower().endswith(".jpg"):
                os.rename(src, dst)
                moved_files += 1

        try:
            os.rmdir(nested_dir)
        except OSError as e:
            print(f"Could not delete folder {nested_dir}: {e}")

def keep_last_per_movement(username):
    user_dir = os.path.join(KNOWN_FACES_DIR, username)
    if not os.path.exists(user_dir):
        print(f"Directory not found: {user_dir}")
        return

    movement_files = {}

    # Group files by movement prefix (e.g., 0_, 1_, etc.)
    for file in os.listdir(user_dir):
        if file.lower().endswith(".jpg"):
            match = re.match(r"^(\d+)_", file)
            if match:
                key = match.group(1)
                movement_files.setdefault(key, []).append(file)

    deleted = 0
    for key, files in movement_files.items():
        files.sort()  # Sort alphabetically, assuming later ones sort later
        # Keep the last, delete the others
        for file_to_delete in files[:-1]:  # all except the last
            file_path = os.path.join(user_dir, file_to_delete)
            os.remove(file_path)
            deleted += 1

    print(f"Cleanup done. Deleted {deleted} files, keeping the last of each movement.")

if __name__ == "__main__":
    start_face_registration()