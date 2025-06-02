from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
import cv2
import os
import threading
import time
from registration import get_face_landmarks, detect_eye_closure, get_iris_centers_mediapipe, detect_gaze_direction, draw_landmarks, clean_user_directory, keep_last_per_movement
from video_predict import FaceAuthenticator, make_prediction
import torch
from src.FaceAntiSpoofing import AntiSpoof

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Global variables
camera = None
registration_active = False
authentication_active = False
current_user = None
current_instruction = None
registration_complete = False

# Initialize models
def initialize_models():
    print("Initializing models...")
    face_detector = torch.hub.load('ultralytics/yolov5', 'custom', 
                                 path='yolov5/runs/train/face-detector-from-scratch14/weights/best.pt', 
                                 force_reload=False)
    anti_spoof = AntiSpoof("saved_models/bestest.onnx")
    authenticator = FaceAuthenticator("known_faces", 0.5)
    print("Models initialized successfully")
    return face_detector, anti_spoof, authenticator

face_detector, anti_spoof, authenticator = initialize_models()

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        # Allow time for camera to initialize
        time.sleep(2)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


KNOWN_FACES_DIR = os.path.join(os.getcwd(), "known_faces")
# Clean feed for registration
def generate_registration_frames():
    global registration_state
    cap = get_camera()
    face_images = []    # List to store images for registration

    try:
        while registration_state['active']:
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            
            # Your facial recognition processing
            landmarks = get_face_landmarks(frame)
            if landmarks:
                left_closed, right_closed = detect_eye_closure(landmarks, frame)
                left_iris, right_iris = get_iris_centers_mediapipe(frame)
                gaze_direction = detect_gaze_direction(left_iris, right_iris, landmarks[36:42] + landmarks[42:48])
                
                # Update registration state based on detection
                current_step = registration_state['current_step']
                if current_step < registration_state['total_steps']:
                    current_movement = registration_state['steps'][current_step]['name']
                    
                    # Check if current movement is completed
                    if (current_movement == 'close left eye' and left_closed and not right_closed) or \
                       (current_movement == 'close right eye' and right_closed and not left_closed) or \
                       (current_movement == 'look left' and gaze_direction == "left") or \
                       (current_movement == 'look right' and gaze_direction == "right") or \
                       (current_movement == 'neutral face' and not left_closed and not right_closed and gaze_direction == "center"):
                        
                        username = registration_state.get('username')
                        if username:
                            user_dir = os.path.join(KNOWN_FACES_DIR, username)
                            os.makedirs(user_dir, exist_ok=True)

                            step_name = registration_state['steps'][current_step]['name'].replace(" ", "_").replace(":", "_")
                            image_index = len(os.listdir(user_dir)) + 1

                            img_path = os.path.join(user_dir, f"{current_step}_{step_name}_{image_index:04}.jpg")

                            if frame is not None and frame.size > 0:
                                success = cv2.imwrite(img_path, frame)
                                if success:
                                    print(f"Saved image to {img_path}")
                                    face_images.append(frame.copy())
                                else:
                                    print(f"Failed to save image to {img_path}")
                            else:
                                print("Frame is empty or invalid, skipping save.")
                        else:
                            print("Username not set, cannot save frame.")

                        registration_state['steps'][current_step]['completed'] = True
                        registration_state['current_step'] += 1

                        # Finalization
                        if registration_state['current_step'] >= registration_state['total_steps']:
                            registration_state['complete'] = True
                            registration_state['active'] = False
                            if username:
                                # clean_user_directory(username=username)   # It was deleting everything, damn...
                                keep_last_per_movement(username=username)
                                authenticator.load_known_faces_from_folder()
                                print(f"Registration for {username} complete and face data loaded.")
                            else:
                                print("No user session found for finalization.")
            
            # Add visual feedback to frame
            draw_landmarks(frame, landmarks)
            cv2.putText(frame, f"Current: {registration_state['steps'][current_step]['name']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    finally:
        pass  # Camera released elsewhere

# Processed feed for authentication
def generate_authentication_frames():
    global current_user
    cam = get_camera()

    consecutive_successes = 0
    required_successes = 5
    authenticated_user = None

    try:
        while authentication_active:
            success, frame = cam.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)

            authenticated = False
            pred = make_prediction(frame, face_detector, anti_spoof, authenticator, 0.7)
            if pred:
                (x1, y1, x2, y2), label, score, auth_name, is_authenticated = pred

                if label == 0 and score > 0.7 and is_authenticated:
                    consecutive_successes += 1
                    authenticated_user = auth_name
                else:
                    consecutive_successes = 0  # Reset on failure

                color = (0, 255, 0) if consecutive_successes >= required_successes else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Optional: show progress on screen
                cv2.putText(frame, f"Authenticated: {consecutive_successes}/{required_successes}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Check if threshold reached
            if consecutive_successes >= required_successes and authenticated_user:
                current_user = authenticated_user
                authenticated = True

            if authenticated:
                cv2.putText(frame, f"Successfully Authenticated: {authenticated_user}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0, 255, 0), 2)


            # Yield frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        pass  # Do not release camera here


@app.route('/')
def index():
    release_camera()  # Ensure camera is released when not in use
    return render_template('index.html')

@app.route('/video_feed_registration')
def video_feed_registration():
    return Response(generate_registration_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_authentication')
def video_feed_authentication():
    return Response(generate_authentication_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


# Global state (replace your existing globals)
registration_state = {
    'active': False,
    'username': None,
    'current_step': 0,
    'total_steps': 5,  # Updated to match your steps
    'steps': [
        {'name': 'close left eye', 'completed': False},
        {'name': 'close right eye', 'completed': False}, 
        {'name': 'look left', 'completed': False},
        {'name': 'look right', 'completed': False},
        {'name': 'neutral face', 'completed': False}
    ],
    'complete': False,
    'current_instruction': 'Waiting to start...'
}

@app.route('/get_registration_status')
def get_registration_status():
    progress = int((registration_state['current_step'] / registration_state['total_steps']) * 100)
    return jsonify({
        'current_instruction': registration_state['steps'][registration_state['current_step']]['name']
            if registration_state['current_step'] < registration_state['total_steps'] else "Complete",
        'complete': registration_state['complete'],
        'current_step': registration_state['current_step'] + 1,
        'total_steps': registration_state['total_steps'],
        'percentage': progress,
        'steps': registration_state['steps']
    })


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        if not username:
            return render_template('register.html', error="Username cannot be empty!")
        
        # Initialize registration
        session['registering_user'] = username
        registration_state['username'] = username
        registration_state['active'] = True
        registration_state['complete'] = False
        registration_state['current_step'] = 0
        for step in registration_state['steps']:
            step['completed'] = False
        
        return redirect(url_for('registration_instructions'))
    
    return render_template('register.html')

@app.route('/registration_instructions')
def registration_instructions():
    username = session.get('registering_user', '')
    if not username:
        return redirect(url_for('register'))
    
    # Get current instruction from registration process
    instruction = current_instruction if registration_active else None
    
    return render_template('registration_instructions.html', 
                         username=username,
                         current_instruction=instruction,
                         complete=registration_complete)

@app.route('/registration_complete')
def registration_complete():
    username = session.get('registering_user', '')
    if not username:
        return redirect(url_for('register'))
    
    session.pop('registering_user', None)
    release_camera()
    return render_template('registration_complete.html', username=username)


@app.route('/login_status')
def login_status():
    if current_user:
        return jsonify({'authenticated': True, 'username': current_user})
    return jsonify({'authenticated': False})

@app.route('/login', methods=['GET', 'POST'])
def login():
    global authentication_active, current_user

    if request.method == 'POST':
        authentication_active = True
        return render_template('login.html', authenticating=True)

    return render_template('login.html', authenticating=authentication_active, authenticated_user=current_user)

@app.route('/login_success')
def login_success():
    global current_user, authentication_active
    user = request.args.get('username', current_user)
    current_user = None
    authentication_active = False
    release_camera()
    return render_template('login_success.html', username=user)


@app.route('/logout')
def logout():
    global authentication_active
    authentication_active = False
    release_camera()
    return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        release_camera()