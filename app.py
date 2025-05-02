import os
import csv
import socket
import traceback
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from flask_socketio import SocketIO, emit
import sys

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching during development
socketio = SocketIO(app, async_mode='threading', logger=True, engineio_logger=True)

# Check if the directory structure exists, create if not
os.makedirs('templates', exist_ok=True)
os.makedirs('utils', exist_ok=True)
os.makedirs('model/keypoint_classifier', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Import needed for KeyPointClassifier
try:
    from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
except ImportError:
    # If import fails, create a simple stub class for testing
    class KeyPointClassifier:
        def __init__(self):
            self._confidence = 0.9
            
        def __call__(self, landmark_list):
            return 0
            
        def get_confidence(self):
            return self._confidence

# Initialize global variables
recognized_text = ""
cap = None
hands = None
keypoint_classifier = None
keypoint_classifier_labels = []
last_prediction_time = 0
PREDICTION_INTERVAL = 0.3  # Shorter interval for more responsive detection
last_prediction = None
PREDICTION_THRESHOLD = 0.5  # Lower threshold for easier detection
PREDICTION_STABILITY = 3  # Number of consistent predictions needed before changing

# Training mode variables
is_training_mode = False
current_training_letter = None
training_samples = {}  # Dictionary to track sample counts for each letter
SAMPLES_PER_LETTER = 10  # Number of samples to capture per letter
last_sample_time = 0
SAMPLE_DELAY = 0.5  # Half-second delay between samples

# Store recent predictions for stability
recent_predictions = []

def find_available_port(start_port=5000, max_port=5050):
    for port in range(start_port, max_port + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', port))
            s.close()
            return port
        except OSError:
            continue
    return 5000  # Default to 5000 if no ports are available

def init_resources():
    global hands, keypoint_classifier, keypoint_classifier_labels, recognized_text, recent_predictions, training_samples
    
    try:
        # Reset recognized text and predictions
        recognized_text = ""
        recent_predictions = []
        training_samples = {}
        
        # Initialize MediaPipe Hands with improved settings
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,  
                min_tracking_confidence=0.7,   
                model_complexity=1             
            )
        except Exception as e:
            print(f"Error initializing MediaPipe Hands: {str(e)}")
            hands = None
        
        # Initialize keypoint classifier
        try:
            keypoint_classifier = KeyPointClassifier()
        except Exception as e:
            print(f"Error initializing KeyPointClassifier: {str(e)}")
            keypoint_classifier = None
        
        # Load labels
        try:
            if os.path.exists("model/keypoint_classifier/keypoint_classifier_label.csv"):
                with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
                    keypoint_classifier_labels = csv.reader(f)
                    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
                    print("Loaded labels:", keypoint_classifier_labels)
            else:
                # Default labels if file doesn't exist
                keypoint_classifier_labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]
                print("Using default labels (A-Z)")
        except Exception as e:
            print(f"Error loading labels: {str(e)}")
            keypoint_classifier_labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]
            
    except Exception as e:
        print(f"Error initializing resources: {str(e)}")
        traceback.print_exc()

def init_camera():
    global cap
    try:
        if cap is None or not cap.isOpened():
            cap = cv.VideoCapture(0)
            if cap.isOpened():
                # Set lower resolution for better stability
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv.CAP_PROP_FPS, 30)
                return True
        return cap is not None and cap.isOpened()
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
        traceback.print_exc()
        return False

def calc_landmark_list(image, landmarks):
    try:
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    except Exception as e:
        print(f"Error calculating landmarks: {str(e)}")
        return []

def pre_process_landmark(landmark_list):
    try:
        temp_landmark_list = landmark_list.copy()

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = np.array(temp_landmark_list).flatten()

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value != 0:
            temp_landmark_list = temp_landmark_list / max_value

        return temp_landmark_list
    except Exception as e:
        print(f"Error pre-processing landmarks: {str(e)}")
        return np.zeros(42)  # Return empty array with enough dimensions

def is_stable_prediction(prediction):
    """Check if the prediction is stable by looking at recent history."""
    global recent_predictions
    
    try:
        # Add the new prediction to history
        recent_predictions.append(prediction)
        
        # Keep only the most recent predictions
        if len(recent_predictions) > PREDICTION_STABILITY:
            recent_predictions.pop(0)
        
        # Check if all recent predictions are the same
        return len(recent_predictions) >= PREDICTION_STABILITY and all(p == recent_predictions[0] for p in recent_predictions)
    except Exception as e:
        print(f"Error checking prediction stability: {str(e)}")
        return False

def save_training_sample(landmark_list, letter):
    """Save a training sample to the keypoint.csv file"""
    global training_samples
    
    try:
        # Convert landmark list to string format
        landmark_str = ','.join(map(str, landmark_list))
        
        # Create the keypoint.csv file if it doesn't exist or is empty
        file_exists = os.path.exists('model/keypoint_classifier/keypoint.csv')
        if not file_exists or os.path.getsize('model/keypoint_classifier/keypoint.csv') == 0:
            with open('model/keypoint_classifier/keypoint.csv', 'w') as f:
                f.write("")  # Create empty file
        
        # Append to CSV file
        with open('model/keypoint_classifier/keypoint.csv', 'a') as f:
            f.write(f"{letter},{landmark_str}\n")
        
        # Update sample count
        if letter not in training_samples:
            training_samples[letter] = 0
        training_samples[letter] += 1
        
        return True
    except Exception as e:
        print(f"Error saving training sample: {str(e)}")
        traceback.print_exc()
        return False

def generate_frames():
    global last_prediction_time, last_prediction, recognized_text, current_training_letter, last_sample_time, training_samples
    
    if not init_camera():
        print("Error: Could not initialize camera")
        return
    
    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except Exception as e:
        print(f"Error initializing MediaPipe drawing utilities: {str(e)}")
        traceback.print_exc()
        return

    # Initialize FPS calculator if available
    try:
        from utils.cvfpscalc import CvFpsCalc
        fps_calc = CvFpsCalc(buffer_len=10)
    except ImportError:
        # Create simple FPS calculator if import fails
        class SimpleFpsCalc:
            def __init__(self):
                self.prev_time = time.time()
            def get(self):
                current_time = time.time()
                fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time
                return round(fps, 2)
        fps_calc = SimpleFpsCalc()
    
    while True:
        try:
            # Check if camera is still open
            if cap is None or not cap.isOpened():
                if not init_camera():
                    time.sleep(0.1)
                    continue
            
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from webcam")
                time.sleep(0.1)  # Add small delay to prevent CPU overload
                continue
                
            frame = cv.flip(frame, 1)  # Mirror image
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Calculate FPS
            fps = fps_calc.get()
            
            # Process image with MediaPipe if hands is available
            results = None
            if hands is not None:
                results = hands.process(frame_rgb)
            
            current_time = time.time()
            
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp.solutions.hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Calculate landmarks
                        landmark_list = calc_landmark_list(frame, hand_landmarks)
                        
                        # Pre-process landmarks
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        
                        if is_training_mode and current_training_letter:
                            # Check if we're ready to capture another sample
                            if current_time - last_sample_time >= SAMPLE_DELAY:
                                # Check if we've already captured enough samples for this letter
                                current_count = training_samples.get(current_training_letter, 0)
                                
                                if current_count < SAMPLES_PER_LETTER:
                                    # Save training sample
                                    if save_training_sample(pre_processed_landmark_list, current_training_letter):
                                        current_count += 1
                                        last_sample_time = current_time
                                        
                                        # Display text on video feed 
                                        cv.putText(frame, f"Captured {current_count}/{SAMPLES_PER_LETTER} for {current_training_letter}", 
                                                 (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        
                                        # Send progress update to client
                                        try:
                                            socketio.emit('training_update', {
                                                'success': True,
                                                'letter': current_training_letter,
                                                'count': current_count,
                                                'total': SAMPLES_PER_LETTER,
                                                'status': f'Captured sample {current_count}/{SAMPLES_PER_LETTER} for letter {current_training_letter}'
                                            })
                                        except Exception as e:
                                            print(f"Error emitting training update: {str(e)}")
                                        
                                        # Check if we've captured all samples for this letter
                                        if current_count >= SAMPLES_PER_LETTER:
                                            try:
                                                socketio.emit('training_update', {
                                                    'success': True,
                                                    'letter': current_training_letter,
                                                    'completed': True,
                                                    'status': f'Completed capturing all samples for letter {current_training_letter}'
                                                })
                                            except Exception as e:
                                                print(f"Error emitting training completion: {str(e)}")
                                            current_training_letter = None  # Reset after completing all samples
                                else:
                                    # Already captured all samples for this letter
                                    cv.putText(frame, f"Completed {SAMPLES_PER_LETTER}/{SAMPLES_PER_LETTER} for {current_training_letter}", 
                                             (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    try:
                                        socketio.emit('training_update', {
                                            'success': True,
                                            'letter': current_training_letter,
                                            'completed': True,
                                            'status': f'Already completed capturing samples for letter {current_training_letter}'
                                        })
                                    except Exception as e:
                                        print(f"Error emitting training status: {str(e)}")
                                    current_training_letter = None  # Reset after completing all samples
                        elif keypoint_classifier is not None and len(keypoint_classifier_labels) > 0:
                            # Classify hand sign
                            try:
                                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                                
                                if hand_sign_id >= 0 and hand_sign_id < len(keypoint_classifier_labels):
                                    # Get confidence score
                                    confidence = keypoint_classifier.get_confidence()
                                    
                                    # Get the predicted letter
                                    predicted_letter = keypoint_classifier_labels[hand_sign_id]
                                    
                                    # Display prediction and confidence on frame
                                    cv.putText(frame, f"{predicted_letter} ({confidence:.2f})", 
                                             (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    
                                    # Emit prediction to frontend if confidence is high enough
                                    if confidence > 0.3:  # Lowered threshold for better detection
                                        try:
                                            socketio.emit('text_update', {
                                                'text': predicted_letter,
                                                'confidence': float(confidence),
                                                'class_id': int(hand_sign_id)
                                            })
                                        except Exception as e:
                                            print(f"Error emitting text update: {str(e)}")
                                        print(f"Emitted prediction: {predicted_letter} (Class: {hand_sign_id}, Confidence: {confidence:.2f})")
                            except Exception as e:
                                print(f"Error in prediction: {str(e)}")
                    except Exception as e:
                        print(f"Error in hand sign classification: {str(e)}")
                        continue
            
            # Show FPS on frame
            cv.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10), 
                     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                     
            # Compress frame with lower quality for faster transmission
            try:
                ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    print("Error: Could not encode frame")
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error encoding frame: {str(e)}")
                time.sleep(0.1)
                continue
            
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            traceback.print_exc()
            time.sleep(0.1)  # Add small delay to prevent CPU overload
            continue

@app.route('/')
def index():
    # Initialize resources when starting a new session
    try:
        init_resources()
    except Exception as e:
        print(f"Error initializing resources: {str(e)}")
        traceback.print_exc()
    return render_template('index.html', recognized_text=recognized_text)

@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed: {str(e)}")
        traceback.print_exc()
        return "Video feed error", 500

@app.route('/save_and_exit', methods=['POST'])
def save_and_exit():
    try:
        data = request.get_json()
        text = data.get('text', '')
        print(f"Saving text: {text}")  # Debug log
        
        # Save the text to a file
        with open('saved_text.txt', 'w') as f:
            f.write(text)
            
        return jsonify({
            'success': True,
            'message': 'Text saved successfully'
        })
    except Exception as e:
        print(f"Error saving text: {str(e)}")  # Debug log
        return jsonify({
            'success': False,
            'message': f'Error saving text: {str(e)}'
        })

@app.route('/exit')
def exit_page():
    print("Exit page requested")
    cleanup()
    return render_template('exit.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Initialize resources when a new client connects
    try:
        init_resources()
        emit('connection_response', {'status': 'connected'})
    except Exception as e:
        print(f"Error in socket connect: {str(e)}")
        traceback.print_exc()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    try:
        cleanup()
    except Exception as e:
        print(f"Error in socket disconnect: {str(e)}")
        traceback.print_exc()

@socketio.on('start_training')
def handle_start_training():
    global is_training_mode, training_samples
    try:
        is_training_mode = True
        
        # Send current training progress to client
        emit('training_update', {
            'status': 'Training mode started',
            'progress': training_samples
        })
    except Exception as e:
        print(f"Error starting training: {str(e)}")
        traceback.print_exc()

@socketio.on('stop_training')
def handle_stop_training():
    global is_training_mode, current_training_letter
    try:
        is_training_mode = False
        current_training_letter = None
        emit('training_update', {'status': 'Training mode stopped'})
    except Exception as e:
        print(f"Error stopping training: {str(e)}")
        traceback.print_exc()

@socketio.on('capture_sample')
def handle_capture_sample(data):
    global current_training_letter, training_samples
    try:
        letter = data.get('letter')
        if letter:
            current_training_letter = letter
            current_count = training_samples.get(letter, 0)
            
            emit('training_update', {
                'success': True,
                'letter': current_training_letter,
                'count': current_count,
                'total': SAMPLES_PER_LETTER,
                'status': f'Ready to capture samples for letter {current_training_letter} ({current_count}/{SAMPLES_PER_LETTER})'
            })
        else:
            emit('training_update', {
                'success': False,
                'status': 'Please select a valid letter'
            })
    except Exception as e:
        print(f"Error capturing sample: {str(e)}")
        traceback.print_exc()

def cleanup():
    global cap, hands, recognized_text
    print("Cleaning up resources")
    
    # Reset recognized text
    recognized_text = ""
    
    # Release camera
    if cap is not None:
        try:
            cap.release()
        except Exception as e:
            print(f"Error releasing camera: {str(e)}")
        cap = None
    
    # Close MediaPipe hands
    if hands is not None:
        try:
            hands.close()
        except Exception as e:
            print(f"Error closing MediaPipe: {str(e)}")
        hands = None

if __name__ == "__main__":
    try:
        # Initialize resources
        init_resources()
        
        # Find available port
        port = find_available_port()
        print(f"Starting server on port {port}")
        
        # Run the application with proper configuration
        socketio.run(
            app,
            debug=False,
            host='127.0.0.1',  # Use localhost instead of 0.0.0.0
            port=port,
            use_reloader=False,
            allow_unsafe_werkzeug=True  # Add this to prevent trace trap
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting...")
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        traceback.print_exc()
    finally:
        cleanup()
        print("Application shut down")