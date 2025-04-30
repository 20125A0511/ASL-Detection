import os
import csv
import socket
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from flask_socketio import SocketIO, emit
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

from utils.cvfpscalc import CvFpsCalc

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching during development
socketio = SocketIO(app, async_mode='threading')

# Check if the directory structure exists, create if not
os.makedirs('templates', exist_ok=True)
os.makedirs('utils', exist_ok=True)
os.makedirs('model/keypoint_classifier', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Create cvfpscalc.py if not exists
if not os.path.exists('utils/cvfpscalc.py'):
    with open('utils/cvfpscalc.py', 'w') as f:
        f.write('''
import time
import cv2 as cv

class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = []
        self._buffer_len = buffer_len

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)
        if len(self._difftimes) > self._buffer_len:
            self._difftimes.pop(0)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
''')

# Create keypoint_classifier.py if not exists
if not os.path.exists('model/keypoint_classifier/keypoint_classifier.py'):
    with open('model/keypoint_classifier/keypoint_classifier.py', 'w') as f:
        f.write('''
import numpy as np
import tensorflow.lite as tflite
import os

class KeyPointClassifier(object):
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._confidence = 0.0
        
        # Create a simple model for default
        model_path = os.path.join(os.path.dirname(__file__), 'keypoint_classifier.tflite')
        
        # If model doesn't exist, create a dummy model for testing
        if not os.path.exists(model_path):
            self.interpreter = None
        else:
            # Model loading
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        if self.interpreter is None:
            self._confidence = 0.9  # Dummy confidence
            return 0  # Return default class
            
        input_details_tensor_index = self.input_details[0]['index']
        
        # Inference implementation
        input_tensor = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(input_details_tensor_index, input_tensor)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        
        # Save confidence score
        self._confidence = np.squeeze(result)[result_index]
        
        return result_index

    def get_confidence(self):
        return float(self._confidence)
''')

# Create keypoint_classifier_label.csv if not exists
if not os.path.exists('model/keypoint_classifier/keypoint_classifier_label.csv'):
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', 'w') as f:
        f.write('''A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
''')

# Create HTML templates
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Hand Sign Recognition</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.4.1/socket.io.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        #video-container {
            margin: 20px 0;
            position: relative;
        }
        #video-feed {
            width: 100%;
            max-width: 640px;
            border-radius: 10px;
        }
        #text-display {
            font-size: 48px;
            margin: 20px 0;
            min-height: 60px;
            font-weight: bold;
        }
        #text-history {
            font-size: 24px;
            margin: 20px 0;
            min-height: 100px;
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        .button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        .button:hover {
            background-color: #45a049;
        }
        #clear-btn {
            background-color: #f44336;
        }
        #clear-btn:hover {
            background-color: #d32f2f;
        }
        #status {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Hand Sign Recognition</h1>
        
        <div id="video-container">
            <img id="video-feed" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>
        
        <div id="text-display">{{ recognized_text }}</div>
        
        <div>
            <p>History:</p>
            <div id="text-history"></div>
        </div>
        
        <div>
            <button id="space-btn" class="button">Space</button>
            <button id="backspace-btn" class="button">Backspace</button>
            <button id="clear-btn" class="button">Clear</button>
            <button id="save-btn" class="button">Save</button>
        </div>
        
        <p id="status">Waiting for connection...</p>
    </div>

    <script>
        // Connect to Socket.IO server
        const socket = io();
        let textHistory = '';
        const textDisplay = document.getElementById('text-display');
        const textHistory_el = document.getElementById('text-history');
        const statusEl = document.getElementById('status');
        
        // Connection events
        socket.on('connect', function() {
            statusEl.textContent = 'Connected';
        });
        
        socket.on('disconnect', function() {
            statusEl.textContent = 'Disconnected';
        });
        
        socket.on('connection_response', function(data) {
            statusEl.textContent = data.status;
        });
        
        // Button handlers
        document.getElementById('space-btn').addEventListener('click', function() {
            addToHistory(' ');
        });
        
        document.getElementById('backspace-btn').addEventListener('click', function() {
            if (textHistory.length > 0) {
                textHistory = textHistory.slice(0, -1);
                textHistory_el.textContent = textHistory;
            }
        });
        
        document.getElementById('clear-btn').addEventListener('click', function() {
            textHistory = '';
            textHistory_el.textContent = textHistory;
        });
        
        document.getElementById('save-btn').addEventListener('click', function() {
            saveText(textHistory);
        });
        
        // Add text to history
        function addToHistory(text) {
            if (text) {
                textHistory += text;
                textHistory_el.textContent = textHistory;
            }
        }
        
        // Save text to server
        function saveText(text) {
            fetch('/save_and_exit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Text saved successfully!');
                    window.location.href = '/exit';
                } else {
                    alert('Error saving text: ' + data.message);
                }
            })
            .catch(error => {
                alert('Error: ' + error);
            });
        }
        
        // Letter recognition variables
        let lastLetter = '';
        let lastAddedLetter = '';
        let letterHeldCount = 0;
        let steadyStateReached = false;
        const LETTER_HOLD_THRESHOLD = 10; // Number of consecutive same letter recognitions
        const STEADY_STATE_COOLDOWN = 2000; // 2 seconds before accepting same letter again
        let lastLetterTime = 0;
        
        // Listen for letter updates from server
        socket.on('text_update', function(data) {
            const currentTime = Date.now();
            const letterFromServer = data.text;
            
            // Update the display with the current recognized letter
            if (letterFromServer) {
                textDisplay.textContent = letterFromServer;
                
                // Logic for determining when to add a letter to history
                if (letterFromServer === lastLetter) {
                    // Same letter detected
                    letterHeldCount++;
                    
                    // If we reach threshold and haven't added this letter yet
                    if (letterHeldCount >= LETTER_HOLD_THRESHOLD && !steadyStateReached) {
                        addToHistory(letterFromServer);
                        lastAddedLetter = letterFromServer;
                        steadyStateReached = true;
                        lastLetterTime = currentTime;
                    }
                } else {
                    // New letter detected, reset counters
                    lastLetter = letterFromServer;
                    letterHeldCount = 1;
                    steadyStateReached = false;
                }
            }
            
            // Reset steady state after cooldown to allow same letter again
            if (steadyStateReached && (currentTime - lastLetterTime > STEADY_STATE_COOLDOWN)) {
                steadyStateReached = false;
            }
        });
    </script>
</body>
</html>
''')

with open('templates/exit.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Hand Sign Recognition - Exit</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        .button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 20px;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Session Ended</h1>
        <p>Your text has been saved successfully.</p>
        <p>Thank you for using Hand Sign Recognition.</p>
        <a href="/" class="button">Start New Session</a>
    </div>
</body>
</html>
''')

# Import needed for KeyPointClassifier
try:
    import tensorflow.lite as tflite
except ImportError:
    pass  # We'll handle this in the KeyPointClassifier class

# Initialize global variables
recognized_text = ""
cap = None
hands = None
keypoint_classifier = None
keypoint_classifier_labels = None
last_prediction_time = 0
PREDICTION_INTERVAL = 0.3  # Shorter interval for more responsive detection
last_prediction = None
PREDICTION_THRESHOLD = 0.5  # Lower threshold for easier detection
PREDICTION_STABILITY = 3  # Number of consistent predictions needed before changing

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
    raise OSError("No available ports found")

def init_resources():
    global hands, keypoint_classifier, keypoint_classifier_labels, recognized_text, recent_predictions
    
    try:
        # Reset recognized text and predictions
        recognized_text = ""
        recent_predictions = []
        
        # Initialize MediaPipe Hands with improved settings
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,  # Lowered for better detection
            min_tracking_confidence=0.7,   # Lowered for better tracking
            model_complexity=1             # Increased complexity for better accuracy
        )
        
        # Initialize keypoint classifier
        keypoint_classifier = KeyPointClassifier()
        
        # Load labels
        with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
            print("Loaded labels:", keypoint_classifier_labels)
            
    except Exception as e:
        print(f"Error initializing resources: {str(e)}")
        raise

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
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                return True
        return cap is not None and cap.isOpened()
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
        return False

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
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

def is_stable_prediction(prediction):
    """Check if the prediction is stable by looking at recent history."""
    global recent_predictions
    
    # Add the new prediction to history
    recent_predictions.append(prediction)
    
    # Keep only the most recent predictions
    if len(recent_predictions) > PREDICTION_STABILITY:
        recent_predictions.pop(0)
    
    # Check if all recent predictions are the same
    return len(recent_predictions) >= PREDICTION_STABILITY and all(p == recent_predictions[0] for p in recent_predictions)

def generate_frames():
    global last_prediction_time, last_prediction, recognized_text
    
    if not init_camera():
        print("Error: Could not initialize camera")
        return
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    fps_calc = CvFpsCalc(buffer_len=10)
    
    while True:
        try:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from webcam")
                time.sleep(0.1)  # Add small delay to prevent CPU overload
                continue
                
            frame = cv.flip(frame, 1)  # Mirror image
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            # Calculate FPS
            fps = fps_calc.get()
            
            # Process image with MediaPipe
            results = hands.process(frame_rgb)
            
            current_time = time.time()
            
            if results.multi_hand_landmarks:
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
                        
                        # Classify hand sign
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
                                socketio.emit('text_update', {
                                    'text': predicted_letter,
                                    'confidence': float(confidence),
                                    'class_id': int(hand_sign_id)
                                })
                                print(f"Emitted prediction: {predicted_letter} (Class: {hand_sign_id}, Confidence: {confidence:.2f})")
                            
                    except Exception as e:
                        print(f"Error in hand sign classification: {str(e)}")
                        continue
            
            # Show FPS on frame
            cv.putText(frame, f"FPS: {fps}", (10, frame.shape[0] - 10), 
                     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                     
            # Compress frame with lower quality for faster transmission
            ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                print("Error: Could not encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            time.sleep(0.1)  # Add small delay to prevent CPU overload
            continue

@app.route('/')
def index():
    # Initialize resources when starting a new session
    init_resources()
    return render_template('index.html', recognized_text=recognized_text)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

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
    init_resources()
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    cleanup()

def cleanup():
    global cap, hands, recognized_text
    print("Cleaning up resources")
    
    # Reset recognized text
    recognized_text = ""
    
    # Release camera
    if cap is not None:
        try:
            cap.release()
        except:
            pass
        cap = None
    
    # Close MediaPipe hands
    if hands is not None:
        try:
            hands.close()
        except:
            pass
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
            host='0.0.0.0',
            port=port,
            use_reloader=False,
            allow_unsafe_werkzeug=True  # Add this to prevent trace trap
        )
    except Exception as e:
        print(f"Error starting application: {str(e)}")
    finally:
        cleanup()