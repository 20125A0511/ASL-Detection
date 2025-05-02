import os

def create_directories():
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    os.makedirs('model/keypoint_classifier', exist_ok=True)
    os.makedirs('static', exist_ok=True)

def create_cvfpscalc():
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

def create_keypoint_classifier():
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

def create_keypoint_classifier_label():
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

def create_templates():
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Automated AI-Based Sign Language Translator</title>
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
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        #text-history {
            font-size: 24px;
            margin: 20px 0;
            min-height: 100px;
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
            word-wrap: break-word;
            overflow-wrap: break-word;
            max-height: 200px;
            overflow-y: auto;
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
        #train-btn {
            background-color: #2196F3;
        }
        #train-btn:hover {
            background-color: #1976D2;
        }
        #status {
            color: #666;
            font-style: italic;
        }
        #training-container {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: none;
        }
        #alphabet-selector {
            margin: 10px 0;
            padding: 8px;
            font-size: 16px;
            width: 200px;
        }
        #capture-btn {
            background-color: #FF9800;
        }
        #capture-btn:hover {
            background-color: #F57C00;
        }
        #stop-training-btn {
            background-color: #9C27B0;
        }
        #stop-training-btn:hover {
            background-color: #7B1FA2;
        }
        .recognition-delay {
            animation: fadeOut 1s ease-in-out;
        }
        @keyframes fadeOut {
            0% { opacity: 1; }
            100% { opacity: 0; }
        }
        .progress-container {
            margin: 10px 0;
            background-color: #e0e0e0;
            border-radius: 5px;
            height: 20px;
            overflow: hidden;
        }
        .progress-bar {
            background-color: #4CAF50;
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
        }
        .training-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 5px;
            margin: 10px 0;
        }
        .letter-box {
            padding: 5px;
            border: 1px solid #ddd;
            text-align: center;
            font-weight: bold;
            position: relative;
        }
        .letter-complete {
            background-color: #c8e6c9;
        }
        .letter-progress {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            background-color: #4CAF50;
            width: 0%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Automated AI-Based Sign Language Translator</h1>
        
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
            <button id="train-btn" class="button">Train Dataset</button>
        </div>

        <div id="training-container">
            <h3>Training Mode</h3>
            <p>Select an alphabet and click "Capture Samples" to record training data.</p>
            <p>For each alphabet, we need to capture 10 samples.</p>
            
            <div class="training-grid" id="letter-grid">
                <!-- Will be filled dynamically -->
            </div>
            
            <select id="alphabet-selector">
                <option value="">Select Alphabet</option>
                <option value="A">A</option>
                <option value="B">B</option>
                <option value="C">C</option>
                <option value="D">D</option>
                <option value="E">E</option>
                <option value="F">F</option>
                <option value="G">G</option>
                <option value="H">H</option>
                <option value="I">I</option>
                <option value="J">J</option>
                <option value="K">K</option>
                <option value="L">L</option>
                <option value="M">M</option>
                <option value="N">N</option>
                <option value="O">O</option>
                <option value="P">P</option>
                <option value="Q">Q</option>
                <option value="R">R</option>
                <option value="S">S</option>
                <option value="T">T</option>
                <option value="U">U</option>
                <option value="V">V</option>
                <option value="W">W</option>
                <option value="X">X</option>
                <option value="Y">Y</option>
                <option value="Z">Z</option>
            </select>
            
            <div class="progress-container" id="sample-progress-container">
                <div class="progress-bar" id="sample-progress-bar"></div>
            </div>
            
            <button id="capture-btn" class="button">Capture Samples</button>
            <button id="stop-training-btn" class="button">Stop Training</button>
            <p id="training-status">Select an alphabet and click Capture Samples to start training</p>
        </div>
        
        <p id="status">Waiting for connection...</p>
    </div>

    <script>
        // Connect to Socket.IO server
        const socket = io();
        let textHistory = '';
        let lastRecognitionTime = 0;
        const RECOGNITION_DELAY = 2000; // 2 seconds delay between recognitions
        const textDisplay = document.getElementById('text-display');
        const textHistory_el = document.getElementById('text-history');
        const statusEl = document.getElementById('status');
        const trainingContainer = document.getElementById('training-container');
        const trainingStatus = document.getElementById('training-status');
        const alphabetSelector = document.getElementById('alphabet-selector');
        const progressBar = document.getElementById('sample-progress-bar');
        const letterGrid = document.getElementById('letter-grid');
        
        // Sample tracking
        const SAMPLES_PER_LETTER = 10;
        let trainingProgress = {};
        let currentlyTrainingLetter = null;
        
        // Initialize letter grid
        function initializeLetterGrid() {
            letterGrid.innerHTML = '';
            for (let charCode = 65; charCode <= 90; charCode++) {
                const letter = String.fromCharCode(charCode);
                const div = document.createElement('div');
                div.classList.add('letter-box');
                div.id = `letter-box-${letter}`;
                div.innerHTML = `
                    ${letter}
                    <div class="letter-progress" id="letter-progress-${letter}"></div>
                `;
                letterGrid.appendChild(div);
            }
        }
        
        // Update letter grid based on progress
        function updateLetterGrid() {
            for (const letter in trainingProgress) {
                const count = trainingProgress[letter];
                const boxEl = document.getElementById(`letter-box-${letter}`);
                const progressEl = document.getElementById(`letter-progress-${letter}`);
                
                if (boxEl && progressEl) {
                    const progressPercent = (count / SAMPLES_PER_LETTER) * 100;
                    progressEl.style.width = `${progressPercent}%`;
                    
                    if (count >= SAMPLES_PER_LETTER) {
                        boxEl.classList.add('letter-complete');
                    } else {
                        boxEl.classList.remove('letter-complete');
                    }
                }
            }
        }
        
        // Initialize the grid on page load
        initializeLetterGrid();
        
        // Connection events
        socket.on('connect', function() {
            statusEl.textContent = 'Connected';
        });
        
        socket.on('disconnect', function() {
            statusEl.textContent = 'Disconnected';
        });
        
        // Handle recognized text updates
        socket.on('text_update', function(data) {
            const currentTime = Date.now();
            if (currentTime - lastRecognitionTime >= RECOGNITION_DELAY) {
                textDisplay.textContent = data.text;
                if (data.text) {
                    textHistory += data.text;
                    textHistory_el.textContent = textHistory;
                    // Scroll to bottom of history
                    textHistory_el.scrollTop = textHistory_el.scrollHeight;
                }
                lastRecognitionTime = currentTime;
                
                // Add fade-out effect
                textDisplay.classList.add('recognition-delay');
                setTimeout(() => {
                    textDisplay.classList.remove('recognition-delay');
                }, 1000);
            }
        });
        
        // Training mode events
        socket.on('training_update', function(data) {
            if (data.progress) {
                trainingProgress = data.progress;
                updateLetterGrid();
            }
            
            if (data.letter && data.count !== undefined) {
                currentlyTrainingLetter = data.letter;
                
                // Update progress bar
                const progress = (data.count / data.total) * 100;
                progressBar.style.width = `${progress}%`;
                
                // Update training status
                trainingStatus.textContent = data.status || `Captured ${data.count}/${data.total} samples for ${data.letter}`;
                
                // Update training progress
                if (!trainingProgress[data.letter]) {
                    trainingProgress[data.letter] = 0;
                }
                trainingProgress[data.letter] = data.count;
                updateLetterGrid();
                
                // If completed, reset current training letter
                if (data.completed) {
                    currentlyTrainingLetter = null;
                    alphabetSelector.value = '';
                }
            } else {
                trainingStatus.textContent = data.status || 'Training status unknown';
            }
        });
        
        // Button event listeners
        document.getElementById('space-btn').addEventListener('click', function() {
            textHistory += ' ';
            textHistory_el.textContent = textHistory;
            textHistory_el.scrollTop = textHistory_el.scrollHeight;
        });
        
        document.getElementById('backspace-btn').addEventListener('click', function() {
            textHistory = textHistory.slice(0, -1);
            textHistory_el.textContent = textHistory;
            textHistory_el.scrollTop = textHistory_el.scrollHeight;
        });
        
        document.getElementById('clear-btn').addEventListener('click', function() {
            textHistory = '';
            textHistory_el.textContent = textHistory;
        });
        
        document.getElementById('save-btn').addEventListener('click', function() {
            if (textHistory) {
                fetch('/save_and_exit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({text: textHistory}),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.href = '/exit';
                    }
                });
            }
        });

        // Training mode buttons
        document.getElementById('train-btn').addEventListener('click', function() {
            trainingContainer.style.display = 'block';
            socket.emit('start_training');
        });

        document.getElementById('capture-btn').addEventListener('click', function() {
            const selectedLetter = alphabetSelector.value;
            if (selectedLetter) {
                socket.emit('capture_sample', { letter: selectedLetter });
                currentlyTrainingLetter = selectedLetter;
            } else {
                trainingStatus.textContent = 'Please select an alphabet first';
            }
        });

        document.getElementById('stop-training-btn').addEventListener('click', function() {
            trainingContainer.style.display = 'none';
            socket.emit('stop_training');
            currentlyTrainingLetter = null;
            trainingStatus.textContent = 'Training stopped';
        });
    </script>
</body>
</html>
''')

    if not os.path.exists('templates/exit.html'):
        with open('templates/exit.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Automated AI-Based Sign Language Translator - Exit</title>
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
        <p>Thank you for using Automated AI-Based Sign Language Translator.</p>
        <a href="/" class="button">Start New Session</a>
    </div>
</body>
</html>
''')

def main():
    print("Setting up project files...")
    create_directories()
    create_cvfpscalc()
    create_keypoint_classifier()
    create_keypoint_classifier_label()
    create_templates()
    print("Setup complete!")

if __name__ == "__main__":
    main() 